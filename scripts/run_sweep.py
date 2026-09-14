#!/usr/bin/env python3
"""
Sweep driver — execute a configuration matrix and keep track of what ran.

The measurement campaign is dozens of runs: routes x experiments x thread
counts x modes x repetitions. Launching those by hand is how runs end up
missing, duplicated, or executed with a parameter nobody wrote down. This
driver takes a matrix file, expands it, executes each run as a **separate
process** (independent repetitions are part of the protocol, and it isolates
runtime state between runs), and writes a sweep log recording what happened.

Matrix file (JSON):

    {
      "name": "e3_thread_scaling",
      "description": "CPU thread scaling at one sampling ratio",
      "defaults": {"repetitions": 3, "warmup": 20, "limit": 0, "dwell_s": 15},
      "matrix": [
        {"route": "cpu_neon",  "experiments": ["celeb_..._8pct"],
         "modes": ["latency"], "threads": [1, 2, 4]},
        {"route": "dpu_int8",  "experiments": ["celeb_..._8pct"],
         "modes": ["latency", "energy"]},
        {"route": "cpu_recon", "experiments": ["celeb_..._8pct"],
         "threads": [4], "norm": ["global"]}
      ]
    }

Plural keys (`experiments`, `modes`, `threads`, `norm`) expand as a cartesian
product; singular keys are passed through unchanged. Unknown keys for a route
are an error rather than a silent no-op, because a typo in a knob would
otherwise produce a full campaign of subtly wrong runs.

Re-running a sweep skips configurations that already have a report (match by
the deterministic report name), so an interrupted overnight campaign resumes
where it stopped. `--force` re-measures everything.
"""

from __future__ import annotations

import argparse
import itertools
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve()
for _src in (_HERE.parents[2] / "app" / "src", _HERE.parents[1] / "src"):
    if (_src / "aspir_unet").is_dir():
        sys.path.insert(0, str(_src))
        break

from aspir_unet import experiments, provenance
from aspir_unet.bench import TIMING_FRAMES

SCRIPTS = {
    "cpu_neon": "run_cpu_ort.py",
    "dpu_int8": "run_dpu_int8.py",
    "gpu_trt": "run_gpu_trt.py",
    "cpu_recon": "run_recon_timing.py",
}

# Per-route: which config keys are accepted and how they map to CLI flags.
# `mode` is handled separately (it selects --energy-mode on the two inference
# routes and is fixed to "recon" on the reconstruction route).
FLAGS: Dict[str, Dict[str, str]] = {
    "cpu_neon": {
        "precision": "--precision",
        "threads": "--threads", "warmup": "--warmup", "limit": "--limit",
        "timing_frames": "--timing-frames",
        "thermal_warmup_s": "--thermal-warmup-s",
        "energy_idle_s": "--energy-idle-s", "energy_active_s": "--energy-active-s",
        "tag": "--tag", "out": "--out", "npz": "--npz", "onnx": "--onnx",
    },
    "dpu_int8": {
        "warmup": "--warmup", "limit": "--limit",
        "timing_frames": "--timing-frames",
        "thermal_warmup_s": "--thermal-warmup-s",
        "energy_idle_s": "--energy-idle-s", "energy_active_s": "--energy-active-s",
        "tag": "--tag", "out": "--out", "npz": "--npz", "xmodel": "--xmodel",
    },
    "gpu_trt": {
        "precision": "--precision", "warmup": "--warmup", "limit": "--limit",
        "timing_frames": "--timing-frames",
        "thermal_warmup_s": "--thermal-warmup-s",
        "energy_idle_s": "--energy-idle-s", "energy_active_s": "--energy-active-s",
        "tag": "--tag", "out": "--out", "npz": "--npz", "engine": "--engine",
    },
    "cpu_recon": {
        "threads": "--threads", "warmup": "--warmup", "limit": "--limit",
        "energy_idle_s": "--energy-idle-s", "energy_active_s": "--energy-active-s",
        "thermal_warmup_s": "--thermal-warmup-s",
        "norm": "--norm", "dmd_khz": "--dmd-khz", "t_inf_ms": "--t-inf-ms",
        "tag": "--tag", "out": "--out", "npz": "--npz", "masks": "--masks",
    },
}

# Boolean switches, per route: passing one to a route that does not define it
# would be an argparse error, so they are declared per route rather than shared.
BOOL_FLAGS: Dict[str, Dict[str, str]] = {
    "cpu_neon": {"save_outputs": "--save-outputs",
                 "no_unload_pl": "--no-unload-pl"},
    "dpu_int8": {"save_outputs": "--save-outputs",
                 "no_load_pl": "--no-load-pl"},
    "gpu_trt": {"save_outputs": "--save-outputs"},
    "cpu_recon": {},
}

PLURAL = {
    "experiments": "experiment",
    "modes": "mode",
    "threads": "threads",
    "precisions": "precision",
    "norm": "norm",
    "limits": "limit",
}

# Keys that configure the sweep itself rather than a run.
SWEEP_KEYS = {"route", "repetitions", "dwell_s"}


def expand(entry: dict, defaults: dict) -> List[dict]:
    """Expand one matrix entry into concrete run configurations."""
    route = entry.get("route")
    if route not in SCRIPTS:
        raise ValueError(f"unknown route '{route}'; known: {sorted(SCRIPTS)}")

    known = set(FLAGS[route]) | set(BOOL_FLAGS[route]) | {"mode", "experiment"}
    # Defaults apply where they are applicable: a default meant for the CPU
    # route must not break a matrix entry for a route that has no such knob.
    base = {k: v for k, v in defaults.items() if k in known}
    # A default that applies to *no* route is a typo or a knob that was never
    # wired up, and dropping it silently means the campaign runs with settings
    # nobody chose — which is how a quality sweep ends up paying for a
    # 10,000-frame timing pass it explicitly switched off.
    every_known = set().union(*(set(FLAGS[r]) | set(BOOL_FLAGS[r]) for r in FLAGS))
    for key in defaults:
        if key not in every_known and key not in SWEEP_KEYS and key != "mode":
            print(f"[sweep] WARNING default '{key}' is accepted by no route "
                  f"and will be ignored")
    fixed = {}
    axes: Dict[str, list] = {}

    for key, value in entry.items():
        if key in SWEEP_KEYS:
            continue
        if key in PLURAL:
            axis = PLURAL[key]
            axes[axis] = list(value) if isinstance(value, (list, tuple)) else [value]
        else:
            fixed[key] = value

    for key in list(fixed) + list(axes):
        if key not in known:
            raise ValueError(
                f"route '{route}' does not accept '{key}'. "
                f"Accepted: {sorted(known)}"
            )

    if "experiment" not in axes and "experiment" not in fixed:
        raise ValueError(f"route '{route}': no experiment given")
    if route == "cpu_recon":
        axes.setdefault("mode", ["recon"])
    elif "mode" not in axes and "mode" not in fixed:
        axes["mode"] = ["latency"]

    names = sorted(axes)
    combos = itertools.product(*(axes[n] for n in names)) if names else [()]
    out = []
    for combo in combos:
        cfg = dict(base)
        cfg.update(fixed)
        cfg.update(dict(zip(names, combo)))
        cfg["route"] = route
        cfg["repetitions"] = int(entry.get("repetitions",
                                           defaults.get("repetitions", 1)))
        out.append(cfg)
    return out


def build_argv(cfg: dict, rep: int, scripts_dir: Path, python: str) -> List[str]:
    route = cfg["route"]
    argv = [python, str(scripts_dir / SCRIPTS[route]),
            "--experiment", str(cfg["experiment"]),
            "--rep", str(rep)]
    for key, flag in FLAGS[route].items():
        if key in cfg and cfg[key] is not None:
            argv += [flag, str(cfg[key])]
    for key, flag in BOOL_FLAGS[route].items():
        if cfg.get(key):
            argv.append(flag)
    if route in ("cpu_neon", "dpu_int8", "gpu_trt", "cpu_recon") and cfg.get("mode") == "energy":
        argv.append("--energy-mode")
    return argv


# Parameters that change *what a run measures*, and so must match before an
# existing report can stand in for a planned run. The filename carries route,
# mode, experiment, repetition, threads and precision; everything here is
# invisible to it. Leaving them out silently accepted E1's 80-frame quality
# report in place of E2's 10,000-frame timing pass, and let an energy run with
# the bitstream unloaded stand in for the control that measures what a loaded
# bitstream costs -- the exact opposite of the intended measurement.
#
# The value is the runner's own default, used when the matrix does not set the
# key, so a matrix that omits it still compares against what would actually run.
DECISIVE_PARAMS: Dict[str, Any] = {
    # precision is NOT in the report filename for the CPU route, so an INT8
    # control run and an FP32 baseline share a name and only the recorded
    # parameter tells them apart.
    "precision": "fp32",
    "threads": None,
    "timing_frames": TIMING_FRAMES,
    "energy_idle_s": None,
    "energy_active_s": None,
    "thermal_warmup_s": None,
    "norm": None,
    "limit": None,
    "warmup": None,
}


def _measures_the_same(cfg: dict, report: Path, rep: int) -> bool:
    """Whether an existing report measured what this planned run would measure."""
    try:
        rec = json.loads(report.read_text()).get("run", {})
    except (OSError, ValueError):
        return False                      # unreadable: re-measure rather than trust
    params = rec.get("params") or {}

    if (cfg.get("tag") or None) != (rec.get("tag") or None):
        return False

    # The CPU route unloads the PL before measuring; the matrix expresses that
    # as no_unload_pl, the report records the resolved unload_pl.
    if "unload_pl" in params:
        if params["unload_pl"] != (not cfg.get("no_unload_pl", False)):
            return False

    for key, default in DECISIVE_PARAMS.items():
        if key not in params:
            continue
        planned = cfg.get(key, default)
        if planned is None:
            continue
        if params[key] != planned:
            return False
    return True


def existing_report(cfg: dict, rep: int, layout) -> Optional[Path]:
    """The report of this exact configuration, if it has already been measured."""
    exp = experiments.resolve(cfg["experiment"], __file__, layout=layout)
    out_dir = Path(cfg["out"]) if cfg.get("out") else exp.results_dir
    if not out_dir.is_dir():
        return None
    pattern = provenance.report_glob(
        cfg["route"], cfg.get("mode", "latency"), exp.experiment_id, rep,
        suffix=provenance.run_suffix(cfg["route"], cfg),
    )
    for hit in sorted(out_dir.glob(pattern), reverse=True):
        if _measures_the_same(cfg, hit, rep):
            return hit
    return None


def describe(cfg: dict, rep: int) -> str:
    bits = [cfg["route"], cfg.get("mode", "latency"), cfg["experiment"]]
    if cfg.get("threads"):
        bits.append(f"{cfg['threads']}t")
    if cfg.get("precision"):
        bits.append(str(cfg["precision"]))
    bits.append(f"rep{rep}")
    return " · ".join(str(b) for b in bits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path, help="Sweep matrix JSON file.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the expanded plan and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Re-measure configurations that already have a report.")
    parser.add_argument("--keep-going", action="store_true",
                        help="Continue after a failing run instead of stopping.")
    parser.add_argument("--only", default=None,
                        help="Substring filter on the run description.")
    parser.add_argument("--dwell-s", type=float, default=None,
                        help="Seconds to idle between runs (thermal settling); "
                             "overrides the matrix default.")
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter used for the runs.")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Where to write the sweep log (default: alongside "
                             "the results of the first experiment).")
    args = parser.parse_args()

    spec = json.loads(args.matrix.read_text())
    defaults = spec.get("defaults", {})
    layout = experiments.discover_layout(__file__)
    scripts_dir = _HERE.parent

    plan: List[tuple] = []
    for entry in spec.get("matrix", []):
        for cfg in expand(entry, defaults):
            for rep in range(cfg["repetitions"]):
                desc = describe(cfg, rep)
                if args.only and args.only not in desc:
                    continue
                plan.append((cfg, rep, desc))

    dwell = args.dwell_s if args.dwell_s is not None else defaults.get("dwell_s", 0)

    # Validate every experiment up front: a matrix naming an experiment that
    # has not been imported yet should say so once, plainly, not raise on the
    # first run of an overnight campaign.
    missing = []
    for exp_id in sorted({cfg["experiment"] for cfg, _, _ in plan}):
        try:
            experiments.resolve(exp_id, __file__, layout=layout)
        except (FileNotFoundError, ValueError) as exc:
            missing.append(f"  {exp_id}: {exc}")
    if missing:
        print(f"[sweep] {len(missing)} experiment(s) cannot be resolved:")
        for line in missing:
            print(line)
        print(f"[sweep] known experiments: "
              f"{', '.join(experiments.list_experiments(layout)) or '(none)'}")
        return 2

    print(f"[sweep] {spec.get('name', args.matrix.stem)}: {len(plan)} runs")
    if spec.get("description"):
        print(f"[sweep] {spec['description']}")

    todo = []
    for cfg, rep, desc in plan:
        done = None if args.force else existing_report(cfg, rep, layout)
        status = "skip (done)" if done else "run"
        print(f"  [{status:11s}] {desc}")
        if not done:
            todo.append((cfg, rep, desc))

    if args.dry_run:
        for cfg, rep, desc in todo:
            print("  $", " ".join(build_argv(cfg, rep, scripts_dir, args.python)))
        return 0

    if not todo:
        print("[sweep] nothing to do")
        return 0

    started = datetime.now().isoformat(timespec="seconds")
    log = {
        "sweep": spec.get("name", args.matrix.stem),
        "matrix_file": str(args.matrix),
        "started_at": started,
        "host": socket.gethostname(),
        "runs": [],
    }

    failures = 0
    for i, (cfg, rep, desc) in enumerate(todo, 1):
        argv = build_argv(cfg, rep, scripts_dir, args.python)
        print(f"\n[sweep] ({i}/{len(todo)}) {desc}")
        print("[sweep] $", " ".join(argv))
        t0 = time.perf_counter()
        proc = subprocess.run(argv)
        elapsed = time.perf_counter() - t0
        report = existing_report(cfg, rep, layout)
        entry = {
            "description": desc,
            "config": {k: v for k, v in cfg.items() if k != "repetitions"},
            "repetition": rep,
            "argv": argv,
            "returncode": proc.returncode,
            "seconds": round(elapsed, 1),
            "report": str(report) if report else None,
        }
        log["runs"].append(entry)
        if proc.returncode != 0:
            failures += 1
            print(f"[sweep] FAILED rc={proc.returncode}: {desc}")
            if not args.keep_going:
                break
        if dwell and i < len(todo):
            print(f"[sweep] dwell {dwell:.0f}s …")
            time.sleep(dwell)

    log["finished_at"] = datetime.now().isoformat(timespec="seconds")
    log["failures"] = failures

    log_dir = args.log_dir
    if log_dir is None:
        first_exp = experiments.resolve(todo[0][0]["experiment"], __file__,
                                        layout=layout)
        log_dir = first_exp.results_dir / "sweeps"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sweep_{log['sweep']}_{started.replace(':', '-')}.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n[sweep] {len(log['runs'])} runs, {failures} failed")
    print(f"[sweep] log: {log_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
