#!/usr/bin/env python3
"""
Aggregate run reports into one tidy CSV — the single input for every figure.

The measurement campaign produces one JSON per run. Figures, tables and the
results notes must all come from the same place, or they drift. This script
walks the reports, flattens each into one row, and writes a CSV whose columns
are stable enough to be the artefact that ships with the paper.

    runs (JSON) ──► aggregate_results.py ──► results.csv ──► figures / tables

Schema v2 reports (with a `run` provenance block) are read in full. Older v1
reports, including the `combined_*.json` produced by the A+C runner, are read
best-effort so historical runs still appear; their missing provenance columns
are left empty rather than guessed.

Usage:
    aggregate_results.py                       # every experiment, CSV to stdout
    aggregate_results.py --out results.csv
    aggregate_results.py --summary             # medians across repetitions
    aggregate_results.py path/to/results ...   # explicit roots
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_HERE = Path(__file__).resolve()
for _src in (_HERE.parents[2] / "app" / "src", _HERE.parents[1] / "src"):
    if (_src / "aspir_unet").is_dir():
        sys.path.insert(0, str(_src))
        break

from aspir_unet import experiments
from aspir_unet.measurement.energy import _is_board_input_label

LEGACY_ROUTES = {"A_neon": "cpu_neon", "C_dpu": "dpu_int8", "B_dpu": "dpu_int8"}

COLUMNS = [
    # identity
    "route", "device", "experiment", "mode", "precision", "threads", "timing_frames",
    "repetition", "tag", "timestamp", "host", "schema_version",
    # dataset / model
    "n_frames", "n_measured", "warmup", "normalisation",
    "model_name", "model_sha256", "dataset_sha256", "dpu_fingerprint",
    # latency
    "lat_mean_ms", "lat_median_ms", "lat_p95_ms", "lat_p99_ms",
    "fps_from_latency", "tail_p99_over_median",
    # boundary decomposition: host-side conversion (DPU) or transfers (GPU)
    "t_transfer_in_ms", "t_execute_ms", "t_transfer_out_ms",
    "int8_layer_fraction",
    # sustained throughput and energy
    "fps_sustained", "p_idle_w", "p_active_w", "p_dynamic_w", "board_input_recomputed",
    "e_total_mj", "e_dynamic_mj", "frames_per_joule",
    # quality
    "psnr_gt_db", "ssim_gt", "mae_gt", "psnr_vs_fp32_db", "ssim_vs_fp32",
    # pipeline budget (reconstruction runs)
    "m_patterns", "sampling_ratio_pct", "t_acq_ms", "t_rec_ms", "t_inf_ms",
    "t_tot_ms", "fps_serial", "fps_pipelined", "bottleneck",
    # environment
    "cpu_governor", "cpu_pinned", "cpu_dvfs", "gpu_pinned", "gpu_mhz",
    "temp_start_max_c", "temp_end_max_c", "temp_run_max_c",
    "git_sha", "git_dirty",
    # provenance of the row itself
    "source_file",
]

SUMMARY_METRICS = [
    "lat_mean_ms", "lat_median_ms", "lat_p95_ms", "lat_p99_ms",
    "fps_sustained", "p_active_w", "e_total_mj", "e_dynamic_mj",
    "frames_per_joule", "psnr_gt_db", "ssim_gt", "psnr_vs_fp32_db",
    "t_rec_ms",
]


def _board_input(phase: dict) -> Optional[float]:
    """Board input power for one phase, summed from the recorded rail averages."""
    rails = phase.get("per_rail_w") or {}
    labels = [label for label in rails if _is_board_input_label(label)]
    return sum(rails[label] for label in labels) if labels else None


def _board_energy(energy_mode: dict) -> tuple:
    """(p_idle, p_active, p_dynamic, e_total_j, e_dynamic_j, recomputed).

    The per-rail averages are the measurement; which rails make up "board
    input" is arithmetic done afterwards, and it can be wrong. Between the PMIC
    rename (f7028de) and the fix of the input-rail filter, the Ultra96 sampler
    summed the two IRPS5401 front-end rails into board input: 5.73 W at idle
    where ir38060 alone reads 4.97 W. Recomputing from the rails with the
    current filter corrects those reports and leaves every other one as it was.
    """
    idle = energy_mode.get("idle") or {}
    active = energy_mode.get("active") or {}
    derived = energy_mode.get("derived") or {}
    rec = (idle.get("board_input_w"), active.get("board_input_w"),
           derived.get("board_dynamic_w"), derived.get("energy_per_frame_j"),
           derived.get("dynamic_energy_per_frame_j"), False)
    p_idle, p_act = _board_input(idle), _board_input(active)
    t_ms = derived.get("t_per_frame_ms")
    if p_idle is None or p_act is None or not t_ms:
        return rec
    # Reports round board_input_w; only a real disagreement is a correction.
    if (rec[0] is not None and abs(p_idle - rec[0]) <= 0.02
            and rec[1] is not None and abs(p_act - rec[1]) <= 0.02):
        return rec
    t_s = t_ms / 1e3
    return p_idle, p_act, p_act - p_idle, p_act * t_s, (p_act - p_idle) * t_s, True


def _g(d: Optional[dict], *keys, default=None):
    """Nested get that tolerates missing intermediate dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _max_temp(temps: Optional[dict]) -> Optional[float]:
    if not isinstance(temps, dict) or not temps:
        return None
    values = [v for v in temps.values() if isinstance(v, (int, float))]
    return round(max(values), 2) if values else None


def _max_temp_report(temp_report: Optional[dict]) -> Optional[float]:
    per = _g(temp_report, "per_sensor")
    if not isinstance(per, dict) or not per:
        return None
    values = [v.get("max_c") for v in per.values() if isinstance(v, dict)]
    values = [v for v in values if isinstance(v, (int, float))]
    return round(max(values), 2) if values else None


def _experiment_from_path(npz: Optional[str]) -> Optional[str]:
    """`.../models/<experiment>/source/test_images.npz` -> `<experiment>`."""
    if not npz:
        return None
    parts = Path(npz).parts
    if "source" in parts:
        i = parts.index("source")
        if i > 0:
            return parts[i - 1]
    return None


def _experiment_from_source(source: Path) -> Optional[str]:
    """Fall back to the report's own location.

    Host layout   `.../models/<experiment>/results/<report>.json`
    Board layout  `.../results/<experiment>/<report>.json`

    Needed for legacy reports that never recorded which experiment they ran,
    in particular the combined A+C files whose embedded routes carry no
    dataset block of their own.
    """
    parts = source.resolve().parts
    if "models" in parts:
        i = parts.index("models")
        if i + 1 < len(parts):
            return parts[i + 1]
    if len(parts) >= 3 and parts[-3] == "results":
        return parts[-2]
    return None


def flatten(payload: dict, source: Path, route_key: Optional[str] = None) -> dict:
    """One report (or one route inside a combined report) -> one CSV row."""
    run = payload.get("run") or {}
    params = run.get("params") or {}
    route = payload.get("route_name") or run.get("route")
    legacy_route = payload.get("route")
    if route is None:
        route = LEGACY_ROUTES.get(legacy_route or route_key or "", legacy_route)

    timing = payload.get("timing") or {}
    energy_mode = payload.get("energy_mode") or {}
    idle = energy_mode.get("idle") or {}
    active = energy_mode.get("active") or {}
    derived = energy_mode.get("derived") or {}
    budget = payload.get("pipeline_budget") or {}

    (p_idle_w, p_active_w, p_dynamic_w, e_total_j, e_dyn_j,
     board_recomputed) = _board_energy(energy_mode)

    median = timing.get("median_ms")
    p99 = timing.get("p99_ms")

    experiment = (_g(run, "experiment", "experiment_id")
                  or _experiment_from_path(_g(payload, "dataset", "npz"))
                  or _experiment_from_source(source))

    breakdown = payload.get("timing_breakdown") or {}

    def part(*names):
        for name in names:
            entry = breakdown.get(name)
            if isinstance(entry, dict) and entry.get("median_ms") is not None:
                return entry["median_ms"]
        return None

    engine_info = _g(run, "accelerator", "engine_info") or {}
    counts = engine_info.get("layer_precision_counts") or {}
    total_layers = sum(counts.values()) if counts else 0
    int8_layers = sum(v for k, v in counts.items() if "int8" in str(k).lower())
    int8_fraction = (round(int8_layers / total_layers, 3)
                     if total_layers else None)

    row = {
        "route": route,
        "device": _g(run, "device", "device"),
        # An older CPU run predates the --precision flag and left the field
        # empty. That is a labelling gap, not a different measurement: the
        # route's default is and was FP32. Filling it keeps repetitions of one
        # configuration in one group instead of splitting them.
        "precision": (payload.get("precision") or params.get("precision")
                      or ("fp32" if route == "cpu_neon" else None)),
        "t_transfer_in_ms": part("h2d_ms", "convert_in_ms"),
        "t_execute_ms": part("execute_ms"),
        "t_transfer_out_ms": part("d2h_ms", "convert_out_ms"),
        "int8_layer_fraction": int8_fraction,
        "experiment": experiment,
        "mode": payload.get("mode") or run.get("mode")
                or ("energy" if energy_mode else "latency"),
        "threads": params.get("threads")
                   or _g(payload, "backend", "intra_op_threads"),
        "repetition": run.get("repetition"),
        "tag": run.get("tag"),
        "timestamp": payload.get("timestamp"),
        "host": payload.get("host") or _g(run, "host", "hostname"),
        "schema_version": payload.get("schema_version", 1),

        "n_frames": _g(payload, "dataset", "n_frames") or payload.get("n_frames"),
        "n_measured": timing.get("n_measured"),
        "warmup": timing.get("n_warmup", params.get("warmup")),
        "normalisation": params.get("normalisation")
                         or _g(payload, "backend", "normalisation"),
        "model_name": _g(run, "artifacts", "model", "name")
                      or _g(payload, "backend", "model"),
        "model_sha256": _g(run, "artifacts", "model", "sha256"),
        "dataset_sha256": _g(run, "artifacts", "dataset", "sha256"),
        "dpu_fingerprint": _g(run, "accelerator", "dpu_fingerprint"),

        "lat_mean_ms": timing.get("mean_ms"),
        "lat_median_ms": median,
        "lat_p95_ms": timing.get("p95_ms"),
        "lat_p99_ms": p99,
        "fps_from_latency": timing.get("fps"),
        "tail_p99_over_median": (round(p99 / median, 3)
                                 if median and p99 else None),

        "fps_sustained": active.get("fps_sustained"),
        "p_idle_w": p_idle_w,
        "p_active_w": p_active_w,
        "p_dynamic_w": p_dynamic_w,
        "board_input_recomputed": board_recomputed,
        "e_total_mj": round(e_total_j * 1e3, 4) if e_total_j else None,
        "e_dynamic_mj": round(e_dyn_j * 1e3, 4) if e_dyn_j is not None else None,
        "frames_per_joule": (round(1.0 / e_total_j, 2)
                             if e_total_j else None),

        "psnr_gt_db": _g(payload, "metrics_vs_originals", "psnr_mean_db"),
        "ssim_gt": _g(payload, "metrics_vs_originals", "ssim_mean"),
        "mae_gt": _g(payload, "metrics_vs_originals", "mae_mean"),
        "psnr_vs_fp32_db": _g(payload, "metrics_vs_pt_gold", "psnr_mean_db"),
        "ssim_vs_fp32": _g(payload, "metrics_vs_pt_gold", "ssim_mean"),

        "m_patterns": budget.get("m_patterns"),
        "sampling_ratio_pct": budget.get("sampling_ratio_pct"),
        "t_acq_ms": budget.get("t_acq_ms"),
        "t_rec_ms": budget.get("t_rec_ms"),
        "t_inf_ms": budget.get("t_inf_ms"),
        "t_tot_ms": budget.get("t_tot_ms"),
        "fps_serial": budget.get("fps_serial"),
        "fps_pipelined": budget.get("fps_pipelined"),
        "bottleneck": budget.get("bottleneck"),

        "cpu_governor": ";".join(_g(run, "host", "cpu_governor") or []) or None,
        # A run measured while the clocks were free to float is not comparable
        # with one measured pinned: on the Orin the same engine varied by more
        # than 2x between the two states. Reports written before the clock
        # state was recorded leave these empty, which is itself the signal.
        # How many frames the latency pass timed. An 80-frame quality loop and
        # a 10,000-frame timing pass are different measurements of the same
        # configuration and must not share a median.
        "timing_frames": _g(run, "params", "timing_frames"),
        "cpu_pinned": _g(run, "clocks", "cpu_pinned"),
        # False means the platform has no cpufreq at all, so the clock cannot
        # float. Absent cpu_pinned with cpu_dvfs False is a fixed clock, not an
        # unpinned one.
        "cpu_dvfs": _g(run, "clocks", "cpu_dvfs"),
        "gpu_pinned": _g(run, "clocks", "gpu_pinned"),
        "gpu_mhz": (max((_g(run, "clocks", "gpu_cur_hz") or {}).values()) // 10**6
                    if (_g(run, "clocks", "gpu_cur_hz") or {}) else None),
        "temp_start_max_c": _max_temp(run.get("thermal_start_c")),
        "temp_end_max_c": _max_temp(run.get("thermal_end_c")),
        "temp_run_max_c": _max_temp_report(payload.get("temperature")),
        "git_sha": _g(run, "code", "git_sha"),
        "git_dirty": _g(run, "code", "dirty"),

        "source_file": str(source),
    }
    return row


def rows_from_file(path: Path) -> List[dict]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"[warn] unreadable {path}: {exc}", file=sys.stderr)
        return []
    if not isinstance(payload, dict):
        return []

    # A sweep log, not a run report.
    if "runs" in payload and "matrix_file" in payload:
        return []

    # Combined A+C report (schema v1): one row per embedded route.
    combined = [(k, v) for k, v in payload.items()
                if k.startswith("route_") and isinstance(v, dict)]
    if combined and "timing" not in payload:
        out = []
        for key, sub in combined:
            merged = dict(sub)
            merged.setdefault("timestamp", payload.get("timestamp"))
            merged.setdefault("host", payload.get("host"))
            merged["route"] = key.replace("route_", "")
            out.append(flatten(merged, path, route_key=merged["route"]))
        return out

    if "timing" not in payload and "energy_mode" not in payload:
        return []
    return [flatten(payload, path)]


def discover_roots(layout) -> List[Path]:
    """Every results directory known to this machine."""
    roots: List[Path] = []
    for exp_id in experiments.list_experiments(layout):
        exp = experiments.resolve(exp_id, __file__, layout=layout)
        if exp.results_dir.is_dir():
            roots.append(exp.results_dir)
    if layout.is_board and layout.deploy_root:
        legacy = layout.deploy_root / "results"
        if legacy.is_dir():
            roots.append(legacy)
    return roots


def collect_rows(roots: Iterable[Path]) -> List[dict]:
    rows: List[dict] = []
    seen: set = set()
    for root in roots:
        for path in sorted(Path(root).rglob("*.json")):
            if path in seen:
                continue
            seen.add(path)
            rows.extend(rows_from_file(path))
    rows.sort(key=lambda r: (str(r.get("experiment") or ""),
                             str(r.get("route") or ""),
                             str(r.get("mode") or ""),
                             str(r.get("threads") or ""),
                             str(r.get("timestamp") or "")))
    return rows


def summarise(rows: List[dict]) -> List[dict]:
    """Median across repetitions of one configuration, with the spread."""
    # `host` is part of the key on purpose: a host-side smoke run and a
    # board run of the same configuration must never end up in one median.
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        # `tag` is part of the key because it is how a matrix marks a run as a
        # deliberate variant of an otherwise identical configuration -- the
        # PL-loaded energy control differs from its pair in nothing else.
        key = (r.get("experiment"), r.get("route"), r.get("mode"),
               r.get("precision"), r.get("threads"), r.get("device"),
               r.get("host"), r.get("timing_frames"), r.get("tag"))
        groups.setdefault(key, []).append(r)

    out = []
    for (experiment, route, mode, precision, threads, device, host,
         timing_frames, tag), items in sorted(
            groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = {
            "experiment": experiment, "route": route, "device": device,
            "mode": mode, "precision": precision, "threads": threads,
            "host": host, "timing_frames": timing_frames, "tag": tag,
            "n_runs": len(items),
        }
        for metric in SUMMARY_METRICS:
            values = [r[metric] for r in items
                      if isinstance(r.get(metric), (int, float))]
            if not values:
                continue
            row[f"{metric}_median"] = round(statistics.median(values), 4)
            if len(values) > 1:
                row[f"{metric}_min"] = round(min(values), 4)
                row[f"{metric}_max"] = round(max(values), 4)
        out.append(row)
    return out


def check_pairing(rows: List[dict]) -> List[str]:
    """Every route and device measuring one grid point must share its inputs.

    The cross-device comparison is only meaningful if the CPU, the DPU and the
    GPU ran the *same* frames through the *same* trained network. Reports carry
    the sha256 of the dataset they loaded, so that can be checked instead of
    assumed: a regenerated npz or a re-exported checkpoint shows up here rather
    than being silently averaged into a table.
    """
    by_experiment: Dict[str, Dict[str, set]] = {}
    for r in rows:
        exp = r.get("experiment")
        if not exp:
            continue
        slot = by_experiment.setdefault(exp, {"dataset": set(), "sources": {}})
        sha = r.get("dataset_sha256")
        if sha:
            slot["dataset"].add(sha)
            slot["sources"].setdefault(sha, []).append(
                f"{r.get('device') or '?'}/{r.get('route')}"
            )

    problems = []
    for exp, slot in sorted(by_experiment.items()):
        if len(slot["dataset"]) > 1:
            detail = "; ".join(
                f"{sha[:12]} <- {', '.join(sorted(set(who)))}"
                for sha, who in sorted(slot["sources"].items())
            )
            problems.append(
                f"{exp}: {len(slot['dataset'])} different datasets across runs "
                f"({detail})"
            )
    return problems


def write_csv(rows: List[dict], columns: List[str], out: Optional[Path]) -> None:
    handle = out.open("w", newline="") if out else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if out:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", type=Path,
                        help="Directories to scan (default: every experiment's "
                             "results directory).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write CSV here instead of stdout.")
    parser.add_argument("--summary", action="store_true",
                        help="Emit medians across repetitions per configuration.")
    parser.add_argument("--min-schema", type=int, default=0,
                        help="Drop reports older than this schema version.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if the pairing check finds runs of "
                             "one experiment that used different inputs.")
    args = parser.parse_args()

    layout = experiments.discover_layout(__file__)
    roots = args.roots or discover_roots(layout)
    if not roots:
        print("[error] no results directories found", file=sys.stderr)
        return 1

    rows = collect_rows(roots)
    if args.min_schema:
        rows = [r for r in rows if (r.get("schema_version") or 1) >= args.min_schema]
    if not rows:
        print("[error] no run reports found", file=sys.stderr)
        return 1

    if args.summary:
        summary = summarise(rows)
        columns: List[str] = []
        for row in summary:
            for k in row:
                if k not in columns:
                    columns.append(k)
        write_csv(summary, columns, args.out)
    else:
        write_csv(rows, COLUMNS, args.out)

    print(f"[aggregate] {len(rows)} runs from {len(roots)} directories",
          file=sys.stderr)
    if args.out:
        print(f"[aggregate] wrote {args.out}", file=sys.stderr)

    problems = check_pairing(rows)
    for line in problems:
        print(f"[pairing] {line}", file=sys.stderr)
    if problems:
        print(f"[pairing] {len(problems)} experiment(s) with mismatched inputs",
              file=sys.stderr)
        if args.strict:
            return 3
    elif rows:
        print("[pairing] all runs of each experiment share their inputs",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
