#!/usr/bin/env python3
"""
Route C (paper name: DPU--INT8) — VART + DPUCZDX8G B512 inference on the
Ultra96-v2's programmable logic.
(formerly "Route B"; the Vitis-flow xclbin route was deprecated and removed)

Two modes via --energy-mode:

  Default (latency): every frame timed individually, hwmon sampled
  alongside, PSNR/SSIM/MAE vs originals and vs the FP32 gold reference
  computed at the end. The energy figures in this pass are unreliable for
  short runs (the DPU completes in ~1 ms — far shorter than a PMBus sweep).

  --energy-mode: idle baseline + sustained inference loop, deriving J/frame
  from the two power averages. This is the paper number.

Which model and dataset run is selected with --experiment; artefacts are
resolved through the experiment registry, so the same command works on the
host repo and on a deployed board. Reports are schema v2, carrying a `run`
block with code identity, artefact hashes, software versions, machine state
and the compiled model's DPU fingerprint.

Boot-time loading: the bitstream is NOT auto-loaded. Run `aspir-load-dpu`
once after each boot so /dev/dpu appears before vart imports.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Locate aspir_unet for both host (REPO/app/src) and board (DEST/src).
_HERE = Path(__file__).resolve()
for _src in (_HERE.parents[2] / "app" / "src", _HERE.parents[1] / "src"):
    if (_src / "aspir_unet").is_dir():
        sys.path.insert(0, str(_src))
        break

from aspir_unet import experiments, provenance
from aspir_unet.bench import (
    TIMING_FRAMES, hostname, measure_energy_mode, measure_timing,
)
from aspir_unet.data import load_test_set
from aspir_unet.inference.dpu import DpuRunner
from aspir_unet.measurement.energy import EnergySampler, TemperatureSampler
from aspir_unet.measurement.metrics import MetricSummary
from aspir_unet.measurement.timing import TimingMeasurer

ROUTE = "dpu_int8"
LEGACY_ROUTE = "C_dpu"           # kept so existing plotting keeps working

DEV_DPU = Path("/dev/dpu")
LOAD_HELPER = "/usr/local/bin/aspir-load-dpu"


def _ensure_pl_loaded() -> bool:
    """Load the DPU overlay when it is missing, and say whether we did.

    Route A unloads the PL before measuring, so any sweep that interleaves the
    two routes leaves the accelerator unconfigured for the next DPU run. VART
    would then fail at runner creation. Loading here keeps an unattended
    campaign running; the report records whether the overlay had to be loaded,
    since a fresh load is a slightly different starting state.
    """
    if DEV_DPU.exists():
        return False
    print(f"[pl] {DEV_DPU} missing, loading the DPU overlay …")
    try:
        r = subprocess.run([LOAD_HELPER], capture_output=True, text=True,
                           timeout=60)
    except FileNotFoundError:
        print(f"[pl] NOTE {LOAD_HELPER} not installed (off-board run?)")
        return False
    except subprocess.TimeoutExpired:
        print("[pl] WARNING aspir-load-dpu timed out")
        return False
    if r.returncode != 0:
        print(f"[pl] WARNING aspir-load-dpu rc={r.returncode}: "
              f"{(r.stderr or r.stdout).strip()}")
        return False
    print(f"[pl] {r.stdout.strip() or 'overlay loaded'}")
    return DEV_DPU.exists()


def _bench_per_frame(runner: DpuRunner, ts, n: int, warmup: int) -> tuple:
    preds = np.empty_like(ts.inputs[:n])
    timer = TimingMeasurer(n_total=n, warmup=warmup)
    energy = EnergySampler()
    temp = TemperatureSampler()
    parts = {"convert_in_ms": [], "execute_ms": [], "convert_out_ms": []}
    print(f"[bench] hwmon: energy_rails={len(energy.rail_labels)} "
          f"temp_sensors={len(temp.sensor_labels)}")
    with energy, temp:
        for i in timer.indices():
            t0 = timer.tic()
            preds[i] = runner.run_one(ts.inputs[i:i + 1])[0]
            timer.toc(t0)
            if i >= warmup:
                for k in parts:
                    parts[k].append(runner.last_breakdown.get(k, 0.0))
    timing = timer.report()
    energy_report = energy.report()
    temp_report = temp.report()

    print("[metrics] computing PSNR / SSIM / MAE …")
    vs_gt = MetricSummary.from_pair(preds, ts.ground_truth[:n],
                                    keep_per_frame=True)
    vs_pt = MetricSummary.from_pair(preds, ts.gold_pt[:n],
                                    keep_per_frame=True)

    def stats(values):
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return {"mean_ms": round(float(arr.mean()), 4),
                "median_ms": round(float(np.median(arr)), 4)}

    payload = {
        "timing": timing.to_dict(),
        "timing_breakdown": {k: stats(v) for k, v in parts.items()},
        "energy": energy_report.to_dict(),
        "temperature": temp_report.to_dict(),
        "metrics_vs_originals": vs_gt.to_dict(),
        "metrics_vs_pt_gold": vs_pt.to_dict(),
    }
    per_frame = {}
    for tag, summary in (("gt", vs_gt), ("fp32", vs_pt)):
        for key, values in (summary.per_frame or {}).items():
            per_frame[f"{key}_vs_{tag}"] = values
    return payload, preds, per_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None,
                        help="Experiment id under models/. Defaults to "
                             f"'{experiments.LEGACY_EXPERIMENT_ID}'.")
    parser.add_argument("--list-experiments", action="store_true",
                        help="Print the available experiment ids and exit.")
    parser.add_argument("--xmodel", type=Path, default=None,
                        help="Override the xmodel resolved from the experiment.")
    parser.add_argument("--npz", type=Path, default=None,
                        help="Override the dataset resolved from the experiment.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Override the results directory.")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Frames discarded before timing starts.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--energy-mode", action="store_true",
                        help="idle baseline + sustained-load loop → J/frame")
    parser.add_argument("--energy-idle-s", type=float, default=10.0)
    parser.add_argument("--energy-active-s", type=float, default=30.0)
    parser.add_argument("--timing-frames", type=int, default=TIMING_FRAMES,
                        help="Frames in the dedicated latency pass; the "
                             "evaluation split is cycled to reach it. Must "
                             "be identical on every path for the tail "
                             "quantiles to be comparable. 0 skips it.")
    parser.add_argument("--thermal-warmup-s", type=float, default=60.0,
                        help="Continuous inference before the energy passes "
                             "so baseline and load share a thermal state.")
    parser.add_argument("--rep", type=int, default=0,
                        help="Repetition index within a sweep.")
    parser.add_argument("--tag", default=None,
                        help="Free-form label recorded in the report.")
    parser.add_argument("--save-outputs", action="store_true",
                        help="Write the predicted frames next to the report "
                             "(needed to compare routes frame by frame).")
    parser.add_argument("--no-load-pl", action="store_true",
                        help="Do not load the DPU overlay when /dev/dpu is "
                             "missing; fail instead. Use when the PL state "
                             "must be exactly what the operator left.")
    args = parser.parse_args()

    layout = experiments.discover_layout(__file__)
    if args.list_experiments:
        for name in experiments.list_experiments(layout):
            print(name)
        return 0

    exp = experiments.resolve(args.experiment, __file__, layout=layout)
    pl_loaded_here = False if args.no_load_pl else _ensure_pl_loaded()
    xmodel_path = args.xmodel or exp.require_xmodel()
    npz_path = args.npz or exp.npz
    out_dir = args.out or exp.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp]  {exp.experiment_id}")
    print(f"[load] dataset {npz_path}")
    ts = load_test_set(npz_path)
    n = ts.n if args.limit <= 0 else min(args.limit, ts.n)
    print(f"[load] {n} frames, range "
          f"[{ts.inputs.min():.4f},{ts.inputs.max():.4f}]")

    print(f"[load] xmodel {xmodel_path}")
    runner = DpuRunner(xmodel_path)
    print(f"[load] DPU input  scale=2^{int(np.log2(runner.in_scale))} "
          f"({runner.in_scale}), dims={runner.in_dims}")
    print(f"[load] DPU output scale=2^{int(np.log2(runner.out_scale))} "
          f"({runner.out_scale}), dims={runner.out_dims}")

    mode = "energy" if args.energy_mode else "latency"
    run_block = provenance.collect(
        route=ROUTE,
        mode=mode,
        experiment=exp,
        params={
            "warmup": args.warmup,
            "limit": args.limit,
            "n_frames": int(n),
            "timing_frames": args.timing_frames,
            "thermal_warmup_s": args.thermal_warmup_s,
            "energy_idle_s": args.energy_idle_s,
            "energy_active_s": args.energy_active_s,
            "pl_loaded_by_runner": pl_loaded_here,
        },
        artifacts={"model": xmodel_path, "xmodel": xmodel_path,
                   "dataset": npz_path},
        script_file=__file__,
        repetition=args.rep,
        tag=args.tag,
    )

    timestamp = datetime.now().isoformat(timespec="seconds")
    report: dict = {
        "schema_version": provenance.SCHEMA_VERSION,
        "timestamp": timestamp,
        "host": hostname(),
        "route": LEGACY_ROUTE,
        "route_name": ROUTE,
        "mode": mode,
        "run": run_block,
        "backend": {
            "framework": "vart",
            "model": experiments.rel_to(xmodel_path, layout),
            "in_scale": runner.in_scale,
            "out_scale": runner.out_scale,
            "in_dims": list(runner.in_dims),
            "out_dims": list(runner.out_dims),
        },
        "dataset": {
            "npz": experiments.rel_to(npz_path, layout),
            "n_frames": int(n),
        },
    }

    preds, per_frame, power_series = None, {}, {}
    if args.energy_mode:
        report["energy_mode"] = measure_energy_mode(
            runner.run_one,
            ts.inputs[:n],
            idle_s=args.energy_idle_s,
            active_s=args.energy_active_s,
            thermal_warmup_s=args.thermal_warmup_s,
            series_out=power_series,
            route_name="dpu",
        )
    else:
        payload, preds, per_frame = _bench_per_frame(runner, ts, n, args.warmup)
        report.update(payload)

    # Quality came from one pass over the split; latency comes from a
    # dedicated pass with a fixed frame count, identical on every route.
    timing_samples = None
    if not args.energy_mode and args.timing_frames > 0:
        report["timing_quality_pass"] = report.pop("timing", None)
        t_report, timing_samples, t_parts = measure_timing(
            runner.run_one, ts.inputs[:n],
            n_frames=args.timing_frames, warmup=args.warmup,
            breakdown_of=lambda: runner.last_breakdown,
            route_name="dpu",
        )
        report["timing"] = t_report.to_dict()
        if t_parts:
            report["timing_breakdown"] = t_parts

    runner.close()
    provenance.finalise(run_block)

    name = provenance.report_filename(
        ROUTE, mode, exp.experiment_id, args.rep, timestamp,
    )
    out_json = out_dir / name
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[save] {out_json}")

    if timing_samples is not None:
        samples_path = out_json.with_name(out_json.stem + "_timings.npy")
        np.save(samples_path, timing_samples)
        report["timing_samples_file"] = samples_path.name
        out_json.write_text(json.dumps(report, indent=2))
        print(f"[save] {samples_path}")

    if power_series:
        power_npz = out_json.with_name(out_json.stem + "_power.npz")
        flat = {f"{pass_name}_{k}": np.asarray(v, dtype=np.float32)
                for pass_name, s in power_series.items() for k, v in s.items()}
        np.savez_compressed(power_npz, **flat)
        report["power_series_file"] = power_npz.name
        out_json.write_text(json.dumps(report, indent=2))
        print(f"[save] {power_npz}")

    if per_frame:
        frames_npz = out_json.with_name(out_json.stem + "_frames.npz")
        np.savez_compressed(frames_npz, **per_frame)
        report["per_frame_metrics_file"] = frames_npz.name
        out_json.write_text(json.dumps(report, indent=2))
        print(f"[save] {frames_npz}")

    if args.save_outputs and preds is not None:
        out_npz = out_json.with_name(out_json.stem + "_preds.npz")
        np.savez_compressed(out_npz, predictions=preds)
        print(f"[save] {out_npz}")

    print(json.dumps({k: v for k, v in report.items() if k != "run"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
