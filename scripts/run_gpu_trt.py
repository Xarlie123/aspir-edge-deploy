#!/usr/bin/env python3
"""
Jetson GPU path (paper name: GPU--TensorRT) — inference on the Orin's Ampere
GPU through a serialised TensorRT engine.

Same two modes and the same protocol as the other routes, so the numbers are
directly comparable:

  Default (latency): every frame timed individually, power sampled alongside,
  PSNR/SSIM/MAE computed against ground truth and against the host FP32
  reference. Additionally reports the transfer/execute breakdown, so a result
  can be quoted at either measurement boundary.

  --energy-mode: idle baseline plus a sustained inference loop, deriving
  J/frame from the two power averages. On the Orin the sampler reads the
  INA3221 rails; the report records which rail was treated as the input plane,
  because it is *module* input here and *board* input on the Ultra96.

The timed region includes host-to-device and device-to-host transfers, exactly
as the DPU route includes its host-side int8 conversion. Timing only the kernel
would flatter this path and break the cross-device comparison.

Engines are built on this device first:

    python3 scripts/build_trt_engine.py --experiment <id> --precision int8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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
from aspir_unet.measurement.energy import EnergySampler, TemperatureSampler
from aspir_unet.measurement.metrics import MetricSummary
from aspir_unet.measurement.timing import TimingMeasurer

ROUTE = "gpu_trt"


def _bench_per_frame(runner, ts, n: int, warmup: int) -> tuple:
    preds = np.empty_like(ts.inputs[:n])
    timer = TimingMeasurer(n_total=n, warmup=warmup)
    energy = EnergySampler()
    temp = TemperatureSampler()
    parts = {"h2d_ms": [], "execute_ms": [], "d2h_ms": []}
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
        "energy": energy.report().to_dict(),
        "temperature": temp.report().to_dict(),
        "metrics_vs_originals": vs_gt.to_dict(),
        "metrics_vs_pt_gold": vs_pt.to_dict(),
    }
    per_frame = {}
    for tag, summary in (("gt", vs_gt), ("fp32", vs_pt)):
        for key, values in (summary.per_frame or {}).items():
            per_frame[f"{key}_vs_{tag}"] = values
    return payload, preds, per_frame


def _power_mode() -> dict:
    """Record the Jetson's power mode: every number on this board depends on it."""
    import subprocess
    info = {}
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                             timeout=10)
        if out.returncode == 0:
            info["nvpmodel"] = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    for path, key in ((Path("/etc/nvpmodel.conf"), None),
                      (Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
                       "cpu0_max_khz")):
        if key and path.is_file():
            try:
                info[key] = int(path.read_text().strip())
            except (OSError, ValueError):
                pass
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--precision",
                        choices=("int8", "int8_nofp16", "int8_strict",
                                 "fp16", "fp32"),
                        default="int8",
                        help="int8: calibrated INT8 with FP16 also allowed — "
                             "the default deployment choice. int8_nofp16: INT8 "
                             "or FP32, no FP16. int8_strict: INT8 with "
                             "precision constraints obeyed, the closest "
                             "comparator to a float-free accelerator.")
    parser.add_argument("--engine", type=Path, default=None,
                        help="Override the engine resolved from the experiment.")
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--energy-mode", action="store_true")
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
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--save-outputs", action="store_true")
    args = parser.parse_args()

    layout = experiments.discover_layout(__file__)
    if args.list_experiments:
        for name in experiments.list_experiments(layout):
            print(name)
        return 0

    exp = experiments.resolve(args.experiment, __file__, layout=layout)
    engine_path = args.engine or exp.require_trt_engine(args.precision)
    npz_path = args.npz or exp.npz
    out_dir = args.out or exp.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp]  {exp.experiment_id}")
    print(f"[load] dataset {npz_path}")
    ts = load_test_set(npz_path)
    n = ts.n if args.limit <= 0 else min(args.limit, ts.n)
    print(f"[load] {n} frames, range "
          f"[{ts.inputs.min():.4f},{ts.inputs.max():.4f}]")

    from aspir_unet.inference.trt import TrtRunner
    print(f"[load] engine {engine_path}")
    runner = TrtRunner(engine_path)
    print(f"[load] TensorRT {runner.trt_version} via {runner.cuda.backend} | "
          f"in={runner.in_dims} out={runner.out_dims}")

    engine_info = runner.engine_info()
    engine_info.pop("layer_information", None)

    mode = "energy" if args.energy_mode else "latency"
    run_block = provenance.collect(
        route=ROUTE,
        mode=mode,
        experiment=exp,
        params={
            "precision": args.precision,
            "warmup": args.warmup,
            "limit": args.limit,
            "n_frames": int(n),
            "timing_frames": args.timing_frames,
            "thermal_warmup_s": args.thermal_warmup_s,
            "energy_idle_s": args.energy_idle_s,
            "energy_active_s": args.energy_active_s,
        },
        artifacts={"model": engine_path, "dataset": npz_path,
                   "onnx": exp.onnx},
        script_file=__file__,
        repetition=args.rep,
        tag=args.tag,
    )
    run_block["accelerator"] = {"engine_info": engine_info,
                                "power_mode": _power_mode()}

    timestamp = datetime.now().isoformat(timespec="seconds")
    report: dict = {
        "schema_version": provenance.SCHEMA_VERSION,
        "timestamp": timestamp,
        "host": hostname(),
        "route": ROUTE,
        "route_name": ROUTE,
        "mode": mode,
        "precision": args.precision,
        "run": run_block,
        "backend": {
            "framework": "tensorrt",
            "version": runner.trt_version,
            "precision": args.precision,
            "model": experiments.rel_to(engine_path, layout),
            "in_dims": list(runner.in_dims),
            "out_dims": list(runner.out_dims),
            "timing_includes": "h2d + execute + d2h",
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
            route_name=f"trt-{args.precision}",
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
            route_name="trt",
        )
        report["timing"] = t_report.to_dict()
        if t_parts:
            report["timing_breakdown"] = t_parts

    runner.close()
    provenance.finalise(run_block)

    name = provenance.report_filename(
        ROUTE, mode, exp.experiment_id, args.rep, timestamp,
        suffix=provenance.run_suffix(ROUTE, {"precision": args.precision}),
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
