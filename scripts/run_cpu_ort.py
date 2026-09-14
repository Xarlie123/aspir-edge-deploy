#!/usr/bin/env python3
"""
Route A (paper name: CPU--NEON) — ONNX Runtime FP32 inference on the
Ultra96-v2's four Cortex-A53 cores.

Two modes via --energy-mode:

  Default (latency): runs every frame, times each, samples hwmon while
  the loop runs, then writes timing/energy/metrics. This is the pass that
  produces PSNR/SSIM and the latency distribution.

  --energy-mode (idle vs sustained): samples power at idle, then under a
  sustained inference loop, derives J/frame from the two averages. This is
  the paper's energy number — the per-frame sampler under-samples short
  NEON calls.

Which model and dataset run is selected with --experiment; every artefact
is resolved through the experiment registry, so the same command works on
the host repo and on a deployed board. Reports are schema v2: they carry a
`run` block with the code, artefact hashes, software versions and machine
state that produced them.
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
from aspir_unet.inference.neon import OnnxNeonRunner
from aspir_unet.measurement.energy import EnergySampler, TemperatureSampler
from aspir_unet.measurement.metrics import MetricSummary
from aspir_unet.measurement.timing import TimingMeasurer

ROUTE = "cpu_neon"
LEGACY_ROUTE = "A_neon"          # kept so existing plotting keeps working


def _bench_per_frame(runner: OnnxNeonRunner, ts, n: int, warmup: int) -> tuple:
    preds = np.empty_like(ts.inputs[:n])
    timer = TimingMeasurer(n_total=n, warmup=warmup)
    energy = EnergySampler()
    temp = TemperatureSampler()
    print(f"[bench] hwmon: energy_rails={len(energy.rail_labels)} "
          f"temp_sensors={len(temp.sensor_labels)}")
    with energy, temp:
        for i in timer.indices():
            t0 = timer.tic()
            preds[i] = runner.run_one(ts.inputs[i:i + 1])[0]
            timer.toc(t0)
    timing = timer.report()
    energy_report = energy.report()
    temp_report = temp.report()

    print("[metrics] computing PSNR / SSIM / MAE …")
    vs_gt = MetricSummary.from_pair(preds, ts.ground_truth[:n],
                                    keep_per_frame=True)
    vs_pt = MetricSummary.from_pair(preds, ts.gold_pt[:n],
                                    keep_per_frame=True)

    payload = {
        "timing": timing.to_dict(),
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


def _unload_pl() -> None:
    """Drop any PL configuration so the CPU baseline is a genuinely idle fabric.

    SSH non-interactive PATH on the board does not include /usr/local/bin, so
    call by full path. Failures are non-fatal (e.g. a host-side smoke test).
    """
    unload = "/usr/local/bin/aspir-unload-dpu"
    try:
        r = subprocess.run([unload], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            print(f"[pl] {r.stdout.strip() or 'PL unloaded'}")
        else:
            print(f"[pl] WARNING aspir-unload-dpu rc={r.returncode}: "
                  f"{(r.stderr or r.stdout).strip()}")
    except FileNotFoundError:
        print(f"[pl] NOTE {unload} not installed (off-board run?); "
              f"continuing without unload.")
    except subprocess.TimeoutExpired:
        print("[pl] WARNING aspir-unload-dpu timed out; continuing anyway.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None,
                        help="Experiment id under models/. Defaults to "
                             f"'{experiments.LEGACY_EXPERIMENT_ID}'.")
    parser.add_argument("--list-experiments", action="store_true",
                        help="Print the available experiment ids and exit.")
    parser.add_argument("--onnx", type=Path, default=None,
                        help="Override the ONNX resolved from the experiment.")
    parser.add_argument("--npz", type=Path, default=None,
                        help="Override the dataset resolved from the experiment.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Override the results directory.")
    parser.add_argument("--precision", choices=("fp32", "int8"), default="fp32",
                        help="fp32 is the deployment default and the reference "
                             "everything else is measured against; int8 selects "
                             "the quantised export and exists as a control, so "
                             "that an INT8 accelerator is not credited with a "
                             "gain that is really the data width.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20,
                        help="Frames discarded before timing starts.")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, run only the first N frames.")
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
    parser.add_argument(
        "--no-unload-pl", action="store_true",
        help="Skip the aspir-unload-dpu call at startup. Default is to "
             "unload any FPGA overlay before measuring so the NEON power "
             "figures reflect a pure PS+idle-PL baseline, not PS+DPU-clock "
             "leakage. Pass this flag only if you want NEON measured with "
             "the DPU bitstream still resident on the PL.")
    args = parser.parse_args()

    layout = experiments.discover_layout(__file__)
    if args.list_experiments:
        for name in experiments.list_experiments(layout):
            print(name)
        return 0

    exp = experiments.resolve(args.experiment, __file__, layout=layout)
    if args.onnx is not None:
        onnx_path = args.onnx
    elif args.precision == "int8":
        onnx_path = exp.require_onnx_int8()
    else:
        onnx_path = exp.require_onnx()
    npz_path = args.npz or exp.npz
    out_dir = args.out or exp.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_unload_pl:
        _unload_pl()

    print(f"[exp]  {exp.experiment_id}")
    print(f"[load] dataset {npz_path}")
    ts = load_test_set(npz_path)
    n = ts.n if args.limit <= 0 else min(args.limit, ts.n)
    print(f"[load] {n} frames, range "
          f"[{ts.inputs.min():.4f},{ts.inputs.max():.4f}]")

    print(f"[load] onnx {onnx_path}  ({args.threads} intra threads, NEON CPU EP)")
    runner = OnnxNeonRunner(onnx_path, intra_threads=args.threads)

    mode = "energy" if args.energy_mode else "latency"
    run_block = provenance.collect(
        route=ROUTE,
        mode=mode,
        experiment=exp,
        params={
            "precision": args.precision,
            "threads": args.threads,
            "warmup": args.warmup,
            "limit": args.limit,
            "n_frames": int(n),
            "timing_frames": args.timing_frames,
            "thermal_warmup_s": args.thermal_warmup_s,
            "energy_idle_s": args.energy_idle_s,
            "energy_active_s": args.energy_active_s,
            "unload_pl": not args.no_unload_pl,
        },
        artifacts={"model": onnx_path, "dataset": npz_path},
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
        "precision": args.precision,
        "run": run_block,
        "backend": {
            "framework": "onnxruntime",
            "providers": [p for p in runner.session.get_providers()],
            "intra_op_threads": args.threads,
            "model": experiments.rel_to(onnx_path, layout),
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
            route_name="neon",
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
            breakdown_of=None,
            route_name="neon",
        )
        report["timing"] = t_report.to_dict()
        if t_parts:
            report["timing_breakdown"] = t_parts

    provenance.finalise(run_block)

    name = provenance.report_filename(
        ROUTE, mode, exp.experiment_id, args.rep, timestamp,
        suffix=provenance.run_suffix(
            ROUTE, {"threads": args.threads, "precision": args.precision}),
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
