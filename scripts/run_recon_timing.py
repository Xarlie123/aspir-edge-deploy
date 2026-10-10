#!/usr/bin/env python3
"""
Linear reconstruction timing on the Arm cores (route name: cpu_recon).

The inference measurement alone is a microbenchmark. To place it in the
instrument's frame budget we need the other two stages of the pipeline:

    t_acq = M / f_DMD        acquisition, from the modulator's pattern rate
    t_rec                    linear reconstruction, measured here
    t_inf                    network inference, measured by the route runners

This script times `t_rec` on the board's CPU using the experiment's own
sensing patterns, with the same protocol as the inference runners (warm-up
discarded, per-frame timing, mean/median/p95/p99), and composes the serial
latency and pipelined throughput when `--t-inf-ms` is supplied.

The reconstruction is the Hadamard correlation (back-projection):

    y  = Phi   @ x           forward model, the optical stage in reality
    xr = Phi.T @ y           back-projection, then min-max normalisation

Only the back-projection and its normalisation count towards `t_rec`; the
forward product is timed separately and reported for information, since on a
real instrument the light does it, not the CPU.

A sanity pass compares the reconstruction computed here against the one
stored in the dataset. It is a diagnostic, not a claim: a low agreement means
the dataset was produced with different normalisation or a regularised
inverse, and the timing still stands for the operation actually executed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Thread control has to happen before numpy pulls in its BLAS, so re-exec
# once with the environment set. The guard variable keeps it to one hop.
_REEXEC_FLAG = "ASPIR_RECON_THREADS_SET"
if "--threads" in sys.argv and _REEXEC_FLAG not in os.environ:
    _idx = sys.argv.index("--threads")
    if _idx + 1 < len(sys.argv):
        _n = sys.argv[_idx + 1]
        os.environ.update({
            _REEXEC_FLAG: _n,
            "OMP_NUM_THREADS": _n,
            "OPENBLAS_NUM_THREADS": _n,
            "MKL_NUM_THREADS": _n,
            "NUMEXPR_NUM_THREADS": _n,
        })
        os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np

# Locate aspir_unet for both host (REPO/app/src) and board (DEST/src).
_HERE = Path(__file__).resolve()
for _src in (_HERE.parents[2] / "app" / "src", _HERE.parents[1] / "src"):
    if (_src / "aspir_unet").is_dir():
        sys.path.insert(0, str(_src))
        break

from aspir_unet import experiments, provenance
from aspir_unet.bench import hostname, measure_energy_mode
from aspir_unet.data import load_test_set
from aspir_unet.measurement.energy import EnergySampler, TemperatureSampler
from aspir_unet.measurement.metrics import psnr_per_image
from aspir_unet.measurement.timing import TimingMeasurer

ROUTE = "cpu_recon"
DEFAULT_DMD_KHZ = 10.752         # Vialux V-650L02 NIR, as in the 2025 study


def _load_masks(path: Path) -> np.ndarray:
    """Return the sensing matrix as (M, N) float32."""
    npz = np.load(str(path))
    key = "masks" if "masks" in npz.files else npz.files[0]
    masks = npz[key]
    if masks.ndim == 3:
        m = masks.reshape(masks.shape[0], -1)
    elif masks.ndim == 2:
        m = masks
    else:
        raise ValueError(f"cannot interpret masks with shape {masks.shape}")
    return np.ascontiguousarray(m.astype(np.float32))


def _backprojection(phi_t: np.ndarray, measurement: np.ndarray,
                    bipolar: bool, divisor: int) -> np.ndarray:
    """The estimator the dataset pipeline actually uses, per pattern family.

    Bipolar (+/-1) patterns carry no DC term, so a plain back-projection
    recovers the image: x = Phi^T y / N, the pseudoinverse (ASPIR >= 1.3,
    ``normalization.linear_divisor == "pixels"``), or Phi^T y / M as exported
    by earlier versions; ``divisor`` is N or M accordingly. Binary (0/1)
    patterns do carry one, and it swamps the estimate unless the measurements
    are centred first — which is the classical ghost-imaging correlation,
    x = Phi^T (y - mean(y)) / M.

    Verified against the shipped reconstructions at >130 dB for both families,
    so what is timed here is the deployed operation and not an approximation
    of it.
    """
    if bipolar:
        return (phi_t @ measurement) / divisor
    return (phi_t @ (measurement - measurement.mean())) / divisor


def _normalise01(arr: np.ndarray) -> np.ndarray:
    """Per-frame min-max to [0, 1]."""
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo + 1e-8)


def _global_scale(raw_batch: np.ndarray) -> tuple:
    """Fixed (lo, span) computed once, as the dataset pipeline does offline."""
    lo = float(raw_batch.min())
    hi = float(raw_batch.max())
    span = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    return lo, span + 1e-8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None,
                        help="Experiment id under models/.")
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--masks", type=Path, default=None,
                        help="Override the masks npz resolved from the experiment.")
    parser.add_argument("--npz", type=Path, default=None,
                        help="Override the dataset resolved from the experiment.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, reconstruct only the first N frames.")
    parser.add_argument("--threads", type=int, default=None,
                        help="BLAS thread count (applied by re-exec before "
                             "numpy loads).")
    parser.add_argument("--norm", choices=("dataset", "per-frame", "global", "none"),
                        default="dataset",
                        help="Normalisation applied after back-projection. "
                             "'dataset' uses the constants the experiment "
                             "exports (lo, span), reproducing the deployed "
                             "pipeline exactly; 'global' derives a fixed scale "
                             "from a probe batch; 'per-frame' recomputes "
                             "min/max per frame and costs slightly more; "
                             "'none' times the bare matvec.")
    parser.add_argument("--dmd-khz", type=float, default=DEFAULT_DMD_KHZ,
                        help="Binary pattern rate of the modulator, for t_acq.")
    parser.add_argument("--t-inf-ms", type=float, default=None,
                        help="Inference time to compose the end-to-end budget "
                             "with; from a route report of the same experiment.")
    parser.add_argument("--energy-mode", action="store_true",
                        help="Idle baseline plus a sustained reconstruction "
                             "loop, deriving J/frame. The paper measures the "
                             "energy of inference; without this it cannot say "
                             "what fraction of the per-frame compute energy the "
                             "linear step takes, which is the other half of the "
                             "bottleneck argument.")
    parser.add_argument("--energy-idle-s", type=float, default=10.0)
    parser.add_argument("--energy-active-s", type=float, default=30.0)
    parser.add_argument("--thermal-warmup-s", type=float, default=60.0)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    layout = experiments.discover_layout(__file__)
    if args.list_experiments:
        for name in experiments.list_experiments(layout):
            print(name)
        return 0

    exp = experiments.resolve(args.experiment, __file__, layout=layout)
    masks_path = args.masks or exp.require_masks()
    npz_path = args.npz or exp.npz
    out_dir = args.out or exp.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp]  {exp.experiment_id}")
    print(f"[load] masks {masks_path}")
    phi = _load_masks(masks_path)
    m_patterns, n_pixels = phi.shape
    cfg = exp.config()
    bipolar = str(cfg.get("mask_polarity", "")).lower() == "bipolar" \
        or float(phi.min()) < 0
    dataset_norm = cfg.get("normalization") or {}
    # The (lo, span) constants were taken on ASPIR's linear output: since ASPIR
    # 1.3 its Hadamard back-projection divides by the pixel count and the export
    # says so; older exports divided by the number of patterns.
    divisor = n_pixels if bipolar and dataset_norm.get("linear_divisor") == "pixels" \
        else m_patterns
    print(f"[load] Phi = ({m_patterns}, {n_pixels})  "
          f"M/N = {100.0 * m_patterns / n_pixels:.2f}%  "
          f"{'bipolar' if bipolar else 'binary'} patterns, back-projection / {divisor}")
    if args.norm == "dataset" and not dataset_norm:
        print("[recon] WARNING the experiment exports no normalisation "
              "constants; falling back to a probe-derived global scale")
        args.norm = "global"

    print(f"[load] dataset {npz_path}")
    ts = load_test_set(npz_path)
    n = ts.n if args.limit <= 0 else min(args.limit, ts.n)
    side = int(round(n_pixels ** 0.5))
    if side * side != n_pixels or ts.ground_truth.shape[-1] != side:
        raise ValueError(
            f"masks describe {n_pixels} pixels but the dataset frames are "
            f"{ts.ground_truth.shape[-2:]}; mismatched experiment artefacts."
        )

    x_flat = ts.ground_truth[:n].reshape(n, -1).astype(np.float32)
    stored = ts.inputs[:n].reshape(n, -1)

    phi_t = np.ascontiguousarray(phi.T)      # (N, M), so t_rec is a plain matvec
    recon = np.empty_like(x_flat)

    # Forward model, timed separately: on the instrument the optics do this.
    fwd_timer = TimingMeasurer(n_total=n, warmup=args.warmup)
    measurements = np.empty((n, m_patterns), dtype=np.float32)
    for i in fwd_timer.indices():
        t0 = fwd_timer.tic()
        measurements[i] = phi @ x_flat[i]
        fwd_timer.toc(t0)

    # The global scale is part of the offline calibration of the pipeline, so
    # it is computed before the timed loop, not inside it.
    if args.norm == "dataset":
        g_lo = float(dataset_norm["lo"])
        g_span = float(dataset_norm["span"])
        print(f"[recon] dataset normalisation: lo={g_lo:.4f} span={g_span:.4f}")
    elif args.norm == "global":
        probe = np.stack([_backprojection(phi_t, measurements[i], bipolar,
                                          divisor)
                          for i in range(min(n, 16))])
        g_lo, g_span = _global_scale(probe)
        print(f"[recon] global normalisation scale: lo={g_lo:.4f} "
              f"span={g_span:.4f} (from {min(n, 16)} frames)")

    # Reconstruction: this is t_rec.
    print(f"[bench] reconstructing {n} frames "
          f"(warmup={args.warmup}, norm={args.norm}) …")
    rec_timer = TimingMeasurer(n_total=n, warmup=args.warmup)
    energy = EnergySampler()
    temp = TemperatureSampler()
    with energy, temp:
        for i in rec_timer.indices():
            t0 = rec_timer.tic()
            raw = _backprojection(phi_t, measurements[i], bipolar, divisor)
            if args.norm in ("dataset", "global"):
                recon[i] = (raw - g_lo) / g_span
            elif args.norm == "per-frame":
                recon[i] = _normalise01(raw)
            else:
                recon[i] = raw
            rec_timer.toc(t0)
    rec_report = rec_timer.report()
    energy_report = energy.report()
    temp_report = temp.report()

    # Diagnostic: does our reconstruction match the one shipped in the dataset?
    agreement = psnr_per_image(
        recon.reshape(n, 1, side, side), stored.reshape(n, 1, side, side)
    )
    agreement_db = float(np.median(agreement[np.isfinite(agreement)])) \
        if np.isfinite(agreement).any() else float("inf")

    # A bipolar pattern is realised on a binary modulator as a complementary
    # pair, so it costs two projections; a binary pattern costs one. Ignoring
    # this understates the acquisition time of the Hadamard families by 2x and
    # would move the bottleneck crossover.
    projections = m_patterns * (2 if bipolar else 1)
    t_acq_ms = projections / (args.dmd_khz * 1e3) * 1e3
    t_rec_ms = rec_report.median_ms
    budget = {
        "m_patterns": int(m_patterns),
        "binary_projections": int(projections),
        "bipolar_patterns": bool(bipolar),
        "n_pixels": int(n_pixels),
        "sampling_ratio_pct": round(100.0 * m_patterns / n_pixels, 3),
        "dmd_khz": args.dmd_khz,
        "t_acq_ms": round(t_acq_ms, 4),
        "t_rec_ms": round(t_rec_ms, 4),
    }
    if args.t_inf_ms is not None:
        t_tot = t_acq_ms + t_rec_ms + args.t_inf_ms
        budget.update({
            "t_inf_ms": args.t_inf_ms,
            "t_tot_ms": round(t_tot, 4),
            "fps_serial": round(1000.0 / t_tot, 3) if t_tot > 0 else 0.0,
            "bottleneck": max(
                (("acquisition", t_acq_ms), ("reconstruction", t_rec_ms),
                 ("inference", args.t_inf_ms)), key=lambda kv: kv[1])[0],
            "fps_pipelined": round(
                1000.0 / max(t_acq_ms, t_rec_ms, args.t_inf_ms), 3),
        })

    mode = "energy" if args.energy_mode else "recon"
    run_block = provenance.collect(
        route=ROUTE,
        mode=mode,
        experiment=exp,
        params={
            "warmup": args.warmup,
            "limit": args.limit,
            "n_frames": int(n),
            "threads": args.threads,
            "normalisation": args.norm,
            "estimator": "backprojection" if bipolar else "ghost_correlation",
            "bipolar_patterns": bool(bipolar),
            "dmd_khz": args.dmd_khz,
            "t_inf_ms": args.t_inf_ms,
        },
        artifacts={"masks": masks_path, "dataset": npz_path},
        script_file=__file__,
        repetition=args.rep,
        tag=args.tag,
    )

    timestamp = datetime.now().isoformat(timespec="seconds")
    report = {
        "schema_version": provenance.SCHEMA_VERSION,
        "timestamp": timestamp,
        "host": hostname(),
        "route": ROUTE,
        "route_name": ROUTE,
        "mode": mode,
        "run": run_block,
        "backend": {
            "framework": "numpy",
            "operation": ("backprojection" if bipolar else "ghost_correlation"),
            "normalisation": args.norm,
            "masks": experiments.rel_to(masks_path, layout),
        },
        "dataset": {
            "npz": experiments.rel_to(npz_path, layout),
            "n_frames": int(n),
        },
        "timing": rec_report.to_dict(),
        "timing_forward_model": fwd_timer.report().to_dict(),
        "energy": energy_report.to_dict(),
        "temperature": temp_report.to_dict(),
        "pipeline_budget": budget,
        "diagnostics": {
            "psnr_vs_stored_reconstruction_db": round(agreement_db, 3),
            "note": "diagnostic only; a low value means the dataset used a "
                    "different normalisation or a regularised inverse.",
        },
    }

    if args.energy_mode:
        # One reconstruction, the same operation the timed loop performs, so
        # the energy figure covers exactly what the latency figure timed.
        def reconstruct_one(frame_batch):
            i = int(reconstruct_one.index % n)
            reconstruct_one.index += 1
            raw = _backprojection(phi_t, measurements[i], bipolar, divisor)
            if args.norm in ("dataset", "global"):
                return (raw - g_lo) / g_span
            if args.norm == "per-frame":
                return _normalise01(raw)
            return raw
        reconstruct_one.index = 0
        power_series = {}
        report["energy_mode"] = measure_energy_mode(
            reconstruct_one, ts.inputs[:n],
            idle_s=args.energy_idle_s, active_s=args.energy_active_s,
            thermal_warmup_s=args.thermal_warmup_s,
            route_name="recon", series_out=power_series,
        )
        if power_series:
            samples_npz = out_dir / (provenance.report_filename(
                ROUTE, mode, exp.experiment_id, args.rep, timestamp,
                suffix=provenance.run_suffix(ROUTE, {"threads": args.threads}),
            ).replace(".json", "_power.npz"))
            np.savez_compressed(samples_npz, **{
                f"{k}_{kk}": np.asarray(vv, dtype=np.float32)
                for k, s in power_series.items() for kk, vv in s.items()})
            report["power_series_file"] = samples_npz.name
            print(f"[save] {samples_npz}")

    provenance.finalise(run_block)
    name = provenance.report_filename(
        ROUTE, mode, exp.experiment_id, args.rep, timestamp,
        suffix=provenance.run_suffix(ROUTE, {"threads": args.threads}),
    )
    out_json = out_dir / name
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[save] {out_json}")
    print(json.dumps({k: v for k, v in report.items() if k != "run"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
