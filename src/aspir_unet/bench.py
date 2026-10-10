"""
Shared bench-runner helpers used by run_neon.py / run_dpu.py / run_combined.py.

Two responsibilities:

1. `default_paths()` — resolve --onnx / --xmodel / --npz / --out so the same
   script works both in the host repo (`REPO_ROOT/app/models/test_beam_CCH`)
   and on the deployed board (`/home/petalinux/aspir/models/test_beam_CCH`).
   We probe both layouts and pick whichever exists.

2. `measure_energy_mode()` — idle-baseline + sustained-load energy
   measurement. The DPU runs in ~1.4 ms and NEON in ~20 ms — both far
   shorter than a full sweep of the ~20 PMBus channels, so per-frame
   energy sampling gives <2 samples and a trapezoidal integral of 0.
   Instead: measure board power at idle, then under a sustained inference
   loop, derive J/frame from the two averages and the achieved fps.
   This is what gets quoted in the paper.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .measurement.energy import EnergySampler, read_temperatures_once
from .measurement.uncertainty import board_input_series, energy_uncertainty
from .measurement.timing import TimingMeasurer


_BOARD_ROOT = Path("/home/petalinux/aspir")
# Canonical experiment: beam_64_gaussian_unet_8pct_cch. Trained with the new
# U-Net (Upsample+Conv2d decoder, features=[8,16], 64×64 images).
_REPO_MODEL_SUB = Path("app/models/beam_64_gaussian_unet_8pct_cch")
_BOARD_MODEL_SUB = Path("models/beam_64_gaussian_unet_8pct_cch")
_ONNX_FILENAME = "beam_64_gaussian_unet_8pct_cch.onnx"


@dataclass(frozen=True)
class BenchPaths:
    """Resolved on-disk locations for an inference run."""
    npz: Path
    onnx: Path
    xmodel: Path
    out_dir: Path
    repo_root: Path | None      # None when running on the board

    @property
    def is_board(self) -> bool:
        return self.repo_root is None


def default_paths(script_file: str | Path) -> BenchPaths:
    """Pick host-repo vs board layout based on which directory exists.

    Args:
        script_file: pass `__file__` from the caller.

    Layout discovered:
      * Host:  REPO_ROOT/app/models/test_beam_CCH/{source,compiled,results}/
      * Board: /home/petalinux/aspir/models/test_beam_CCH/{source,compiled}/
               + /home/petalinux/aspir/results/
    """
    here = Path(script_file).resolve()
    # Host repo: scripts/ sibling of src/ under app/, app/ under REPO_ROOT.
    repo_root = here.parents[2] if len(here.parents) >= 3 else None
    repo_models = repo_root / _REPO_MODEL_SUB if repo_root else None
    if repo_models and repo_models.is_dir():
        out = repo_models / "results"
        return BenchPaths(
            npz=repo_models / "source" / "test_images.npz",
            onnx=repo_models / "source" / _ONNX_FILENAME,
            xmodel=repo_models / "phase_c_dpu" / "unet_b512_u96v2.xmodel",
            out_dir=out,
            repo_root=repo_root,
        )
    board_models = _BOARD_ROOT / _BOARD_MODEL_SUB
    return BenchPaths(
        npz=board_models / "source" / "test_images.npz",
        onnx=board_models / "source" / _ONNX_FILENAME,
        xmodel=board_models / "phase_c_dpu" / "unet_b512_u96v2.xmodel",
        out_dir=_BOARD_ROOT / "results",
        repo_root=None,
    )


def hostname() -> str:
    return socket.gethostname()


def rel_path(path: Path, root: Path | None) -> str:
    """Best-effort relative-to-root, falls back to the absolute path on the board."""
    if root is None:
        return str(path)
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


# Frames per timing run. Latency for a fixed-size CNN is input-independent —
# no data-dependent branching, identical FLOPs per frame — so a timing pass
# cycles the evaluation split rather than growing it. 10,000 leaves ~100
# samples above the p99, which is what makes that quantile stable, and the
# count must be identical on every path: a p99 over 10,000 frames and one over
# 1,000 are not the same estimator.
TIMING_FRAMES = 10_000


def measure_timing(
    run_one: Callable[[np.ndarray], np.ndarray],
    frames: np.ndarray,
    *,
    n_frames: int = TIMING_FRAMES,
    warmup: int = 20,
    breakdown_of: Callable[[], dict] | None = None,
    route_name: str = "route",
) -> tuple:
    """Dedicated latency pass: cycle the split until `n_frames` are timed.

    Cycling real frames rather than repeating one buffer keeps the memory
    traffic of an actual deployment and avoids a single-image cache artefact.

    Returns (TimingReport, samples_ms array, breakdown stats). The raw
    per-frame samples are returned so the caller can archive them: a CDF of the
    four execution paths argues determinism far better than a p99 in a table,
    and 10,000 float32 samples are 40 KB.
    """
    n = frames.shape[0]
    if n == 0:
        raise ValueError("measure_timing requires at least one frame")

    timer = TimingMeasurer(n_total=n_frames, warmup=warmup)
    parts: dict = {}
    print(f"[{route_name}] timing pass: {n_frames} frames "
          f"(cycling {n}, warmup {warmup}) …")
    for i in timer.indices():
        j = i % n
        t0 = timer.tic()
        run_one(frames[j:j + 1])
        timer.toc(t0)
        if breakdown_of is not None and i >= warmup:
            for key, value in (breakdown_of() or {}).items():
                parts.setdefault(key, []).append(value)

    report = timer.report()
    samples = np.asarray(report.samples_ms, dtype=np.float32)
    stats = {
        key: {"mean_ms": round(float(np.mean(v)), 4),
              "median_ms": round(float(np.median(v)), 4)}
        for key, v in parts.items()
    }
    print(f"[{route_name}] median {report.median_ms:.3f} ms | "
          f"p95 {report.p95_ms:.3f} | p99 {report.p99_ms:.3f} | "
          f"{report.fps:.1f} fps")
    return report, samples, stats


def measure_idle(seconds: float, label: str = "idle") -> dict:
    """Sample board power for `seconds` of pure sleep. Returns the EnergyReport dict."""
    print(f"[{label}] sampling energy for {seconds:.1f}s …")
    with EnergySampler() as energy:
        time.sleep(seconds)
    return energy.report().to_dict()


def measure_energy_mode(
    run_one: Callable[[np.ndarray], np.ndarray],
    frames: np.ndarray,
    *,
    idle_s: float = 10.0,
    active_s: float = 30.0,
    thermal_warmup_s: float = 60.0,
    route_name: str = "route",
    series_out: dict | None = None,
) -> dict:
    """Idle baseline + sustained-load → J/frame.

    Args:
        run_one: callable that performs one inference on a (1, ...) batch.
        frames: pre-loaded (N, ...) test inputs; cycled through the active loop.
        idle_s: seconds to sample board power with no inference running.
        active_s: seconds to run a tight inference loop while sampling power.
        thermal_warmup_s: seconds of continuous inference before anything is
            recorded, so the idle baseline and the load pass are both taken in
            the same thermal state. Without it a comparison started cold
            favours whichever platform has more thermal headroom — the Orin
            throttles, the 5 W Ultra96 largely does not — and the difference
            would be read as an efficiency result.
        route_name: tag used in the printed log lines.
        series_out: when given, filled with the raw power samples of both
            passes, which the caller archives next to the report. The averages
            answer "how much energy"; the series answers "was this measured or
            asserted", and only the second makes a power-trace figure possible.

    Returns:
        Dict with idle / active / derived sections, suitable for direct JSON dump.
    """
    n = frames.shape[0]
    if n == 0:
        raise ValueError("measure_energy_mode requires at least one frame")

    temps = {"before_warmup": read_temperatures_once()}

    if thermal_warmup_s > 0:
        print(f"[{route_name}] thermal warm-up: {thermal_warmup_s:.0f}s of "
              f"continuous inference (not recorded) …")
        t_end = time.perf_counter() + thermal_warmup_s
        k = 0
        while time.perf_counter() < t_end:
            run_one(frames[k % n:k % n + 1])
            k += 1
        temps["after_warmup"] = read_temperatures_once()

    print(f"[{route_name}] idle baseline: {idle_s:.0f}s (sampling) …")
    with EnergySampler() as e_idle:
        time.sleep(idle_s)
    idle = e_idle.report()
    temps["after_idle"] = read_temperatures_once()
    if series_out is not None:
        series_out["idle"] = e_idle.series()

    print(f"[{route_name}] sustained load: ~{active_s:.0f}s "
          f"(tight inference loop) …")
    with EnergySampler() as e_active:
        t0 = time.perf_counter()
        t_end = t0 + active_s
        count = 0
        # Completion time of every frame, on the sampler's clock, so the
        # load pass can be split into blocks afterwards (uncertainty below).
        frame_ns = []
        while time.perf_counter() < t_end:
            run_one(frames[count % n:count % n + 1])
            frame_ns.append(time.perf_counter_ns())
            count += 1
        elapsed = time.perf_counter() - t0
    frame_t_s = (np.asarray(frame_ns, dtype=np.int64) - e_active._t_start_ns) / 1e9
    active = e_active.report()
    temps["after_active"] = read_temperatures_once()
    active_series = e_active.series()
    if series_out is not None:
        series_out["active"] = dict(active_series, frame_t_s=frame_t_s)

    p_idle = idle.board_input_w_avg
    p_active = active.board_input_w_avg

    # Dynamic energy with the uncertainty convention of ASPIR (settled
    # segments, 1 Hz idle readings, 10 blocks); the fields above keep their
    # meaning.
    idle_series = e_idle.series()
    idle_p = board_input_series(idle_series, idle.board_input_rails)
    active_p = board_input_series(active_series, active.board_input_rails)
    if idle_p is not None and active_p is not None:
        uncertainty = energy_uncertainty(idle_series["t_s"], idle_p,
                                         active_series["t_s"], active_p,
                                         frame_t_s=frame_t_s)
    else:
        uncertainty = {"error": "no board-input rail sampled"}
    p_dyn = p_active - p_idle
    t_frame_s = elapsed / count if count else 0.0
    e_total_per_frame = p_active * t_frame_s
    e_dyn_per_frame = p_dyn * t_frame_s
    fps = count / elapsed if elapsed else 0.0

    print(f"[{route_name}] idle {p_idle:.3f} W | active {p_active:.3f} W | "
          f"dynamic {p_dyn:.3f} W")
    print(f"[{route_name}] {count} frames in {elapsed:.1f}s "
          f"({fps:.1f} fps) | "
          f"{e_total_per_frame*1e3:.3f} mJ/frame "
          f"({e_dyn_per_frame*1e3:.3f} mJ/frame dynamic)")

    return {
        "protocol": {
            "thermal_warmup_s": thermal_warmup_s,
            "idle_s": idle_s,
            "active_s": active_s,
            "note": "idle baseline and load pass are taken in the same thermal "
                    "state, after a warm-up that is not recorded",
        },
        "temperatures_c": temps,
        "idle": {
            "duration_s": round(idle.duration_s, 2),
            "n_samples": idle.n_samples,
            "board_input_w": round(p_idle, 3),
            "per_rail_w": {k: round(v, 4)
                           for k, v in sorted(idle.rail_avg_w.items())},
        },
        "active": {
            "duration_s": round(elapsed, 2),
            "n_samples": active.n_samples,
            "n_frames": count,
            "fps_sustained": round(fps, 1),
            "board_input_w": round(p_active, 3),
            "per_rail_w": {k: round(v, 4)
                           for k, v in sorted(active.rail_avg_w.items())},
        },
        "derived": {
            "board_dynamic_w": round(p_dyn, 3),
            "t_per_frame_ms": round(t_frame_s * 1e3, 4),
            "energy_per_frame_j": round(e_total_per_frame, 6),
            "dynamic_energy_per_frame_j": round(e_dyn_per_frame, 6),
        },
        "uncertainty": uncertainty,
    }
