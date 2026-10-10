"""
Uncertainty of the dynamic energy per frame, with the convention of ASPIR.

ASPIR (``simulation_engine/_5_analyzer/baseline_capture.py``) and the paper
report the dynamic energy per frame as

    E_dyn = (P_bar - P_idle) * t_bar

where P_bar and t_bar are means over blocks of the load pass, and
P_idle +/- sigma_idle is the idle baseline. The uncertainty has two
independent components, reported separately and combined in quadrature:

* idle baseline: sigma_idle * t_bar, with sigma_idle the *standard deviation*
  of the idle readings (not their standard error, which hides the baseline
  drifting between the idle window and the run). The standard-error version
  is stored too;
* blocks: first-order (delta-method) propagation of the sampling error of
  P_bar and t_bar over the B blocks, covariance included,
  sigma^2 = [t^2 s_P^2 + (P - P_idle)^2 s_t^2 + 2 t (P - P_idle) s_Pt] / B.

Sensor resolution. Some rails report power in coarse steps: the Ultra96's
board input (ir38060, PMBus LINEAR11) reads only 5.00 W or 6.25 W, a step of
1.25 W, and a mean of samples that never leave one level is known only to
within that step. Its standard deviation is then zero, which says nothing about
the precision. The step Delta is estimated from the samples (smallest gap
between the distinct values seen), and each mean gets a resolution term
Delta / sqrt(12), so the dynamic power gets Delta / sqrt(6). That term is
reported apart (`dynamic_energy_sigma_resolution_mj`) and added to the
combination in `dynamic_energy_sigma_with_resolution_mj`. The two-component
fields keep exactly the meaning they have in ASPIR, where NVML (mW) and RAPL
(uJ) make the term negligible.

A 95 % interval comes from a bootstrap that resamples the blocks and draws
P_idle ~ N(P_idle, sigma_idle); the seed is fixed. `dynamic_energy_uncertainty`
is the same function as in ASPIR; keep the two copies identical.

How the hwmon telemetry maps onto those inputs:

* Both segments drop their first `SETTLE_S` seconds, as the figures already
  do (`figures/derive.py`): each pass starts with a transient.
* Idle readings are the means of consecutive `IDLE_WINDOW_S` windows of the
  settled idle segment. ASPIR polls its backends at 1 Hz and the Jetson
  analysis used jtop at about 1 Hz; averaging the 18-100 Hz hwmon samples to
  the same cadence keeps sigma_idle comparable across platforms. The standard
  deviation of the raw samples is stored as well.
* The settled load segment is split into `N_BLOCKS` equal time blocks. P_b is
  the mean power of the samples in block b and t_b its duration divided by
  the frames completed in it. Runs that recorded frame times use them; older
  runs only have the sustained frame rate, so every t_b is 1/fps and the
  block term then carries the power spread only.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

SETTLE_S = 2.5          # same window as figures/derive.py
IDLE_WINDOW_S = 1.0     # 1 Hz, the cadence of ASPIR's idle baseline
N_BLOCKS = 10           # as ASPIR's 800 runs in 10 blocks
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0


def dynamic_energy_uncertainty(
    block_energy_mj,
    block_power_w,
    block_time_ms,
    idle: Optional[dict],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Dynamic energy per frame and its uncertainty (see the module docstring).

    Energies in mJ, powers in W, times in ms; ``idle`` is
    ``{"mean_W", "std_W", "sem_W", "n"}`` or None.
    """
    E = np.asarray(block_energy_mj, dtype=float)
    P = np.asarray(block_power_w, dtype=float)
    t = np.asarray(block_time_ms, dtype=float)
    B = int(min(E.size, P.size, t.size))
    out: dict[str, Any] = {"n_blocks": B}
    if B == 0:
        return out
    E, P, t = E[:B], P[:B], t[:B]
    out["energy_block_mean_mj"] = float(E.mean())
    out["energy_block_sem_mj"] = float(E.std(ddof=1) / math.sqrt(B)) if B > 1 else None
    if idle is None:
        return out

    p_idle, s_idle = float(idle["mean_W"]), float(idle["std_W"])
    p_bar, t_bar = float(P.mean()), float(t.mean())
    p_dyn = p_bar - p_idle
    e_dyn = p_dyn * t_bar
    s_idle_term = s_idle * t_bar
    sem_idle = float(idle.get("sem_W", float("nan")))
    if B > 1:
        cov = np.cov(np.vstack([P, t]), ddof=1)
        var_blocks = (t_bar ** 2 * cov[0, 0] + p_dyn ** 2 * cov[1, 1]
                      + 2 * t_bar * p_dyn * cov[0, 1]) / B
        s_blocks = float(math.sqrt(max(var_blocks, 0.0)))
        s_p_blocks = float(math.sqrt(cov[0, 0] / B))
    else:
        s_blocks, s_p_blocks = 0.0, 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B, size=(draws, B))
    idle_draws = rng.normal(p_idle, s_idle, size=draws)
    boot = (P[idx].mean(axis=1) - idle_draws) * t[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    out.update({
        "idle_power_W": p_idle,
        "idle_power_std_W": s_idle,
        "idle_power_sem_W": sem_idle,
        "n_idle_samples": int(idle.get("n", 0)),
        "dynamic_power_block_W": p_dyn,
        "dynamic_power_sigma_W": float(math.sqrt(s_idle ** 2 + s_p_blocks ** 2)),
        "dynamic_energy_pt_mj": e_dyn,
        "dynamic_energy_sigma_idle_mj": s_idle_term,
        "dynamic_energy_sigma_idle_sem_mj": sem_idle * t_bar,
        "dynamic_energy_sigma_blocks_mj": s_blocks,
        "dynamic_energy_sigma_mj": float(math.sqrt(s_idle_term ** 2 + s_blocks ** 2)),
        "dynamic_energy_ci95_mj": [float(lo), float(hi)],
        "idle_share_of_energy": float(p_idle * t_bar / E.mean()) if E.mean() > 0 else None,
    })
    return out


def _settled(t_s, p_w, settle_s: float):
    t = np.asarray(t_s, dtype=float)
    p = np.asarray(p_w, dtype=float)
    keep = (t >= settle_s) & np.isfinite(p)
    return t[keep], p[keep]


def idle_statistics(t_s, p_w, settle_s: float = SETTLE_S,
                    window_s: float = IDLE_WINDOW_S) -> Optional[dict]:
    """Idle baseline from the settled idle segment, at a 1 Hz cadence.

    ``mean_W`` is the mean of the settled samples; ``std_W`` the sample
    standard deviation of the ``window_s`` means (``n`` of them), and
    ``sem_W`` = ``std_W`` / sqrt(n). ``raw_std_W`` / ``n_raw`` describe the
    individual samples.
    """
    t, p = _settled(t_s, p_w, settle_s)
    if p.size < 2:
        return None
    edges = np.arange(t[0], t[-1] + 1e-9, window_s)
    means = [p[(t >= a) & (t < a + window_s)].mean() for a in edges[:-1]
             if np.any((t >= a) & (t < a + window_s))]
    means = np.asarray(means, dtype=float)
    if means.size < 2:
        means = p
    std = float(means.std(ddof=1))
    return {"mean_W": float(p.mean()), "std_W": std, "sem_W": std / math.sqrt(means.size),
            "n": int(means.size), "window_s": window_s, "settle_s": settle_s,
            "raw_std_W": float(p.std(ddof=1)), "n_raw": int(p.size)}


def active_blocks(t_s, p_w, frame_t_s: Optional[Sequence[float]] = None,
                  fps: Optional[float] = None, settle_s: float = SETTLE_S,
                  n_blocks: int = N_BLOCKS) -> Optional[dict]:
    """Per-block mean power and time per frame of the settled load segment."""
    t, p = _settled(t_s, p_w, settle_s)
    if p.size < n_blocks:
        return None
    edges = np.linspace(t[0], t[-1], n_blocks + 1)
    frames = None if frame_t_s is None else np.asarray(frame_t_s, dtype=float)
    power, time_ms, energy = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (t >= a) & (t < b) if b < edges[-1] else (t >= a) & (t <= b)
        if not np.any(sel):
            continue
        pw = float(p[sel].mean())
        if frames is not None:
            n = int(np.count_nonzero((frames >= a) & (frames < b)))
            if n == 0:
                continue
            tf = 1000.0 * (b - a) / n
        elif fps:
            tf = 1000.0 / float(fps)
        else:
            return None
        power.append(pw)
        time_ms.append(tf)
        energy.append(pw * tf)
    return {"power_W": power, "time_ms": time_ms, "energy_mj": energy,
            "settle_s": settle_s, "source": "frame_times" if frames is not None else "fps"}


def sensor_resolution(*series) -> Optional[float]:
    """Power step of a rail: the smallest gap between the distinct values read.

    None when every sample has the same value (the step cannot be seen).
    Values are rounded to 0.1 mW first so float32 storage noise is not taken
    for a step.
    """
    vals = np.unique(np.round(np.concatenate([np.asarray(x, dtype=float).ravel()
                                              for x in series]), 4))
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None
    return round(float(np.min(np.diff(vals))), 6)


def energy_uncertainty(idle_t_s, idle_p_w, active_t_s, active_p_w,
                       frame_t_s=None, fps=None, settle_s: float = SETTLE_S) -> dict[str, Any]:
    """The `uncertainty` block of an energy-mode report, from the raw series."""
    idle = idle_statistics(idle_t_s, idle_p_w, settle_s)
    blocks = active_blocks(active_t_s, active_p_w, frame_t_s, fps, settle_s)
    if blocks is None:
        return {"error": "load segment too short for the block split"}
    out = dynamic_energy_uncertainty(blocks["energy_mj"], blocks["power_W"],
                                     blocks["time_ms"], idle)
    out["idle"] = idle
    out["blocks"] = blocks
    step = sensor_resolution(idle_p_w, active_p_w)
    out["sensor_resolution_W"] = step
    if step is not None and "dynamic_energy_pt_mj" in out:
        t_bar = float(np.mean(blocks["time_ms"]))
        s_res = step / math.sqrt(6.0) * t_bar
        out["dynamic_energy_sigma_resolution_mj"] = s_res
        out["dynamic_energy_sigma_with_resolution_mj"] = float(
            math.sqrt(out["dynamic_energy_sigma_mj"] ** 2 + s_res ** 2))
    out["convention"] = ("ASPIR: E_dyn = (P - P_idle) t over blocks; sigma_idle = std of "
                         f"{IDLE_WINDOW_S:g} s idle means; delta-method block term; "
                         "95 % bootstrap over blocks with N(P_idle, sigma_idle)")
    return out


def board_input_series(series: dict, rails: Sequence[str]) -> Optional[np.ndarray]:
    """Sum of the board-input rails in a sampler series (or a sidecar segment)."""
    cols = [np.asarray(series[r], dtype=float) for r in rails if r in series]
    if not cols:
        return None
    return np.sum(cols, axis=0)
