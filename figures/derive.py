#!/usr/bin/env python3
"""data/  ->  figures/data/*.csv

The ONLY script permitted to read data/. It reduces the sidecars -- raw
per-frame latencies, per-frame quality metrics, predicted frames, power
traces -- into small CSVs that the plotting scripts consume. Everything a
figure needs must end up here, so that figures regenerate on a machine that
does not have the 159 MB campaign.

Run after a new campaign lands; commit the CSVs it writes.

Conventions enforced (figures/README.md):
  1. clip to [0,1] before any metric
  2. only runs with clocks pinned
  3. medians across repetitions, spread reported
  4. quality has one repetition on purpose
  6. GPU INT8 comes in variants; keep them distinct
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import common as C


def _pinned(rep: dict) -> bool:
    """Clock state fixed, by either board's convention -- see common.clocks_fixed."""
    clocks = rep.get("run", {}).get("clocks", {}) or {}
    if clocks.get("gpu_pinned") or clocks.get("cpu_pinned"):
        return True
    return clocks.get("cpu_dvfs") is False


def _iter_reports(board: str):
    root = C.RAW / "results" / board
    if not root.is_dir():
        return
    for js in sorted(root.rglob("*.json")):
        if js.parent.name == "sweeps":
            continue
        try:
            rep = json.load(open(js))
        except Exception:
            continue
        if "run" not in rep:
            continue
        yield js, rep


def _experiment_id(rep: dict) -> str:
    """`run.experiment` is a block, not a string: it carries the resolved paths
    of everything the run loaded. The identifier is one field inside it."""
    e = rep.get("run", {}).get("experiment")
    if isinstance(e, dict):
        return e.get("experiment_id", "?")
    return e or "?"


def _meta(experiment: str) -> dict:
    p = C.RAW / "datasets" / experiment / "meta.json"
    return json.load(open(p)) if p.is_file() else {}


def _engine(rep: dict) -> dict:
    """Engine composition, flattened.

    `layer_precision_counts` is the field that decides whether a run labelled
    INT8 actually executed in INT8. Two builds of the same network under the
    same flag produced {INT8: 22, FP32: 1} and {FP32: 20, INT8: 7}; both were
    aggregated under one name until this was surfaced. It belongs in the CSV as
    a first-class column, not buried in the report.

    Note that the layer total varies with the precision assignment, because
    TensorRT fuses differently, so the INT8 *fraction* is comparable across
    engines only loosely. Coverage by weights or FLOPs would be better; the
    reports do not carry it yet.
    """
    acc = rep.get("run", {}).get("accelerator") or {}
    ei = acc.get("engine_info") or {}
    lpc = ei.get("layer_precision_counts") or {}
    total = sum(lpc.values()) or None
    out = {
        "engine_name": ei.get("engine", ""),
        "engine_n_layers": ei.get("n_layers") or total or "",
        "trt_version": ei.get("tensorrt_version", ""),
        "layers_int8": lpc.get("INT8", 0) if lpc else "",
        "layers_fp16": lpc.get("FP16", 0) if lpc else "",
        "layers_fp32": lpc.get("FP32", 0) if lpc else "",
        "layers_total": total or "",
        "int8_layer_fraction": (round(lpc.get("INT8", 0) / total, 4)
                                if total else ""),
        "precision_map": ";".join(f"{k}={v}" for k, v in sorted(lpc.items())),
        "build_commit": (rep.get("run", {}).get("code") or {}).get("git_sha", ""),
        "deployed_at": (rep.get("run", {}).get("code") or {}).get("deployed_at", ""),
    }
    return out


def _route_key(rep: dict) -> tuple[str, str, str]:
    run = rep["run"]
    return (run.get("route", "?"),
            run.get("device", {}).get("device", "?"),
            (run.get("params", {}) or {}).get("precision", "") or "")


# ---------------------------------------------------------------------------
def derive_latency_cdf(board: str):
    """Raw per-frame latencies -> a quantile curve per route.

    Storing 10,000 samples per run in the repository would be wasteful and
    would defeat the point of the aggregate. A dense quantile grid preserves
    the shape of the distribution, tail included, at a few hundred rows --
    which is all the CDF figure needs.
    """
    q = np.concatenate([
        np.linspace(0.0, 0.98, 100),
        np.linspace(0.98, 0.9999, 120),          # the tail is the argument
    ])
    out = []
    for js, rep in _iter_reports(board):
        if rep["run"].get("mode") != "latency" or not _pinned(rep):
            continue
        side = js.with_name(js.stem + "_timings.npy")
        if not side.is_file():
            continue
        t = np.load(side).astype(np.float64)
        if t.size < 100:
            continue
        route, device, prec = _route_key(rep)
        vals = np.quantile(t, q)
        for qq, vv in zip(q, vals):
            out.append(dict(board=board, route=route, device=device,
                            precision=prec,
                            experiment=_experiment_id(rep),
                            threads=(rep["run"].get("params") or {}).get("threads", ""),
                            repetition=rep["run"].get("repetition", 0),
                            quantile=round(float(qq), 6),
                            latency_ms=round(float(vv), 6),
                            n_samples=int(t.size)))
    if out:
        C.write_csv(f"latency_cdf_{board}",
                    ["board", "route", "device", "precision", "experiment", "threads",
                     "repetition", "quantile", "latency_ms", "n_samples"], out)


def derive_frame_quality(board: str):
    """Per-frame quality -> mean + bootstrap interval over frames.

    Rule 1 is applied upstream by the measurement code, which clips before
    computing; this step only aggregates. The bootstrap interval is what makes
    a 100-frame split say something, and it cannot be recovered from a mean,
    which is why the per-frame values exist at all.
    """
    out = []
    seen = set()
    for js, rep in _iter_reports(board):
        if not _pinned(rep):
            continue
        side = js.with_name(js.stem + "_frames.npz")
        if not side.is_file():
            continue
        # Quality is deterministic given fixed weights (README rule 4), so the
        # repetitions and the thread sweep all produce identical frames. Keep
        # one row per (route, precision, experiment) instead of eight copies.
        route_, device_, prec_ = _route_key(rep)
        eng = _engine(rep)
        # The composition is part of the identity. Two builds under one label
        # are two engines and must not be collapsed into one row.
        key = (route_, device_, prec_, _experiment_id(rep), eng["precision_map"])
        if key in seen:
            continue
        seen.add(key)
        d = np.load(side)
        route, device, prec = _route_key(rep)
        exp = _experiment_id(rep)
        row = dict(board=board, route=route, device=device, precision=prec,
                   experiment=exp,
                   paper_label=_meta(exp).get("paper_label", ""),
                   n_frames=0)
        row.update(_engine(rep))
        for key, stem in (("psnr_db_vs_gt", "psnr_gt"),
                          ("ssim_vs_gt", "ssim_gt"),
                          ("mae_vs_gt", "mae_gt"),
                          ("psnr_db_vs_fp32", "psnr_fp32"),
                          ("ssim_vs_fp32", "ssim_fp32")):
            if key not in d:
                continue
            v = np.asarray(d[key], dtype=np.float64).ravel()
            v = v[np.isfinite(v)]
            # A route compared against itself saturates the metric; the
            # measurement code stores 100 dB as the sentinel. Not a value.
            if stem.endswith("fp32") and v.size and float(np.median(v)) >= 99.0:
                continue
            if v.size == 0:
                continue
            row["n_frames"] = max(row["n_frames"], int(v.size))
            lo, hi = C.bootstrap_ci(v.tolist())
            row[f"{stem}_mean"] = round(float(v.mean()), 4)
            row[f"{stem}_lo"] = round(float(lo), 4) if lo is not None else ""
            row[f"{stem}_hi"] = round(float(hi), 4) if hi is not None else ""
        if row["n_frames"]:
            out.append(row)
    if out:
        cols = ["board", "route", "device", "precision", "experiment",
                "paper_label", "n_frames",
                "layers_int8", "layers_fp16", "layers_fp32", "layers_total",
                "int8_layer_fraction", "precision_map", "engine_name",
                "trt_version", "build_commit"]
        for stem in ("psnr_gt", "ssim_gt", "mae_gt", "psnr_fp32", "ssim_fp32"):
            cols += [f"{stem}_mean", f"{stem}_lo", f"{stem}_hi"]
        out = [{c: r.get(c, "") for c in cols} for r in out]
        C.write_csv(f"frame_quality_{board}", cols, out)


def derive_qualitative_strips(board: str):
    """Predicted frames -> a small NPZ per image class for the strip figures.

    Only a handful of frames are kept, so the figure regenerates without the
    campaign. Note this writes .npz, not .csv: images are the one thing that
    cannot go in a tidy table.
    """
    picks: dict[str, dict] = {}
    for js, rep in _iter_reports(board):
        if not _pinned(rep):
            continue
        side = js.with_name(js.stem + "_preds.npz")
        if not side.is_file():
            continue
        exp = _experiment_id(rep)
        route, device, prec = _route_key(rep)
        picks.setdefault(exp, {})[f"{route}_{prec}".strip("_")] = side

    n_keep = 3
    for exp, routes in sorted(picks.items()):
        if exp in C.LEGACY_EXPERIMENTS:
            continue
        src = C.RAW / "datasets" / exp / "source" / "test_images.npz"
        if not src.is_file():
            continue
        s = np.load(src)
        idx = np.linspace(0, s["originals"].shape[0] - 1, n_keep).astype(int)
        bundle = {
            "index": idx,
            "originals": np.clip(s["originals"][idx], 0, 1),
            "reconstructions": np.clip(s["reconstructions"][idx], 0, 1),
            "fp32": np.clip(s["denoised"][idx], 0, 1),
        }
        for key, path in sorted(routes.items()):
            d = np.load(path)
            arr = d[list(d.keys())[0]]
            bundle[key] = np.clip(np.asarray(arr)[idx], 0, 1)
        outp = C.DATA / f"strip_{exp}.npz"
        C.DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outp, **bundle)
        print(f"  wrote figures/data/{outp.name}  ({', '.join(bundle)})")


def derive_power_traces(board: str):
    """Per-sample telemetry -> one downsampled trace per route.

    The samples live in a `_power.npz` sidecar, keyed `{idle,active}_<rail>`
    plus `{idle,active}_t_s`, at roughly 100 Hz. Idle and load are separate
    passes, so the trace is stitched with the idle segment first and the load
    segment following it, and the boundary recorded.

    `rail4` is dropped: it was an artefact of pairing a shunt-voltage register
    with a summed-current register by index, multiplying microvolts by
    milliamps and calling the result watts. Reports written before 2026-08-28
    still carry the key.
    """
    # Logical rail -> candidate keys in the sidecar, per board. The Ultra96
    # keys changed on 2026-08-30 when the second IRPS5401 controller was
    # added (bare `irps5401.*` is the processing-system controller).
    TRACE_RAILS = {
        "jetson": {"VDD_IN": ("ina3221.VDD_IN",),
                   "VDD_CPU_GPU_CV": ("ina3221.VDD_CPU_GPU_CV",),
                   "VDD_SOC": ("ina3221.VDD_SOC",)},
        "ultra96v2": {"board_input": ("ir38060.pout1",),
                      "ps_core": ("irps5401_ps.pout4", "irps5401.pout4"),
                      "pl_core": ("irps5401_pl.pout4",),
                      "ps_ctrl_in": ("irps5401_ps.pin*", "irps5401.pin*"),
                      "pl_ctrl_in": ("irps5401_pl.pin*",)},
    }
    spec = TRACE_RAILS.get(board)
    if spec is None:
        return
    RAILS = tuple(spec)

    def _series(d, seg, cands):
        """First candidate present; `pin*` sums the five input channels."""
        for c in cands:
            if c.endswith("pin*"):
                ks = [f"{seg}_{c[:-1]}{i}" for i in range(1, 6)]
                if all(k in d.files for k in ks):
                    return sum(np.asarray(d[k], dtype=np.float64) for k in ks)
            elif f"{seg}_{c}" in d.files:
                return np.asarray(d[f"{seg}_{c}"], dtype=np.float64)
        return None

    # One representative trace per route; where several sidecars exist the
    # one exposing the most rails wins (the dual-controller Ultra96 sessions).
    best = {}
    for js, rep in _iter_reports(board):
        if rep["run"].get("mode") != "energy" or not _pinned(rep):
            continue
        side = js.with_name(js.stem + "_power.npz")
        if not side.is_file():
            continue
        route, device, prec = _route_key(rep)
        exp = _experiment_id(rep)
        key = (route, prec, exp)
        n_rails = sum(_series(np.load(side), "idle", spec[r_]) is not None
                      for r_ in RAILS)
        if key not in best or n_rails > best[key][0]:
            best[key] = (n_rails, side, route, prec, exp)
    out = []
    for n_rails, side, route, prec, exp in best.values():
        d = np.load(side)
        t0 = None
        for seg in ("idle", "active"):
            tk = f"{seg}_t_s"
            if tk not in d.files:
                continue
            t = np.asarray(d[tk], dtype=np.float64)
            if t0 is None:
                t0 = 0.0
            else:
                t0 = out[-1]["t_s"] if out else 0.0
            offset = (out[-1]["t_s"] + 0.05) if out and out[-1]["experiment"] == exp \
                and out[-1]["route"] == route and seg == "active" else 0.0
            step = max(1, t.size // 400)
            for i in range(0, t.size, step):
                row = dict(board=board, route=route, precision=prec,
                           experiment=exp, segment=seg,
                           t_s=round(float(t[i]) + offset, 4))
                for r_ in RAILS:
                    s_ = _series(d, seg, spec[r_])
                    row[r_] = round(float(s_[i]), 4) if s_ is not None else ""
                out.append(row)
    if out:
        C.write_csv(f"power_trace_{board}",
                    ["board", "route", "precision", "experiment", "segment",
                     "t_s"] + list(RAILS), out)
    else:
        print("  (no _power.npz sidecars found; fig:power_trace stays pending)")


SETTLE_S = 2.5
"""Seconds discarded from the start of each telemetry segment.

Both passes begin with a transient, in opposite directions: the idle segment
starts high and decays as the board settles, the load segment starts low and
ramps as the workload spins up. Averaging either in full biases it, and both
biases push P_act - P_idle the same way, so the dynamic energy is understated
by about 6 %.

2.5 s is chosen from the data, not by eye: the median time to settle within
1 % of the final level is 1.67 s and the 95th percentile is 1.99 s, and beyond
2.5 s the segment mean moves by under 1 mW. The same window is applied to both
segments so the correction stays symmetric.
"""


def derive_energy_corrected(board: str):
    """Recompute the energy figures from the settled part of each segment.

    The reports carry averages over the whole segment, computed on the board.
    This does not replace them -- both are emitted, so the size of the
    correction stays visible -- but the settled values are what the figures
    use.
    """
    # Each board names its rails differently, and only one of them exposes a
    # clean compute domain. On the Orin, VDD_CPU_GPU_CV is the processors and
    # the accelerator. On the Ultra96, ir38060.pout1 is the board input and
    # irps5401.pout4 is the rail that responds to compute activity -- it rises
    # 0.451 W above idle when the processors are saturated and 0.093 W when the
    # overlay runs, a factor of 4.8, while the board input rises by the same
    # 1.263 W either way. Its identity is inferred from that response and not
    # from documentation; confirming it against the board schematic is pending.
    RAILS_BY_BOARD = {
        "jetson":    {"input": "VDD_IN", "compute": "VDD_CPU_GPU_CV",
                      "other": ("VDD_SOC",)},
        "ultra96v2": {"input": "ir38060.pout1", "compute": "irps5401.pout4",
                      "other": ("irps5401.pout3", "irps5401.pout1")},
    }
    spec = RAILS_BY_BOARD.get(board)
    if spec is None:
        return
    RAILS = (spec["input"], spec["compute"]) + spec["other"]
    out = []
    for js, rep in _iter_reports(board):
        if rep["run"].get("mode") != "energy" or not _pinned(rep):
            continue
        side = js.with_name(js.stem + "_power.npz")
        em = rep.get("energy_mode") or {}
        if not side.is_file() or not em:
            continue
        d = np.load(side)
        if "idle_t_s" not in d.files or "active_t_s" not in d.files:
            continue
        route, device, prec = _route_key(rep)
        fps = ((em.get("active") or {}).get("fps_sustained"))
        if not fps:
            continue
        row = dict(board=board, route=route, device=device, precision=prec,
                   experiment=_experiment_id(rep),
                   threads=(rep["run"].get("params") or {}).get("threads", ""),
                   repetition=rep["run"].get("repetition", 0),
                   fps_sustained=round(float(fps), 2),
                   settle_s=SETTLE_S)
        for rail in RAILS:
            ki = ka = None
            for pre in ("ina3221.", ""):
                if f"idle_{pre}{rail}" in d.files:
                    ki, ka = f"idle_{pre}{rail}", f"active_{pre}{rail}"
                    break
            if ki is None or ka not in d.files:
                continue
            ti = np.asarray(d["idle_t_s"], float)
            ta = np.asarray(d["active_t_s"], float)
            pi = np.asarray(d[ki], float)[ti >= SETTLE_S]
            pa = np.asarray(d[ka], float)[ta >= SETTLE_S]
            if pi.size < 50 or pa.size < 50:
                continue
            p_idle, p_act = float(pi.mean()), float(pa.mean())
            tag = ("input" if rail == spec["input"]
                   else "compute" if rail == spec["compute"] else rail)
            row[f"{tag}_idle_w"] = round(p_idle, 4)
            row[f"{tag}_active_w"] = round(p_act, 4)
            if rail == spec["input"]:
                row["e_total_mj"] = round(1000.0 * p_act / fps, 4)
                row["e_dynamic_mj"] = round(1000.0 * (p_act - p_idle) / fps, 4)
                row["frames_per_joule"] = round(fps / p_act, 2)
            if rail == spec["compute"]:
                row["e_total_compute_mj"] = round(1000.0 * p_act / fps, 4)
                row["e_dynamic_compute_mj"] = round(1000.0 * (p_act - p_idle) / fps, 4)
        # Ultra96: the front-end regulator reports in 1.25 W steps (its
        # settled mean is 5.00 W idle and 6.25 W under every load), so the
        # input-plane increment is one quantisation step, not a measurement.
        # The IRPS5401 input registers resolve 31 mW; their sum over both
        # controllers (processing system + programmable logic) carries the
        # dynamic term. Sessions before 2026-08-30 14:39 read only the PS
        # controller, whose fabric companion moves under 0.08 W under
        # processor load but carries the overlay's draw.
        if board == "ultra96v2":
            ctrl_i = ctrl_a = 0.0
            ti = np.asarray(d["idle_t_s"], float)
            ta = np.asarray(d["active_t_s"], float)
            n_ctrl = 0
            for pre in ("irps5401_ps", "irps5401_pl", "irps5401"):
                ks = [f"{pre}.pin{i}" for i in range(1, 6)]
                if all(f"idle_{k}" in d.files for k in ks):
                    n_ctrl += 1
                    for k in ks:
                        ctrl_i += float(np.asarray(d[f"idle_{k}"], float)[ti >= SETTLE_S].mean())
                        ctrl_a += float(np.asarray(d[f"active_{k}"], float)[ta >= SETTLE_S].mean())
            if n_ctrl:
                row["ctrl_in_idle_w"] = round(ctrl_i, 4)
                row["ctrl_in_active_w"] = round(ctrl_a, 4)
                row["ctrl_in_controllers"] = n_ctrl
                row["e_dynamic_ctrl_mj"] = round(1000.0 * (ctrl_a - ctrl_i) / fps, 4)
            row["session_tag"] = rep["run"].get("tag") or ""
        # what the uncorrected report said, so the delta stays visible
        der = em.get("derived") or {}
        row["e_total_mj_uncorrected"] = round(1000.0 * (der.get("energy_per_frame_j") or 0), 4)
        row["e_dynamic_mj_uncorrected"] = round(1000.0 * (der.get("dynamic_energy_per_frame_j") or 0), 4)
        out.append(row)
    if out:
        cols = ["board", "route", "device", "precision", "experiment", "threads",
                "repetition", "fps_sustained", "settle_s"]
        for tag in ("input", "compute"):
            cols += [f"{tag}_idle_w", f"{tag}_active_w"]
        cols += ["e_total_mj", "e_dynamic_mj", "frames_per_joule",
                 "e_total_compute_mj", "e_dynamic_compute_mj",
                 "e_total_mj_uncorrected", "e_dynamic_mj_uncorrected"]
        if board == "ultra96v2":
            cols += ["ctrl_in_idle_w", "ctrl_in_active_w", "ctrl_in_controllers",
                     "e_dynamic_ctrl_mj", "session_tag"]
        C.write_csv(f"energy_{board}", cols,
                    [{c: r.get(c, "") for c in cols} for r in out])


def derive_engines(board: str):
    """One row per distinct engine actually built, per experiment.

    Exists so that a build that silently changes composition cannot hide. If a
    single engine name appears with two precision maps, this table shows it and
    `audit.py` fails on it.
    """
    seen = {}
    for js, rep in _iter_reports(board):
        if not _pinned(rep):
            continue
        eng = _engine(rep)
        if not eng["precision_map"]:
            continue
        route, device, prec = _route_key(rep)
        exp = _experiment_id(rep)
        key = (exp, prec, eng["precision_map"], eng["build_commit"])
        if key in seen:
            seen[key]["n_runs"] += 1
            continue
        row = dict(board=board, experiment=exp, route=route, precision=prec)
        row.update(eng)
        row["n_runs"] = 1
        seen[key] = row
    if seen:
        rows = sorted(seen.values(), key=lambda r: (r["experiment"], r["precision"]))
        cols = ["board", "experiment", "route", "precision", "layers_int8",
                "layers_fp16", "layers_fp32", "layers_total",
                "int8_layer_fraction", "precision_map", "engine_name",
                "engine_n_layers", "trt_version", "build_commit", "deployed_at",
                "n_runs"]
        C.write_csv(f"engines_{board}", cols, [{c: r.get(c, "") for c in cols}
                                               for r in rows])


def derive_fp32_reference():
    """PSNR of the shipped single-precision output against ground truth.

    Needed at every grid point, because the error-floor model takes it as an
    input and the boards were only measured in single precision at a few. It is
    legitimate to take it from the dataset: FP32 inference is deterministic, so
    the host output is what the board would produce -- and where both exist they
    agree to the fourth decimal (36.2584 against 36.25, 41.5321 against 41.53,
    42.1856 against 42.19).

    Computed with the same convention as every other metric here: clipped to
    [0, 1] before evaluation.
    """
    out = []
    root = C.RAW / "datasets"
    if not root.is_dir():
        return
    for d in sorted(root.iterdir()):
        f = d / "source" / "test_images.npz"
        if not d.is_dir() or d.name.startswith("_") or not f.is_file():
            continue
        z = np.load(f)
        a = np.clip(np.asarray(z["denoised"], np.float64), 0, 1)
        b = np.clip(np.asarray(z["originals"], np.float64), 0, 1)
        mse = ((a - b) ** 2).mean(axis=(1, 2))
        mse = mse[mse > 0]
        if not mse.size:
            continue
        out.append(dict(experiment=d.name,
                        n_frames=int(mse.size),
                        psnr_fp32_gt_db=round(float((10 * np.log10(1.0 / mse)).mean()), 4)))
    if out:
        C.write_csv("fp32_reference",
                    ["experiment", "n_frames", "psnr_fp32_gt_db"], out)


def derive_quality_reference():
    """PSNR and SSIM against ground truth of the linear reconstruction and of
    the shipped single-precision output, over the 100 evaluation frames.

    The "before and after denoising" reference for fig:quality. Same metric
    conventions as the campaign (aspir_unet/measurement/metrics.py): clip to
    [0, 1], PSNR with data_range 1, SSIM from scikit-image with its default
    window. Bootstrap intervals over frames, as for every other quality
    figure.
    """
    from skimage.metrics import structural_similarity as ssim
    out = []
    root = C.RAW / "datasets"
    if not root.is_dir():
        return
    for d in sorted(root.iterdir()):
        f = d / "source" / "test_images.npz"
        if not d.is_dir() or d.name.startswith("_") or not f.is_file():
            continue
        z = np.load(f)
        gt = np.clip(np.asarray(z["originals"], np.float64), 0, 1)
        row = dict(experiment=d.name, n_frames=int(gt.shape[0]))
        # The target carries a multiplicative speckle of level 0.05 (see the
        # aspir-edge-data README): per-pixel noise variance 0.05^2 * I^2, whose realisation
        # is not recoverable from the measurements. Its power bounds the PSNR
        # any output can reach against this target.
        row["speckle_bound_db"] = round(float(-10 * np.log10(0.05 ** 2 * np.mean(gt ** 2))), 2)
        for stem, key in (("recon", "reconstructions"), ("fp32", "denoised")):
            a = np.clip(np.asarray(z[key], np.float64), 0, 1)
            mse = ((a - gt) ** 2).mean(axis=(1, 2))
            psnr = [10 * np.log10(1.0 / m) if m > 0 else None for m in mse]
            ss = [float(ssim(a[i], gt[i], data_range=1.0)) for i in range(gt.shape[0])]
            for name, vals in ((f"psnr_{stem}", psnr), (f"ssim_{stem}", ss)):
                v = [x for x in vals if x is not None]
                lo, hi = C.bootstrap_ci(v)
                row[f"{name}_mean"] = round(float(np.mean(v)), 4)
                row[f"{name}_lo"] = round(lo, 4)
                row[f"{name}_hi"] = round(hi, 4)
        out.append(row)
    if out:
        C.write_csv("quality_reference", list(out[0].keys()), out)


def derive_sensing(board: str = "jetson"):
    """Sensing configuration per experiment, from the datasets' own records.

    This is what the sensing table in the paper is built from, and it is where
    the projections-versus-patterns distinction lives.

    Polarity is read from the mask values themselves rather than from the
    configuration string, and the two are cross-checked: they encode the same
    fact, so a disagreement means one of them is wrong and the script should
    stop rather than pick a winner. (A previous version of this function
    tested `"bi" in polarity`, which is true of "binary" -- hence the check.)
    """
    out = []
    root = C.RAW / "datasets"
    if not root.is_dir():
        return
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        meta = _meta(d.name)
        cfgp = d / "source" / "test_config.json"
        cfg = json.load(open(cfgp)) if cfgp.is_file() else {}
        m = cfg.get("M_real") or meta.get("M_real")
        if m is None:
            continue

        # ground truth: what the modulator is actually asked to display
        maskp = d / "source" / "masks.npz"
        bipolar = None
        if maskp.is_file():
            mk = np.load(maskp)
            arr = mk[list(mk.keys())[0]]
            bipolar = bool(np.any(np.asarray(arr) < 0))

        declared = (cfg.get("mask_polarity") or meta.get("mask_polarity") or "").strip().lower()
        if declared in ("bipolar", "binary"):
            declared_bipolar = declared == "bipolar"
            if bipolar is None:
                bipolar = declared_bipolar
            elif bipolar != declared_bipolar:
                raise SystemExit(
                    f"{d.name}: mask values say "
                    f"{'bipolar' if bipolar else 'binary'} but test_config.json "
                    f"declares {declared!r}. Resolve before plotting anything.")
        if bipolar is None:
            raise SystemExit(f"{d.name}: polarity undeterminable; refusing to guess")

        proj = int(m) * (2 if bipolar else 1)

        # M_over_N_real is stored as a fraction, not a percentage
        ratio = cfg.get("M_over_N_real", meta.get("M_over_N_real"))
        ratio_pct = 100.0 * float(ratio) if ratio is not None else 100.0 * int(m) / 4096

        out.append(dict(
            experiment=d.name,
            paper_label=meta.get("paper_label", ""),
            image_class=C.image_class(d.name),
            family=C.family(d.name),
            polarity="bipolar" if bipolar else "binary",
            m_patterns=int(m),
            binary_projections=proj,
            sampling_ratio_pct=round(ratio_pct, 3),
            t_acq_ms=round(proj / C.DMD_KHZ, 3),
        ))
    if out:
        C.write_csv("sensing", list(out[0].keys()), out)


def derive_patterns():
    """One representative pattern per family, for the overview figure.

    Cake-cutting index 5 (a low-order Hadamard block), scatter index 0 and the
    sweep bar with most energy through the centre of the field. Values are kept as stored: bipolar
    +-1 for Hadamard, 0/1 for the binary families.
    """
    root = C.RAW / "datasets"
    picks = {"cch": ("irbeam_64_unet_cch_8pct_8_16", 5),
             "scatter": ("irbeam_64_unet_scatter_8_16", 0),
             "sweep": ("irbeam_64_unet_sweep_4degrees_2_bar_width_1_stride_8_16", None)}
    out = {}
    for fam, (exp, idx) in picks.items():
        m = np.load(root / exp / "source" / "masks.npz")["masks"]
        if idx is None:   # sweep: the bar with most energy through the centre
            idx = int(np.argmax(m[:, 28:36, 28:36].sum(axis=(1, 2))))
        out[fam] = np.asarray(m[idx], dtype=np.float32)
    np.savez_compressed(C.DATA / "patterns.npz", **out)
    print("  wrote figures/data/patterns.npz")


def main():
    if not C.RAW.is_dir():
        sys.exit("data/ not present: nothing to derive. The CSVs under "
                 "figures/data/ are committed and the plots run without it.")
    boards = [p.name for p in (C.RAW / "results").iterdir() if p.is_dir()] \
        if (C.RAW / "results").is_dir() else []
    print("deriving sensing configuration")
    derive_sensing()
    derive_patterns()
    derive_fp32_reference()
    derive_quality_reference()
    for b in boards:
        n = sum(1 for _ in _iter_reports(b))
        if not n:
            print(f"\n{b}: no reports yet, skipping")
            continue
        print(f"\n{b}: {n} reports")
        derive_engines(b)
        derive_energy_corrected(b)
        derive_latency_cdf(b)
        derive_frame_quality(b)
        derive_qualitative_strips(b)
        derive_power_traces(b)


if __name__ == "__main__":
    main()
