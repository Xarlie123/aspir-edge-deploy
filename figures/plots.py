#!/usr/bin/env python3
"""figures/data/*.csv  ->  figures/*.pdf

Reads nothing outside figures/data/, so every figure regenerates on a machine
without the raw campaign. Run `derive.py` first only when new measurements
land; the CSVs it writes are committed.

    python plots.py            # all figures
    python plots.py --only precision latency
    python plots.py --list
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C

plt = C.setup_mpl()

BOARD = "jetson"          # extend to a loop when the second board is measured


# Colormap for image frames in the methodology figures (fig:overview thumbnails
# and fig:samples). Perceptually uniform, prints legibly in greyscale.
IMAGE_CMAP = "inferno"

def _read(stem):
    p = C.DATA / f"{stem}.csv"
    if not p.is_file():
        return None
    return [{k: C._num(v) for k, v in r.items()} for r in csv.DictReader(open(p))]


def _label(route, device, precision):
    return C.route_label({"route": route, "device": device,
                          "precision": precision or ""})


def _style(label):
    return dict(C.STYLE.get(label, dict(color="#333333", marker="o", ls="-", ms=4)))


def _sensing_index():
    rows = _read("sensing") or []
    return {r["experiment"]: r for r in rows}


# ---------------------------------------------------------------- precision
def fig_precision():
    """The paper's organising finding: on this network INT8 does not pay on the
    GPU. Latency and fidelity on one pair of axes so the trade-off is visible
    in a single glance rather than inferred across two tables."""
    q = _read(f"frame_quality_{BOARD}")
    m = C.load()
    if not q:
        return print("  skip precision: no frame quality yet")

    op = "irbeam_64_unet_cch_8pct_8_16"          # the operating point
    prec_order = ["fp32", "fp16", "int8", "int8_nofp16"]

    lat = {}
    for r in m:
        if r["route"] != "gpu_trt" or r["experiment"] != op or not C.is_timing_pass(r):
            continue
        lat.setdefault(r["precision"], []).append(r["lat_median_ms"])
    lat = {k: C.median(v) for k, v in lat.items()}

    qual = {r["precision"]: r for r in q
            if r["route"] == "gpu_trt" and r["experiment"] == op}

    have = [p for p in prec_order if p in lat and p in qual]
    if not have:
        return print("  skip precision: operating point not covered")

    fig, ax = plt.subplots(figsize=(C.COL_W, C.COL_H * 0.85))
    for p in have:
        lab = C.PRECISION_LABEL.get(p, p)
        st = _style(f"Orin GPU ({lab})")
        ax.scatter(lat[p], qual[p]["psnr_gt_mean"], s=45,
                   color=st["color"], marker=st["marker"], zorder=3,
                   edgecolors="white", linewidths=0.5, label=lab)
        lo, hi = qual[p]["psnr_gt_lo"], qual[p]["psnr_gt_hi"]
        if lo not in (None, "") and hi not in (None, ""):
            ax.plot([lat[p], lat[p]], [lo, hi], color=st["color"], lw=1.0, zorder=2)
        ax.annotate(lab, (lat[p], qual[p]["psnr_gt_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7,
                    color=st["color"])
    ax.set_xlabel("Median inference latency (ms), lower is better")
    ax.set_ylabel("PSNR vs ground truth (dB)")
    ax.set_title("Embedded GPU: precision selection at the operating point", pad=6)
    ax.margins(x=0.18, y=0.18)
    C.save(fig, "fig_precision")
    plt.close(fig)


# ------------------------------------------------------------------ latency
def fig_latency():
    """Latency at the operating point, two panels with the same rows.

    Left: median latency per execution path of tab:main, logarithmic. Right:
    the tail above the median, in microseconds and linear, to the 99th
    (dark) and 99.9th (light) percentile. Speed and determinism are the two
    claims and they need different axes, so they get different panels. Quantiles are
    the median over the three repetitions of each path; processors at four
    threads. The line stops at the 99.9th percentile (10 frames of 10,000):
    the single slowest frame varies by a factor of three between repetitions
    of the same path and is not a statistic.
    """
    op = "irbeam_64_unet_cch_8pct_8_16"
    paths = [  # (board, route, precision, threads, label)
        ("ultra96v2", "cpu_neon", "fp32", "4", "Ultra96-V2  CPU FP32"),
        ("ultra96v2", "cpu_neon", "int8", "4", "Ultra96-V2  CPU INT8"),
        ("ultra96v2", "dpu_int8", "", "", "Ultra96-V2  DPU INT8"),
        ("jetson", "cpu_neon", "fp32", "4", "Orin NX  CPU FP32"),
        ("jetson", "cpu_neon", "int8", "4", "Orin NX  CPU INT8"),
        ("jetson", "gpu_trt", "int8_strict", "", "Orin NX  GPU INT8"),
        ("jetson", "gpu_trt", "fp16", "", "Orin NX  GPU FP16"),
    ]
    data = {}
    for board in ("ultra96v2", "jetson"):
        rows = _read(f"latency_cdf_{board}") or []
        for r in rows:
            if r["experiment"] != op:
                continue
            key = (board, r["route"], r["precision"] or "", str(r.get("threads") or "").split(".")[0])
            data.setdefault(key, {}).setdefault(float(r["quantile"]), []).append(float(r["latency_ms"]))
    if not data:
        return print("  skip latency: no CDF data")

    def qv(curve, target):
        qs = sorted(curve)
        qn = min(qs, key=lambda x: abs(x - target))
        return C.median(curve[qn])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(C.COL_W, C.COL_H * 0.72), sharey=True,
                                 gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.08})
    ys, labels = [], []
    for k, (board, route, prec, thr, lab) in enumerate(paths):
        curve = data.get((board, route, prec, thr))
        if not curve:
            print(f"    no data for {lab}")
            continue
        y = len(paths) - k
        med, p99, p999 = (qv(curve, q) for q in (0.5, 0.99, 0.999))
        col = "#8A4FBF" if board == "ultra96v2" else "#C4451C"
        if route == "cpu_neon":
            col = "#4C4C4C" if board == "jetson" else "#7A7A7A"
        a1.barh(y, med, height=0.62, color=col, zorder=3)
        a2.barh(y, (p999 - med) * 1000, height=0.62, color=col, alpha=0.3, zorder=2)
        a2.barh(y, (p99 - med) * 1000, height=0.62, color=col, zorder=3)
        ys.append(y); labels.append(lab)
    a1.set_yticks(ys); a1.set_yticklabels(labels, fontsize=6.8)
    a1.set_xscale("log")
    ticks = [0.3, 1, 3, 10, 30]
    a1.set_xticks(ticks); a1.set_xticklabels([f"{v:g}" for v in ticks])
    a1.xaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
    a1.set_xlim(0.1, 30)
    a1.set_xlabel("Median latency (ms)")
    a2.set_xlabel("Tail above the median (µs)")
    a2.set_xlim(0, None)
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.25)
        ax.axhline(len(paths) - 2.5, color="#BBBBBB", lw=0.6)
    a2.annotate("dark: 99th percentile   light: 99.9th", (1.0, -0.3), xycoords="axes fraction",
                ha="right", fontsize=6, color="#666666")
    C.save(fig, "fig_latency")
    plt.close(fig)
    plt.close(fig)


# ------------------------------------------------------------------ quality
def fig_quality():
    """Denoising quality against ground truth across the cake-cutting sweep:
    the linear reconstruction before denoising, the single-precision output
    after it, and the two quantised accelerator paths. PSNR and SSIM, one row
    per image class, every point the mean over the 100 evaluation frames with
    its bootstrap interval.

    The reference rows come from the datasets themselves (quality_reference.csv,
    derive.py); FP32 inference is deterministic, so the host output is what the
    boards would produce. The accelerator rows are what the boards produced.
    """
    ref = _read("quality_reference")
    sens = _sensing_index()
    if not ref:
        return print("  skip quality: no quality_reference.csv yet")
    boards = (("ultra96v2", "dpu_int8", "", "ZU3EG DPU (INT8)"),
              ("jetson", "gpu_trt", "int8_strict", "Orin GPU (INT8)"))
    fig, axes = plt.subplots(2, 2, figsize=(C.COL_W, C.COL_H * 1.55), sharex=True)
    for row_ax, cls in zip(axes, ("irbeam", "celebA")):
        # reference series from the datasets
        for stem, lab, st in (("recon", "Linear reconstruction", _style("Linear recon.")),
                              ("fp32", "U-Net, FP32", dict(color="#222222", marker="o", ls="-", ms=4))):
            pts = sorted((sens[r["experiment"]]["sampling_ratio_pct"], r)
                         for r in ref if r["experiment"] in sens
                         and sens[r["experiment"]]["image_class"] == cls
                         and sens[r["experiment"]]["family"] == "cch")
            for ax, key in zip(row_ax, ("psnr", "ssim")):
                x = [p[0] for p in pts]
                y = [p[1][f"{key}_{stem}_mean"] for p in pts]
                lo = [y[k] - p[1][f"{key}_{stem}_lo"] for k, p in enumerate(pts)]
                hi = [p[1][f"{key}_{stem}_hi"] - y[k] for k, p in enumerate(pts)]
                ax.errorbar(x, y, yerr=[lo, hi], capsize=2, elinewidth=0.7, lw=1.2,
                            color=st["color"], marker=st["marker"], ls=st["ls"],
                            ms=st["ms"], label=lab)
        # what the boards produced
        for board, route, prec, lab in boards:
            q = _read(f"frame_quality_{board}") or []
            st = _style(lab)
            pts = {}
            for r in q:
                s_ = sens.get(r["experiment"])
                if (not s_ or s_["image_class"] != cls or s_["family"] != "cch"
                        or r["route"] != route or (r["precision"] or "") != prec):
                    continue
                pts.setdefault(s_["sampling_ratio_pct"], []).append(r)
            x = sorted(pts)
            for ax, key in zip(row_ax, ("psnr_gt", "ssim_gt")):
                y = [C.median([r[f"{key}_mean"] for r in pts[k]]) for k in x]
                lo = [y[i] - C.median([r[f"{key}_lo"] for r in pts[k]]) for i, k in enumerate(x)]
                hi = [C.median([r[f"{key}_hi"] for r in pts[k]]) - y[i] for i, k in enumerate(x)]
                ax.errorbar(x, y, yerr=[lo, hi], capsize=2, elinewidth=0.7, lw=1.2,
                            color=st["color"], marker=st["marker"], ls=st["ls"],
                            ms=st["ms"], label=lab)
        bound = [r["speckle_bound_db"] for r in ref if r["experiment"] in sens
                 and sens[r["experiment"]]["image_class"] == cls and r.get("speckle_bound_db")]
        if bound:
            lo_y, hi_y = row_ax[0].get_ylim()
            if lo_y < bound[0] < hi_y + 3:
                row_ax[0].axhline(bound[0], color="#888888", ls=(0, (1, 2)), lw=0.9,
                                  label="Speckle bound of the target")
        row_ax[0].set_ylabel(f"{C.CLASS_LABEL[cls]}\nPSNR vs ground truth (dB)")
        row_ax[1].set_ylabel("SSIM vs ground truth")
    for ax in axes[1]:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"Sampling ratio $M/N$ (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.06), handlelength=2.4)
    fig.text(0.5, -0.10,
             "Mean over the 100 evaluation frames; error bars: 95 % bootstrap interval. "
             "Cake-cutting family.",
             ha="center", fontsize=6, color="#666666")
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    C.save(fig, "fig_quality")
    plt.close(fig)


# --------------------------------------------------------------- fidelity
def fig_fidelity():
    """Quantisation fidelity of the two accelerator paths: the eight-bit output
    against the single-precision output of the same network, in PSNR (left) and
    SSIM (right), across the cake-cutting sweep and both image classes.

    Both boards on one axis. The ZU3EG DPU has one precision (INT8, per-tensor
    power-of-two scales); the Orin GPU path is the engine with every layer
    pinned to INT8 (`int8_strict`), the only GPU engine that is a quantised
    path in the sense the DPU is one. The paired comparison is where the
    statistical power lives with 100 frames, so the bars are its bootstrap
    interval.
    """
    sens = _sensing_index()
    paths = (("ultra96v2", "dpu_int8", "", "ZU3EG DPU (INT8)"),
             ("jetson", "gpu_trt", "int8_strict", "Orin GPU (INT8)"))
    fig, axes = plt.subplots(1, 2, figsize=(C.COL_W, C.COL_H * 0.8))
    any_pt = False
    for board, route, prec, lab in paths:
        q = _read(f"frame_quality_{board}") or []
        st = _style(lab)
        for cls, mk, cls_ls in (("irbeam", "o", "-"), ("celebA", "s", "--")):
            pts = {}
            for r in q:
                s_ = sens.get(r["experiment"])
                if (not s_ or s_["image_class"] != cls or s_["family"] != "cch"
                        or r["route"] != route or (r["precision"] or "") != prec
                        or r.get("psnr_fp32_mean") in (None, "")):
                    continue
                pts.setdefault(s_["sampling_ratio_pct"], []).append(r)
            if not pts:
                continue
            any_pt = True
            x = sorted(pts)
            for ax, key in zip(axes, ("psnr_fp32", "ssim_fp32")):
                y = [C.median([r[f"{key}_mean"] for r in pts[k]]) for k in x]
                lo = [y[i] - C.median([r[f"{key}_lo"] for r in pts[k]]) for i, k in enumerate(x)]
                hi = [C.median([r[f"{key}_hi"] for r in pts[k]]) - y[i] for i, k in enumerate(x)]
                ax.errorbar(x, y, yerr=[lo, hi], capsize=2, elinewidth=0.7,
                            color=st["color"], marker=mk, ls=cls_ls, ms=4.5, lw=1.2,
                            label=f"{lab}, {C.CLASS_LABEL[cls]}")
    if not any_pt:
        plt.close(fig)
        return print("  skip fidelity: no vs-FP32 metrics present")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"Sampling ratio $M/N$ (%)")
    axes[0].set_ylabel("PSNR vs FP32 output (dB)")
    axes[1].set_ylabel("SSIM vs FP32 output")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.16), handlelength=2.4)
    fig.text(0.5, -0.21,
             "Error bars: 95 % bootstrap interval over 100 evaluation frames. "
             "Cake-cutting family.",
             ha="center", fontsize=6, color="#666666")
    fig.tight_layout(w_pad=1.2)
    C.save(fig, "fig_fidelity")
    plt.close(fig)


# -------------------------------------------------------------------- energy
def fig_energy():
    """Energy per frame, from the settled part of each telemetry segment.

    See SETTLE_S in derive.py: averaging a segment in full biases it, and both
    biases push the dynamic figure the same way.
    """
    op = "irbeam_64_unet_cch_8pct_8_16"
    rows = []
    for board in ("ultra96v2", "jetson"):
        for r in (_read(f"energy_{board}") or []):
            if r["experiment"] == op:
                r["device"] = "u96v2" if board == "ultra96v2" else "orin_nx"
                rows.append(r)
    if not rows:
        return print("  skip energy: no corrected energy table")
    if not rows:
        return print("  skip energy: operating point not covered")

    def key(r):
        return _label(r["route"], r["device"], r["precision"])

    agg = {}
    for r in rows:
        d = agg.setdefault(key(r), dict(t=[], dy=[], fj=[], tc=[]))
        for fld, c in (("e_total_mj", "t"), ("e_dynamic_mj", "dy"),
                       ("frames_per_joule", "fj"), ("e_total_compute_mj", "tc")):
            if r.get(fld) not in (None, ""):
                d[c].append(float(r[fld]))
    labels = sorted(agg, key=lambda k: C.median(agg[k]["t"]) or 0)
    if not labels:
        return print("  skip energy: nothing to plot")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(C.COL_W * 1.75, C.COL_H * 0.95))
    xs = range(len(labels))
    w = 0.27
    a1.bar([x - w for x in xs], [C.median(agg[l]["t"]) for l in labels], w,
           label=r"$E_\mathrm{tot}$ (input plane)", color="#1F6FB4")
    a1.bar(list(xs), [C.median(agg[l]["tc"]) for l in labels], w,
           label=r"$E_\mathrm{tot}$ (compute plane)", color="#7FA9D0")
    a1.bar([x + w for x in xs], [C.median(agg[l]["dy"]) for l in labels], w,
           label=r"$E_\mathrm{dyn}$", color="#C4451C", hatch="///",
           edgecolor="white")
    a1.set_ylabel("Energy per frame (mJ)")
    a1.set_yscale("log")
    a1.legend(frameon=False, fontsize=6.2)
    a2.bar(list(xs), [C.median(agg[l]["fj"]) for l in labels], 0.6,
           color="#2E9B57")
    a2.set_ylabel("Efficiency (frames/J)")
    a2.set_yscale("log")
    for ax in (a1, a2):
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6.5)
    fig.text(0.5, -0.10,
             "Median of three repetitions, telemetry averaged over the settled "
             "part of each segment. Energy at pinned clocks.",
             ha="center", fontsize=6.5, color="#666666")
    C.save(fig, "fig_energy")
    plt.close(fig)


# ------------------------------------------------------------ frame budget
def fig_budget():
    """Per-frame budget across sampling ratios, both platforms.

    Same construction as the corresponding figure of the previous study --
    stacked acquisition, reconstruction and inference, paired baseline against
    accelerator at each ratio -- extended to two boards. Acquisition uses
    projections rather than patterns, so the bipolar family carries twice the
    cost at equal M.
    """
    sens = _sensing_index()
    rows = C.load_all()
    exps = sorted((e for e in sens
                   if sens[e]["image_class"] == "irbeam" and sens[e]["family"] == "cch"),
                  key=lambda e: sens[e]["sampling_ratio_pct"])
    if not exps:
        return print("  skip budget: no sensing table")

    # reconstruction time per experiment, per board
    rec = {}
    for r in rows:
        if r["mode"] == "recon" and r.get("t_rec_ms") is not None:
            rec.setdefault((r["board"], r["experiment"]), []).append(r["t_rec_ms"])
    # inference at the operating point defines the per-board pair; the network
    # is identical across the grid so latency does not depend on the ratio
    def inf(board, route, prec):
        xs = [r["lat_median_ms"] for r in rows
              if r["board"] == board and C.is_timing_pass(r)
              and r["route"] == route and (r["precision"] or "") == prec]
        return C.median(xs)

    pairs = [("ultra96v2", ("cpu_neon", "fp32"), ("dpu_int8", ""), "Ultra96-V2", "#8A4FBF"),
             ("jetson", ("cpu_neon", "fp32"), ("gpu_trt", "fp16"), "Orin NX", "#C4451C")]
    c_acq, c_rec, c_cpu = "#A8C8A0", "#F0B27A", "#6B6B6B"

    fig, axes = plt.subplots(1, 2, figsize=(C.COL_W, C.COL_H * 0.85), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, (board, base, accel, name, cacc) in zip(axes, pairs):
        t_base, t_acc = inf(board, *base), inf(board, *accel)
        if t_base is None or t_acc is None:
            ax.set_visible(False)
            continue
        x = np.arange(len(exps), dtype=float)
        w = 0.38
        acq = [sens[e]["t_acq_ms"] for e in exps]
        rc = [C.median(rec.get((board, e), [])) or 0.0 for e in exps]
        for off, t_inf, col in ((-w / 2, t_base, c_cpu), (+w / 2, t_acc, cacc)):
            ax.bar(x + off, acq, w, color=c_acq)
            ax.bar(x + off, rc, w, bottom=acq, color=c_rec)
            ax.bar(x + off, [t_inf] * len(exps), w,
                   bottom=[a + b for a, b in zip(acq, rc)], color=col)
        for i, e in enumerate(exps):
            tot_b = acq[i] + rc[i] + t_base
            tot_a = acq[i] + rc[i] + t_acc
            ax.annotate(f"{tot_b / tot_a:.1f}x", (x[i], max(tot_b, tot_a)),
                        textcoords="offset points", xytext=(0, 2.5),
                        ha="center", fontsize=6.5, color=cacc, weight="bold")
        ax.set_xlim(-0.55, len(exps) - 0.45)
        ax.set_ylim(0, 158)
        ax.set_xticks(x)
        ax.set_xticklabels(["1", "2", "4", "8", "16"], fontsize=7)
        ax.set_xlabel(r"Sampling ratio $M/N$ (%)")
        ax.set_title(name, pad=4)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Per-frame time (ms)")
    from matplotlib.patches import Patch
    handles = [Patch(color=c_acq, label="acquisition"), Patch(color=c_rec, label="reconstruction"),
               Patch(color=c_cpu, label="inference, processor"),
               Patch(color="#8A4FBF", label="inference, DPU"),
               Patch(color="#C4451C", label="inference, GPU FP16")]
    fig.legend(handles=handles, frameon=False, fontsize=6.2, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.16), handlelength=1.2, columnspacing=1.2)
    fig.text(0.5, -0.24, "Annotations: end-to-end speed-up of the accelerated path over its own processor path.",
             ha="center", fontsize=6.2, color="#666666")
    C.save(fig, "fig_pipeline")
    plt.close(fig)


# ------------------------------------------------------------- thread sweep
def fig_threads():
    rows = [r for r in C.load()
            if r["route"] == "cpu_neon" and C.is_timing_pass(r) and r["threads"]]
    if not rows:
        return print("  skip threads: no CPU thread sweep")
    op = "irbeam_64_unet_cch_8pct_8_16"
    agg = C.aggregate([r for r in rows if r["experiment"] == op],
                      ["threads"], "fps_from_latency")
    if not agg:
        return print("  skip threads: operating point not covered")
    ref = C.aggregate([r for r in C.load()
                       if r["route"] == "gpu_trt" and C.is_timing_pass(r)
                       and r["precision"] == "fp16" and r["experiment"] == op],
                      ["precision"], "fps_from_latency")
    ts = sorted(agg, key=lambda k: k[0])
    fig, ax = plt.subplots(figsize=(C.COL_W, C.COL_H * 0.8))
    ax.plot([t[0] for t in ts], [agg[t][0] for t in ts], marker="o", ms=4,
            color="#4C4C4C", label="Orin CPU (FP32)")
    if ref:
        v = list(ref.values())[0][0]
        ax.axhline(v, ls="--", lw=1.0, color="#2E9B57",
                   label="Orin GPU (FP16)")
    ax.set_xlabel("Intra-operator threads")
    ax.set_ylabel("Throughput (fps)")
    ax.set_yscale("log")
    ax.set_xticks([t[0] for t in ts])
    ax.legend(frameon=False)
    C.save(fig, "fig_thread_scaling")
    plt.close(fig)


# -------------------------------------------------------------- qualitative
def _strip(image_class, stem, title, frame=0):
    """A ladder: one row per sampling ratio, columns ground truth to INT8.

    The figure plan asks for one row per ratio, which is also what
    dodges a gap in the campaign: predicted frames were saved at every
    cake-cutting ratio except 7.98 %, the operating point.
    """
    import numpy as np
    sens = _sensing_index()
    exps = sorted((e for e, s_ in sens.items()
                   if s_["image_class"] == image_class and s_["family"] == "cch"
                   and (C.DATA / f"strip_{e}.npz").is_file()),
                  key=lambda e: sens[e]["sampling_ratio_pct"])
    if not exps:
        return print(f"  skip {stem}: no strips for {image_class}")

    first = np.load(C.DATA / f"strip_{exps[0]}.npz")
    # Withdrawn engines (common.WITHDRAWN_PRECISIONS) are kept in the strips
    # on disk but never drawn: the column would name a different precision
    # mix at every ratio.
    gpu_keys = [k for k in first.files if k.startswith("gpu_trt")
                and k.split("_", 2)[-1] not in C.WITHDRAWN_PRECISIONS]
    cols = [("originals", "Ground truth"), ("reconstructions", r"Linear $\tilde{x}$"),
            ("fp32", "U-Net FP32")]
    cols += [(k, "GPU " + C.PRECISION_LABEL.get(k.split("_", 2)[-1], k)) for k in gpu_keys]

    fig, axes = plt.subplots(len(exps), len(cols),
                             figsize=(0.95 * len(cols), 0.95 * len(exps)))
    axes = np.asarray(axes).reshape(len(exps), len(cols))
    for i, e in enumerate(exps):
        d = np.load(C.DATA / f"strip_{e}.npz")
        for j, (key, name) in enumerate(cols):
            ax = axes[i, j]
            if key in d.files:
                img = np.squeeze(d[key][frame])   # some routes save (1, 64, 64)
                ax.imshow(img, cmap=IMAGE_CMAP, vmin=0, vmax=1,
                          interpolation="nearest")
            else:
                ax.set_facecolor("#EEEEEE")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if i == 0:
                ax.set_title(name, fontsize=6.5, pad=3)
            if j == 0:
                ax.set_ylabel(f"{sens[e]['sampling_ratio_pct']:.2f}%",
                              fontsize=6.5, rotation=0, ha="right", va="center",
                              labelpad=12)
    fig.suptitle(title, fontsize=7.5, y=1.005)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    C.save(fig, stem)
    plt.close(fig)


def fig_qualitative():
    _strip("irbeam", "fig_qualitative",
           "Beam profiles, cake-cutting")


def fig_qualitative_struct():
    _strip("celebA", "fig_qualitative_struct",
           "Structured class, cake-cutting")



def fig_dataset_samples():
    """§3.1: the two image classes along the primary axis.

    One evaluation frame per class (the first of the strips, identical across
    configurations): ground truth, then its linear reconstruction from
    cake-cutting patterns at the five sampling ratios of the sweep. One
    variable only; the pattern families appear in the overview figure.
    """
    import numpy as np
    sens = _sensing_index()
    rows = [("irbeam", "Beam profiles"), ("celebA", "Structured")]
    exps = {cls: sorted((e for e, s_ in sens.items()
                         if s_["image_class"] == cls and s_["family"] == "cch"),
                        key=lambda e: sens[e]["sampling_ratio_pct"])
            for cls, _ in rows}
    ncol = 1 + len(exps["irbeam"])
    # Drop an image class whose strips are absent -- CelebA, when working from
    # aspir-edge-data, which cannot redistribute its frames.
    rows = [(cls, lbl) for cls, lbl in rows
            if exps[cls] and all((C.DATA / f"strip_{e}.npz").is_file() for e in exps[cls])]
    if not rows:
        return print("  skip fig_dataset_samples: no strips")
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4.6, 1.75))
    axes = np.asarray(axes).reshape(len(rows), ncol)
    for i, (cls, rlabel) in enumerate(rows):
        for j in range(ncol):
            ax = axes[i, j]
            if j == 0:
                d = np.load(C.DATA / f"strip_{exps[cls][0]}.npz")
                img, title = d["originals"][0], "Ground truth"
            else:
                e = exps[cls][j - 1]
                d = np.load(C.DATA / f"strip_{e}.npz")
                img = d["reconstructions"][0]
                title = f"{sens[e]['sampling_ratio_pct']:.2f}%"
            ax.imshow(np.squeeze(img), cmap=IMAGE_CMAP, vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if i == 0:
                ax.set_title(title, fontsize=6.5, pad=3)
            if j == 0:
                ax.set_ylabel(rlabel, fontsize=6.5)
    fig.suptitle(r"Linear reconstruction $\tilde{x}$, cake-cutting, by sampling ratio $M/N$",
                 fontsize=6.5, y=1.02)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    C.save(fig, "fig_dataset_samples")
    plt.close(fig)

def fig_overview_thumbs():
    """Thumbnails embedded in the TikZ overview (figures/fig_overview.tex).

    Real frames, nearest-neighbour upscaled, written to figures/thumbs/.
    Beam frames at the operating point; face ground truth; one pattern per
    family from figures/data/patterns.npz.
    """
    import numpy as np
    from matplotlib.image import imsave
    out = C.FIGS / "thumbs"; out.mkdir(parents=True, exist_ok=True)
    def put(name, img, vmin=0, vmax=1, cmap=IMAGE_CMAP):
        img = np.kron(np.squeeze(img), np.ones((4, 4)))
        imsave(out / f"{name}.png", img, cmap=cmap, vmin=vmin, vmax=vmax)
    b = np.load(C.DATA / "strip_irbeam_64_unet_cch_8pct_8_16.npz")
    put("beam_gt", b["originals"][0])
    face = C.DATA / "strip_celebA_64_unet_cch_8pct_8_16.npz"
    if face.is_file():
        put("face_gt", np.load(face)["originals"][0])
    else:
        # CelebA frames are not redistributable, so aspir-edge-data does not
        # ship this strip; the face thumbnail needs a local CelebA copy.
        print("  skip face_gt: no CelebA strip")
    put("beam_rec", b["reconstructions"][0]); put("beam_out", b["fp32"][0])
    pat = np.load(C.DATA / "patterns.npz")
    put("pat_cch", pat["cch"], vmin=-1, vmax=1, cmap="gray")   # masks stay binary
    put("pat_scatter", pat["scatter"], cmap="gray"); put("pat_sweep", pat["sweep"], cmap="gray")
    print("  wrote figures/thumbs/*.png")


# --------------------------------------------------------------- error floor
def _floor_points():
    """(P32, F, P, board, class, experiment) for every accelerator grid point.

    One entry per (board, experiment): the frame-quality CSVs carry repeated
    rows where a configuration was re-run, and each is collapsed to its median
    so that a point is a grid point, not a run. P32 is the single-precision
    output of the same network against ground truth (from the dataset; see
    derive_fp32_reference), F the engine's fidelity against that output, P the
    deployed path against ground truth.
    """
    sens = _sensing_index()
    ref = {r["experiment"]: float(r["psnr_fp32_gt_db"])
           for r in (_read("fp32_reference") or [])}
    pts = []
    for board, accel in (("jetson", ("gpu_trt", "int8_strict")),
                         ("ultra96v2", ("dpu_int8", ""))):
        q = _read(f"frame_quality_{board}") or []
        by_exp = {}
        for r in q:
            if (r["route"], r.get("precision") or "") != accel:
                continue
            if r["experiment"] not in ref or r.get("psnr_fp32_mean") in (None, ""):
                continue
            if r["experiment"] in C.LEGACY_EXPERIMENTS:   # 2025 dataset, not on the grid
                continue
            by_exp.setdefault(r["experiment"], []).append(r)
        for e, rows in by_exp.items():
            fid = C.median([float(r["psnr_fp32_mean"]) for r in rows])
            meas = C.median([float(r["psnr_gt_mean"]) for r in rows])
            pts.append((ref[e], fid, meas, board,
                        sens.get(e, {}).get("image_class", ""), e))
    return pts


def fig_error_floor():
    """Quantisation as an error floor, in one panel with one message.

    If the quantisation error is independent of the task error the two combine
    as noise powers, so the cost of quantisation P32 - P depends only on the
    distance P32 - F between the single-precision model and the engine's
    fidelity:  cost = 10 log10(1 + 10^((P32 - F)/10)).  No fitted parameter.
    Every grid point of both accelerators and both image classes is plotted
    against that one curve. One marker shape; the boards by colour; the image
    classes as the regions they occupy, not as marker shapes.
    """
    import math
    pts = _floor_points()
    if not pts:
        return print("  skip error_floor: no paired points")
    style = {"ultra96v2": _style("ZU3EG DPU (INT8)") | {"label": "ZU3EG DPU (INT8)"},
             "jetson": _style("Orin GPU (INT8)") | {"label": "Orin GPU (INT8)"}}
    fig, ax = plt.subplots(figsize=(C.COL_W, C.COL_H * 0.78))
    # the regions the two image classes occupy
    xs_c = [p[0] - p[1] for p in pts if p[4] == "celebA"]
    xs_b = [p[0] - p[1] for p in pts if p[4] == "irbeam"]
    for xs_, lab in ((xs_c, "structured class"), (xs_b, "beam class")):
        if xs_:
            ax.axvspan(min(xs_) - 0.6, max(xs_) + 0.6, color="#000000", alpha=0.05, lw=0)
            ax.annotate(lab, ((min(xs_) + max(xs_)) / 2, 8.35), ha="center",
                        fontsize=6.5, color="#666666")
    # the curve
    xs = [x / 10 for x in range(-220, 90)]
    ax.plot(xs, [10 * math.log10(1 + 10 ** (x / 10)) for x in xs],
            color="#555555", lw=1.1, zorder=2, label="Eq. (2)")
    # the points
    seen = set()
    for p32, fid, meas, board, cls, e in pts:
        st = style[board]
        ax.scatter(p32 - fid, p32 - meas, s=30, color=st["color"], marker="o",
                   edgecolors="white", linewidths=0.5, zorder=3,
                   label=st["label"] if board not in seen else None)
        seen.add(board)
    for board in style:
        sub = [p for p in pts if p[3] == board]
        if sub:
            mae = sum(abs((p[0] - p[2]) - 10 * math.log10(1 + 10 ** ((p[0] - p[1]) / 10)))
                      for p in sub) / len(sub)
            print(f"    {board}: MAE {mae:.3f} dB over {len(sub)} points")
    print(f"    total points: {len(pts)}")
    # one guide: 3 dB where model and floor coincide
    ax.plot([0, 0], [-0.3, 3], color="#999999", lw=0.6, ls=":")
    ax.annotate("3 dB at the floor", (-0.6, 3.15), fontsize=6.3, color="#444444", ha="right")
    ax.set_xlabel("FP32 quality minus engine fidelity (dB)")
    ax.set_ylabel("Quality lost to\nquantisation (dB)")
    ax.set_xlim(-20.5, 8.8)
    ax.set_ylim(-0.3, 8.9)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    C.save(fig, "fig_error_floor")
    plt.close(fig)



# -------------------------------------------------------------- power trace
def fig_power_trace():
    """Board and compute power through idle -> load -> idle.

    Shows that the energy figures come from a measurement rather than a
    specification: the idle floor, the level from which energy per frame is
    derived, and the settling at each transition are all visible.
    """
    rows = _read(f"power_trace_{BOARD}")
    if not rows:
        return print("  skip power_trace: no telemetry series")
    op = "irbeam_64_unet_cch_8pct_8_16"
    want = [("gpu_trt", "fp16"), ("cpu_neon", "fp32")]
    fig, axes = plt.subplots(len(want), 1, figsize=(C.COL_W * 1.3, C.COL_H * 1.25),
                             sharex=True)
    fig.subplots_adjust(hspace=0.35)
    drawn = 0
    for ax, (route, prec) in zip(np.atleast_1d(axes), want):
        sub = [r for r in rows if r["experiment"] == op and r["route"] == route
               and (r["precision"] or "fp32") == prec]
        if not sub:
            ax.set_visible(False)
            continue
        drawn += 1
        for seg, alpha in (("idle", 0.55), ("active", 1.0)):
            ss = sorted((r for r in sub if r["segment"] == seg),
                        key=lambda r: r["t_s"])
            if not ss:
                continue
            t = [r["t_s"] for r in ss]
            ax.plot(t, [r["VDD_IN"] for r in ss], color="#1F6FB4", lw=0.9,
                    alpha=alpha, label="module input" if seg == "active" else None)
            ax.plot(t, [r["VDD_CPU_GPU_CV"] for r in ss], color="#C4451C", lw=0.9,
                    alpha=alpha, label="compute rail" if seg == "active" else None)
            if seg == "idle":
                ax.axvspan(min(t), max(t), color="#000000", alpha=0.04)
                ax.annotate("idle", (0.5 * (min(t) + max(t)), ax.get_ylim()[1]),
                            ha="center", va="top", fontsize=6, color="#777777")
        ax.set_ylabel("Power (W)")
        ax.set_title(_label(route, "orin_nx", prec), fontsize=7.5, pad=4)
        ax.legend(frameon=False, fontsize=6.2, loc="center right")
    if not drawn:
        plt.close(fig)
        return print("  skip power_trace: operating point not covered")
    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    C.save(fig, "fig_power_trace")
    plt.close(fig)


FIGURES = {
    "precision": fig_precision,
    "latency": fig_latency,
    "quality": fig_quality,
    "fidelity": fig_fidelity,
    "error_floor": fig_error_floor,
    "power_trace": fig_power_trace,
    "energy": fig_energy,
    "budget": fig_budget,
    "threads": fig_threads,
    "qualitative": fig_qualitative,
    "qualitative_struct": fig_qualitative_struct,
    "dataset_samples": fig_dataset_samples,
    "overview_thumbs": fig_overview_thumbs,
}


# The figures the manuscript includes (sections/*.tex, figures/*.tex). The
# rest of FIGURES are kept for the released artefact and for exploration;
# they are git-ignored so that `--all` does not dirty the tree.
PAPER = ("dataset_samples", "overview_thumbs", "quality", "qualitative",
         "error_floor", "latency", "budget")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES))
    ap.add_argument("--all", action="store_true",
                    help="every figure, not only those the manuscript uses")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        return print("\n".join(f"{n}{'  (paper)' if n in PAPER else ''}"
                               for n in sorted(FIGURES)))
    names = a.only or (sorted(FIGURES) if a.all else list(PAPER))
    for name in names:
        print(f"{name}:")
        FIGURES[name]()


if __name__ == "__main__":
    main()
