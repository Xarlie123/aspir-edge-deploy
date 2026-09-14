"""Shared conventions for every figure and table in the paper.

The conventions listed in figures/README.md are implemented here once, so
that a plotting script cannot quietly break one. Read that file before
changing anything in this module.

Chain:  data/  ->  figures/data/*.csv  ->  figures/*.pdf

Only `derive.py` may read data/. Every plotting script reads figures/data/
alone, so figures regenerate without the raw campaign (~159 MB, git-ignored,
and partly not redistributable).
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The raw campaign is the aspir-edge-data repository: point ASPIR_EDGE_DATA at
# a checkout of it. Derived CSVs are written to, and plotted from,
# ASPIR_FIGURE_DATA -- set it to <aspir-edge-data>/derived to plot the
# published tables without re-deriving them.
RAW = Path(os.environ.get("ASPIR_EDGE_DATA", ROOT / "data"))
DATA = Path(os.environ.get("ASPIR_FIGURE_DATA", ROOT / "figures" / "data"))
FIGS = ROOT / "figures" / "output"

DMD_KHZ = 10.752

# ---------------------------------------------------------------- route names
# The paper's notation fixes these. Legends use them verbatim, everywhere, including the
# precision: a bare "GPU" or "DPU" would let a cross-device comparison silently
# report the cost of a data type as a property of the fabric.
# Base names carry NO precision: route_label() appends it from the row. An
# earlier version baked "(FP32)" into the CPU entries and short-circuited on a
# trailing parenthesis, so every CPU INT8 row was labelled FP32.
ROUTE_LABEL = {
    ("cpu_neon", "orin_nx"): "Orin CPU",
    ("gpu_trt", "orin_nx"): "Orin GPU",
    ("cpu_recon", "orin_nx"): "Orin CPU recon.",
    ("cpu_recon", "u96v2"): "ZU3EG CPU recon.",
    ("cpu_neon", "u96v2"): "ZU3EG CPU",
    ("dpu_int8", "u96v2"): "ZU3EG DPU",
}
DEFAULT_PRECISION = {"cpu_neon": "fp32", "dpu_int8": "int8"}
PRECISION_LABEL = {
    "int8_strict": "INT8",
    "int8_nofp16": "INT8 (FP16 off)",
    "int8": "INT8",
    "fp16": "FP16",
    "fp32": "FP32",
}
# `int8` means different things by route. On the processor it is a genuine
# eight-bit path through the runtime. On the graphics accelerator it is the
# engine built with tactic selection left free, which for this network contains
# no eight-bit layers at all -- so it is named for what was requested, not for
# what it runs, and the distinction has to survive into every legend.
PRECISION_LABEL_BY_ROUTE = {
    ("gpu_trt", "int8"): "INT8 req.",
}

# Distinguished by marker and dash pattern as well as hue: the journal prints
# in black and white and colour alone would not survive.
STYLE = {
    "Orin CPU (FP32)":  dict(color="#4C4C4C", marker="o", ls="-",   ms=4),
    "ZU3EG CPU (INT8)": dict(color="#9A6BC4", marker="p", ls=":",   ms=4),
    "Orin GPU (FP32)":  dict(color="#1F6FB4", marker="s", ls="--",  ms=4),
    "Orin GPU (FP16)":  dict(color="#2E9B57", marker="^", ls="-.",  ms=4),
    "Orin GPU (INT8)":  dict(color="#C4451C", marker="D", ls=":",   ms=4),
    "Orin GPU (INT8 req.)":  dict(color="#E08A2E", marker="d", ls=(0,(3,1,1,1)), ms=4),
    "Orin GPU (INT8 (FP16 off))": dict(color="#D4713F", marker="d", ls=(0,(4,1)), ms=4),
    "Orin CPU (INT8)":  dict(color="#6B6B6B", marker="o", ls="--",  ms=4),
    "ZU3EG CPU (FP32)": dict(color="#7A7A7A", marker="v", ls="-",   ms=4),
    "Orin GPU (INT8 mixed)": dict(color="#E08A2E", marker="d", ls=(0,(3,1,1,1)), ms=4),
    "Orin CPU recon.":  dict(color="#8C8C8C", marker="*", ls="-",   ms=5),
    "ZU3EG CPU recon.": dict(color="#A0A0A0", marker="*", ls="--",  ms=5),
    "ZU3EG CPU (FP32)": dict(color="#7A7A7A", marker="v", ls="-",   ms=4),
    "ZU3EG DPU (INT8)": dict(color="#8A4FBF", marker="P", ls="--",  ms=4),
    "Linear recon.":    dict(color="#B0B0B0", marker="x", ls="-",   ms=4),
}

CLASS_LABEL = {"irbeam": "Beam profiles", "celebA": "Structured (faces)"}


def route_label(row) -> str:
    """Full route name including precision. Rule 5: ratios name both sides."""
    base = ROUTE_LABEL.get((row["route"], row["device"]))
    if base is None:
        return f"{row['route']}/{row['device']}"
    prec = (row.get("precision") or "").strip()
    if not prec:
        prec = DEFAULT_PRECISION.get(row.get("route"), "")
    if not prec or base.endswith("recon."):
        return base
    lab = PRECISION_LABEL_BY_ROUTE.get((row.get("route"), prec)) \
        or PRECISION_LABEL.get(prec, prec.upper())
    return f"{base} ({lab})"


def image_class(experiment: str) -> str:
    return experiment.split("_", 1)[0]


def family(experiment: str) -> str:
    for f in ("cch", "sweep", "scatter"):
        if f"_{f}_" in experiment or f"_{f}" in experiment:
            return f
    return "?"


# ------------------------------------------------------------------ reading
def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return v


def clocks_fixed(row) -> bool:
    """True when the clock state is fixed, by either board's convention."""
    if str(row.get("gpu_pinned")) == "True" or str(row.get("cpu_pinned")) == "True":
        return True
    dvfs = str(row.get("cpu_dvfs", "")).strip().lower()
    return dvfs == "false"          # no DVFS at all: fixed by construction


LEGACY_EXPERIMENTS = {"beam_64_gaussian_unet_8pct_cch",
                      # Rows whose "experiment" is the board name come from
                      # combined_*.json and phase_*.json at the top of
                      # results/ultra96v2/: summary files from the May and July
                      # exploratory phases, not run reports. They carry no
                      # clock state and no experiment identity.
                      "ultra96v2", "jetson", "orin_nx", "u96v2"}
"""Runs from the 2025 dataset, kept on disk and excluded from every figure.

Different dataset, different network, different sensing grid. They predate the
fourteen-point campaign and share no axis with it.
"""

BOARD_CSV = {"jetson": "measurements_jetson.csv",
             "ultra96v2": "measurements_ultra96v2.csv"}


def load_all(**kw) -> list[dict]:
    """Every board, in one list. Rows carry `board`."""
    out = []
    for board, name in BOARD_CSV.items():
        if not (DATA / name).is_file():
            continue
        for r in load(name, **kw):
            r["board"] = board
            out.append(r)
    return out


def load(name: str = "measurements_jetson.csv", *, pinned_only: bool = True,
         on_board_only: bool = True, include_withdrawn: bool = False) -> list[dict]:
    """Load an aggregate CSV, applying the filters that make rows comparable.

    Rule 2 -- clocks must be pinned. A run measured with the governor free is
    not comparable with one that was not: the same engine moved between 0.31
    and 0.73 ms across states, and its tail measured the governor rather than
    the workload.

    `on_board_only` additionally drops rows measured on the workstation. Those
    exist for the reconstruction route and are a legitimate reference point,
    but they are not part of any on-board frame budget.
    """
    rows = []
    for r in csv.DictReader(open(DATA / name)):
        r = {k: _num(v) for k, v in r.items()}
        if pinned_only and not clocks_fixed(r):
            continue
        if on_board_only and r.get("device") == "host":
            continue
        if not include_withdrawn and (r.get("precision") or "") in WITHDRAWN_PRECISIONS:
            continue
        if r.get("experiment") in LEGACY_EXPERIMENTS:
            continue
        r["label"] = route_label(r)
        r["class"] = image_class(r["experiment"])
        r["family"] = family(r["experiment"])
        rows.append(r)
    return rows


WITHDRAWN_PRECISIONS = {"int8_nofp16"}
"""Engine labels withdrawn from the record.

`int8_nofp16` asked TensorRT to avoid half precision but left tactic selection
free, and the selector assigned a different precision mix to every build: the
observed INT8 fraction ranges from 0.26 to 0.96 across three deployments of
what was nominally one configuration. "INT8" therefore named a different thing
at each grid point, and no quantisation claim can rest on it. The runs are kept
on disk and excluded here; `int8_strict`, rebuilt with the precision of every
layer pinned, replaces them.

`int8` -- the free-selector engine -- is NOT withdrawn. It contains zero INT8
layers, and that is a reported result about the tooling rather than a defect:
asking for eight bits and receiving half precision is worth showing.
"""

TIMING_PASS_MIN_FRAMES = 1000
"""Below this, a run is not a timing pass.

Some runs carry `mode: latency` but measured only the ~100-frame quality loop.
Their `lat_*` fields are the incidental timing of that loop -- the thing
figures/README.md warns against using -- and a p99 over 80 frames is the maximum,
not a quantile. `is_timing_pass` is the gate; every latency figure applies it.
"""


def is_timing_pass(row) -> bool:
    n = row.get("n_measured")
    return row.get("mode") == "latency" and n is not None and n >= TIMING_PASS_MIN_FRAMES


# --------------------------------------------------------------- statistics
def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def spread(xs):
    """Min-max range. Rule 3: a single run is not a result; report dispersion."""
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else (None, None)


def aggregate(rows, keys, field):
    """Median across repetitions, with the range, grouped by `keys`.

    Rule 3 for timing and energy. Quality has one repetition on purpose
    (rule 4: deterministic given fixed weights), and grouping it this way is
    harmless -- the median of one value is that value.
    """
    out = {}
    for r in rows:
        if r.get(field) is None:
            continue
        out.setdefault(tuple(r[k] for k in keys), []).append(r[field])
    return {k: (median(v), *spread(v), len(v)) for k, v in out.items()}


def bootstrap_ci(values, n=2000, alpha=0.05, seed=12345):
    """Percentile bootstrap over frames.

    With a 100-frame evaluation split the interesting statement is an interval
    over frames, not a bare mean (figures/README.md). Deterministic
    seed so a figure regenerates identically.
    """
    import random
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return (None, None)
    rng = random.Random(seed)
    n_v = len(vals)
    means = []
    for _ in range(n):
        means.append(sum(vals[rng.randrange(n_v)] for _ in range(n_v)) / n_v)
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n) - 1]
    return (lo, hi)


# ------------------------------------------------------------------ plotting
def setup_mpl():
    """Figure style for the svjour3 smallextended single-column layout.

    Fonts are set so the effective size in the final PDF lands at 8-9 pt
    without \\includegraphics rescaling, per the paper's figure plan.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    return plt


COL_W = 4.8   # inches, one column of the smallextended layout
COL_H = 3.0


def save(fig, stem: str):
    """Write the figure, deterministically.

    Matplotlib stamps a creation date into the PDF, so regenerating an
    unchanged figure still produces a binary diff and the repository fills with
    churn. Setting the metadata explicitly makes the output a function of the
    data alone.
    """
    FIGS.mkdir(exist_ok=True)
    meta = {"CreationDate": None}
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}.{ext}",
                    metadata=meta if ext == "pdf" else None)
    print(f"  wrote figures/{stem}.pdf (+ .png)")


def write_csv(stem: str, fieldnames, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"{stem}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote figures/data/{stem}.csv  ({len(rows)} rows)")
