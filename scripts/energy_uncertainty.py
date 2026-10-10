#!/usr/bin/env python3
"""
Dynamic energy per frame with its uncertainty, recomputed from archived runs.

Every energy-mode report written by the runners has a `_power.npz` sidecar with
the raw idle and load telemetry. This script applies the uncertainty
convention of ASPIR (`aspir_unet.measurement.uncertainty`) to those series, so
runs measured before the convention existed get the same fields as new ones:

    python3 scripts/energy_uncertainty.py <results-dir> [<results-dir> ...] [--out table.csv]

One row per run: idle mean +/- sigma (1 s means of the settled idle segment)
and its standard error, total and dynamic energy per frame, the two
uncertainty components of ASPIR, their combination and the 95 % bootstrap
interval, plus the sensor-resolution term (decisive on the Ultra96 board
input, which reads in 1.25 W steps).
Runs written before frame times were recorded use the sustained frame rate
for every block (column `blocks_from`).

New runs carry the same numbers in their report, under
`energy_mode.uncertainty`; for those the script reads the series again and
should agree with the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aspir_unet.measurement.energy import _is_board_input_label  # noqa: E402
from aspir_unet.measurement.uncertainty import energy_uncertainty  # noqa: E402

COLUMNS = [
    "file", "route", "precision", "tag", "experiment", "repetition", "board_input_rails",
    "fps_sustained", "blocks_from", "n_blocks",
    "p_idle_w", "sigma_idle_w", "sem_idle_w", "n_idle_windows", "raw_sigma_idle_w",
    "p_active_block_w", "e_total_mj", "e_dynamic_mj",
    "sigma_idle_mj", "sigma_idle_sem_mj", "sigma_blocks_mj", "sigma_mj",
    "ci95_low_mj", "ci95_high_mj", "idle_share",
    "sensor_resolution_w", "sigma_resolution_mj", "sigma_with_resolution_mj",
]


def _board_rails(keys, segment):
    """Board-input rails present in a sidecar segment, by the runners' own rule."""
    prefix = f"{segment}_"
    labels = [k[len(prefix):] for k in keys if k.startswith(prefix) and k != f"{segment}_t_s"]
    return [lbl for lbl in labels if "." in lbl and _is_board_input_label(lbl)]


def run_row(report_path: Path) -> dict | None:
    rep = json.loads(report_path.read_text())
    em = rep.get("energy_mode") or {}
    side = report_path.with_name(report_path.stem + "_power.npz")
    if not em or not side.is_file():
        return None
    d = np.load(side)
    if "idle_t_s" not in d.files or "active_t_s" not in d.files:
        return None
    rails = _board_rails(d.files, "idle")
    if not rails:
        return None
    idle_p = np.sum([d[f"idle_{r}"] for r in rails], axis=0)
    active_p = np.sum([d[f"active_{r}"] for r in rails], axis=0)
    frames = d["active_frame_t_s"] if "active_frame_t_s" in d.files else None
    fps = (em.get("active") or {}).get("fps_sustained")
    u = energy_uncertainty(d["idle_t_s"], idle_p, d["active_t_s"], active_p,
                           frame_t_s=frames, fps=fps)
    if "dynamic_energy_pt_mj" not in u:
        return None
    run = rep.get("run", {})
    exp = run.get("experiment")
    lo, hi = u["dynamic_energy_ci95_mj"]
    return {
        "file": report_path.name,
        "route": run.get("route", ""),
        "precision": (run.get("params") or {}).get("precision", ""),
        "tag": run.get("tag", ""),
        "experiment": exp.get("experiment_id", "") if isinstance(exp, dict) else (exp or ""),
        "repetition": run.get("repetition", ""),
        "board_input_rails": "+".join(rails),
        "fps_sustained": fps,
        "blocks_from": u["blocks"]["source"],
        "n_blocks": u["n_blocks"],
        "p_idle_w": round(u["idle_power_W"], 4),
        "sigma_idle_w": round(u["idle_power_std_W"], 4),
        "sem_idle_w": round(u["idle_power_sem_W"], 4),
        "n_idle_windows": u["n_idle_samples"],
        "raw_sigma_idle_w": round(u["idle"]["raw_std_W"], 4),
        "p_active_block_w": round(float(np.mean(u["blocks"]["power_W"])), 4),
        "e_total_mj": round(u["energy_block_mean_mj"], 5),
        "e_dynamic_mj": round(u["dynamic_energy_pt_mj"], 5),
        "sigma_idle_mj": round(u["dynamic_energy_sigma_idle_mj"], 5),
        "sigma_idle_sem_mj": round(u["dynamic_energy_sigma_idle_sem_mj"], 5),
        "sigma_blocks_mj": round(u["dynamic_energy_sigma_blocks_mj"], 5),
        "sigma_mj": round(u["dynamic_energy_sigma_mj"], 5),
        "ci95_low_mj": round(lo, 5),
        "ci95_high_mj": round(hi, 5),
        "idle_share": round(u["idle_share_of_energy"], 4) if u["idle_share_of_energy"] else "",
        "sensor_resolution_w": u.get("sensor_resolution_W") or "",
        "sigma_resolution_mj": round(u["dynamic_energy_sigma_resolution_mj"], 5)
        if "dynamic_energy_sigma_resolution_mj" in u else "",
        "sigma_with_resolution_mj": round(u["dynamic_energy_sigma_with_resolution_mj"], 5)
        if "dynamic_energy_sigma_with_resolution_mj" in u else "",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("roots", nargs="+", type=Path, help="results directories (searched recursively)")
    ap.add_argument("--out", type=Path, help="write CSV here instead of stdout")
    args = ap.parse_args(argv)
    rows = []
    for root in args.roots:
        for js in sorted(root.rglob("*.json")):
            if js.parent.name == "sweeps":
                continue
            try:
                row = run_row(js)
            except (OSError, ValueError, KeyError) as exc:
                print(f"[warn] {js.name}: {exc}", file=sys.stderr)
                continue
            if row:
                rows.append(row)
    handle = args.out.open("w", newline="") if args.out else sys.stdout
    w = csv.DictWriter(handle, fieldnames=COLUMNS)
    w.writeheader()
    w.writerows(rows)
    if args.out:
        handle.close()
        print(f"[energy_uncertainty] {len(rows)} runs -> {args.out}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
