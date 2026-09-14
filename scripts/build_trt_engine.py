#!/usr/bin/env python3
"""
Build a TensorRT engine for one experiment — runs **on the Jetson**.

A serialised engine is tied to the GPU architecture and the TensorRT build that
produced it, so it cannot be cross-built on the workstation the way the DPU's
`.xmodel` can. Everything else about the artefact stays host-defined: the ONNX
comes from the experiment directory, and INT8 calibration uses the same
training-split frames the DPU quantiser used, so the two INT8 paths differ only
in their quantisation scheme and not in their data.

    python3 scripts/build_trt_engine.py --experiment <id> --precision int8
    python3 scripts/build_trt_engine.py --experiment <id> --precision fp16
    python3 scripts/build_trt_engine.py --experiment <id> --precision fp32

Writes into `models/<experiment>/jetson_trt/`:

    <experiment>_<precision>.engine        the engine
    <experiment>_<precision>.calib_cache   INT8 calibration table (reused)
    <experiment>_<precision>.layers.json   per-layer precision, from the inspector
    <experiment>_<precision>.build.json    the build report

The layer map matters: asking for INT8 does not mean every layer ran in INT8 —
TensorRT builds mixed-precision engines by default. The paper reports what
fraction actually did, which is only knowable from here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve()
for _src in (_HERE.parents[2] / "app" / "src", _HERE.parents[1] / "src"):
    if (_src / "aspir_unet").is_dir():
        sys.path.insert(0, str(_src))
        break

from aspir_unet import experiments, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None,
                        help="Experiment id under models/.")
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
    parser.add_argument("--onnx", type=Path, default=None,
                        help="Override the ONNX resolved from the experiment.")
    parser.add_argument("--calib-npz", type=Path, default=None,
                        help="Override the calibration set (INT8 only).")
    parser.add_argument("--calib-limit", type=int, default=200,
                        help="Calibration frames to use.")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the engine already exists.")
    parser.add_argument("--require-int8-fraction", type=float, default=None,
                        help="Fail, and delete the engine, if fewer than this "
                             "fraction of its layers ended up in INT8. An "
                             "engine that is not what was asked for must not "
                             "produce numbers: TensorRT's tactic selection can "
                             "quietly return an FP16 engine when INT8 was "
                             "requested, and the label would then be a lie.")
    parser.add_argument("--require-forced-layers", type=int, default=None,
                        help="Fail if fewer than this many compute layers were "
                             "pinned to INT8 (int8_strict only).")
    args = parser.parse_args()

    layout = experiments.discover_layout(__file__)
    if args.list_experiments:
        for name in experiments.list_experiments(layout):
            print(name)
        return 0

    exp = experiments.resolve(args.experiment, __file__, layout=layout)
    onnx_path = args.onnx or exp.require_onnx()
    engine_path = exp.trt_engine(args.precision)

    if engine_path.is_file() and not args.force:
        print(f"[skip] {engine_path} exists (use --force to rebuild)")
        return 0

    calib = None
    if args.precision.startswith("int8"):
        calib = args.calib_npz or exp.require_calibration()
        print(f"[calib] {calib} (up to {args.calib_limit} frames)")

    print(f"[exp]   {exp.experiment_id}")
    print(f"[onnx]  {onnx_path}")
    print(f"[build] precision={args.precision} -> {engine_path}")

    from aspir_unet.inference.trt import build_engine

    report = build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=args.precision,
        calib_npz=calib,
        calib_limit=args.calib_limit,
        workspace_mib=args.workspace_mib,
    )
    report["built_at"] = datetime.now().isoformat(timespec="seconds")
    report["experiment"] = exp.experiment_id
    report["device"] = provenance.device_identity()
    report["onnx_fingerprint"] = provenance.file_fingerprint(onnx_path)
    if calib is not None:
        report["calibration_fingerprint"] = provenance.file_fingerprint(Path(calib))

    # Verify before the engine is allowed to exist as a measurement input.
    counts = (report.get("engine_info") or {}).get("layer_precision_counts") or {}
    total = sum(counts.values()) or 0
    int8_fraction = counts.get("INT8", 0) / total if total else 0.0
    forced = report.get("layers_forced_int8")
    problems = []
    if args.require_int8_fraction is not None and \
            int8_fraction < args.require_int8_fraction:
        problems.append(
            f"INT8 fraction {int8_fraction:.2f} < required "
            f"{args.require_int8_fraction:.2f} ({counts})")
    if args.require_forced_layers is not None and \
            (forced or 0) < args.require_forced_layers:
        problems.append(
            f"only {forced} compute layers pinned, required "
            f"{args.require_forced_layers}")
    if problems:
        engine_path.unlink(missing_ok=True)
        for line in problems:
            print(f"[FAIL]  {line}")
        print(f"[FAIL]  engine deleted; it is not what was requested")
        return 2

    out = engine_path.with_suffix(".build.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"[save]  {out}")

    counts = (report.get("engine_info") or {}).get("layer_precision_counts")
    if counts:
        total = sum(counts.values())
        print("[build] layer precisions: " + ", ".join(
            f"{k}={v} ({100.0 * v / total:.0f}%)" for k, v in sorted(counts.items())))
    print(f"[build] {report['build_seconds']}s, "
          f"{report['engine_bytes'] / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
