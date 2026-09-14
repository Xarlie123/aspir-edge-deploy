#!/usr/bin/env python3
"""
Quantise an experiment's ONNX to INT8 for the CPU route (ONNX Runtime).

The point of this artefact is a control, not a deployment: comparing an INT8
accelerator against an FP32 CPU baseline mixes two effects — the hardware and
the data width. Running the CPU at INT8 too separates them.

It also probes an architectural asymmetry between the two boards. The Orin's
Cortex-A78AE is ARMv8.2 and carries the integer dot-product instruction
(`asimddp`); the Ultra96's Cortex-A53 is ARMv8.0 and does not. INT8 should pay
on one and not on the other, and that is worth measuring rather than asserting.

Calibration uses the **same 200 training-split frames** the DPU quantiser and
the TensorRT calibrator use, so a difference between the three INT8 paths is
attributable to the scheme and not to the data.

Run it with the same ONNX Runtime version that will execute the model — the
quantised graph is written for the runtime that reads it.

    python3 scripts/quantize_cpu_int8.py --experiment <id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _src in (_HERE.parents[2] / "app" / "src", _HERE.parents[1] / "src"):
    if (_src / "aspir_unet").is_dir():
        sys.path.insert(0, str(_src))
        break

from aspir_unet import experiments, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--calib-limit", type=int, default=200)
    parser.add_argument("--per-channel", action="store_true",
                        help="Per-channel weight scales. Off by default to "
                             "stay closer to the DPU's per-tensor scheme.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    layout = experiments.discover_layout(__file__)
    if args.list_experiments:
        for name in experiments.list_experiments(layout):
            print(name)
        return 0

    exp = experiments.resolve(args.experiment, __file__, layout=layout)
    src = exp.require_onnx()
    dst = src.with_name(src.stem + "_int8.onnx")
    if dst.is_file() and not args.force:
        print(f"[skip] {dst.name} exists (use --force)")
        return 0

    calib = exp.require_calibration()
    frames = np.load(calib)["reconstructions"][:args.calib_limit]
    if frames.ndim == 3:
        frames = frames[:, None]
    frames = np.ascontiguousarray(frames.astype(np.float32))

    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)

    input_name = ort.InferenceSession(
        str(src), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter([{input_name: frames[i:i + 1]}
                            for i in range(frames.shape[0])])

        def get_next(self):
            return next(self.it, None)

    print(f"[exp]   {exp.experiment_id}")
    print(f"[calib] {calib.name}, {frames.shape[0]} frames")
    print(f"[quant] {src.name} -> {dst.name} "
          f"(QDQ, {'per-channel' if args.per_channel else 'per-tensor'})")

    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    # Report what the quantised graph is, and prove it still runs.
    import onnx
    model = onnx.load(str(dst))
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: frames[:1]})[0]

    meta = {
        "experiment": exp.experiment_id,
        "source_onnx": provenance.file_fingerprint(src),
        "quantised_onnx": provenance.file_fingerprint(dst),
        "calibration": provenance.file_fingerprint(calib),
        "calibration_frames": int(frames.shape[0]),
        "onnxruntime": ort.__version__,
        "quant_format": "QDQ",
        "per_channel": bool(args.per_channel),
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "op_counts": ops,
        "output_shape": list(out.shape),
    }
    (dst.with_suffix(".quant.json")).write_text(json.dumps(meta, indent=2))
    print(f"[quant] ops: " + ", ".join(f"{k}={v}" for k, v in sorted(ops.items())))
    print(f"[save]  {dst.name} ({dst.stat().st_size / 1024:.0f} KiB) + .quant.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
