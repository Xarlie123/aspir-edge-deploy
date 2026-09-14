"""
Route A — ONNX Runtime + ARM NEON inference backend.

Wraps `onnxruntime.InferenceSession` for the FP32 ONNX export of our
U-Net. Runs entirely on the four Cortex-A53 cores via NEON-vectorised
matmul kernels in the CPUExecutionProvider — no DPU involvement.

Inputs and outputs stay in float32 the whole way; there is no
quantisation step on this route by design (ONNX Runtime can run a
quantised .onnx too, but for the paper we want a pure FP32 baseline
to compare DPU INT8 against).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import onnxruntime as ort


class OnnxNeonRunner:
    """Single-image FP32 inference over ONNX Runtime CPU EP."""

    def __init__(self, onnx_path: Path | str, intra_threads: int = 4) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run_one(self, x: np.ndarray) -> np.ndarray:
        """Run inference on a single (1, 1, 64, 64) float32 frame.

        Returns (1, 1, 64, 64) float32 prediction.
        """
        if x.ndim == 3:
            x = x[None]
        return self.session.run([self.output_name], {self.input_name: x})[0]

    def run_batch(self, xs: np.ndarray) -> Iterator[np.ndarray]:
        """Yield per-frame predictions iterating over the batch axis."""
        for i in range(xs.shape[0]):
            yield self.run_one(xs[i:i + 1])
