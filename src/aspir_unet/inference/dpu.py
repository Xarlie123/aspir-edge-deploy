"""
Route B — VART + DPUCZDX8G inference backend.

Wraps `vart.Runner` for the .xmodel produced by `vai_c_xir` against
our DPU's arch.json. The DPU operates in INT8 throughout the
convolutional body; the only CPU-side work per frame is

    1. float32 input  → INT8 with the input tensor's per-tensor scale,
    2. submit job to the DPU, wait for completion,
    3. INT8 output    → float32 with the output tensor's per-tensor scale.

The two scales are baked into the compiled .xmodel and exposed via
`vart.Runner.get_input_tensors()[0].get_attr("fix_point")`. Power-of-
two fixed-point arithmetic — `fix_point=k` means the int8 value
encodes `value × 2^-k` in the float domain. We multiply / divide by
`2^k` at the host boundary.

This module is a no-op stub on systems that do not have VART/XIR
installed (e.g. local PyCharm autocomplete on the host). The actual
import only happens when an OnnxDpuRunner is instantiated, so the
module imports cleanly off-board for editing / static analysis.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, Iterator

import numpy as np


class DpuRunner:
    """Single-image INT8 DPU inference via VART."""

    def __init__(self, xmodel_path: Path | str) -> None:
        # XLNX_ENABLE_FINGERPRINT_CHECK=0 must be set BEFORE vart imports
        # its native libs — VART checks the env once at load time and
        # SIGBUSes during runner creation if the DPU fingerprint doesn't
        # match the one stored in the xmodel. /etc/environment sets this
        # for login shells but ssh non-interactive sessions don't pick
        # it up, so we set it here defensively. No-op when the env is
        # already in place.
        import os
        os.environ.setdefault("XLNX_ENABLE_FINGERPRINT_CHECK", "0")

        # Local imports so the module can be parsed on a host without
        # vart/xir installed (PyCharm autocomplete, unit tests, etc.).
        import vart
        import xir

        self._vart = vart
        self._xir = xir

        graph = xir.Graph.deserialize(str(xmodel_path))
        # Pick the single DPU subgraph (vai_c_xir reported "DPU subgraph
        # number 1" for our U-Net, see README §Phase C).
        root = graph.get_root_subgraph()
        dpu_subgraphs = [
            s for s in root.toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"
        ]
        if not dpu_subgraphs:
            raise RuntimeError(f"No DPU subgraph found in {xmodel_path}")
        if len(dpu_subgraphs) > 1:
            raise RuntimeError(
                f"Expected exactly 1 DPU subgraph, got {len(dpu_subgraphs)} "
                f"in {xmodel_path}"
            )
        self.runner: Any = vart.Runner.create_runner(dpu_subgraphs[0], "run")
        self.in_tensor = self.runner.get_input_tensors()[0]
        self.out_tensor = self.runner.get_output_tensors()[0]

        # fix_point: the per-tensor scale is 2 ** fix_point. The .xmodel
        # was produced by torch_quantizer in 8-bit power-of-two mode
        # (default), so this is always available.
        self.in_scale = float(2 ** self.in_tensor.get_attr("fix_point"))
        self.out_scale = float(2 ** self.out_tensor.get_attr("fix_point"))

        # Tensor dims may come back NCHW or NHWC depending on the
        # compiler; record both for sanity checks.
        self.in_dims = tuple(self.in_tensor.dims)
        self.out_dims = tuple(self.out_tensor.dims)
        self.last_breakdown: Dict[str, float] = {}

    def run_one(self, x_fp32: np.ndarray) -> np.ndarray:
        """Run inference on a single (1, 1, 64, 64) float32 frame.

        Returns (1, 1, 64, 64) float32 prediction (already dequantised).
        """
        if x_fp32.ndim == 3:
            x_fp32 = x_fp32[None]

        # Input tensor on the DPU is typically NHWC for DPUCZDX8G; our
        # fed array is NCHW. Convert if needed.
        if self.in_dims[-1] == 1 and x_fp32.shape[1] == 1:
            x_fp32 = x_fp32.transpose(0, 2, 3, 1)         # NCHW → NHWC

        t0 = perf_counter_ns()
        x_int = np.clip(np.rint(x_fp32 * self.in_scale), -128, 127).astype(np.int8)

        out_int = np.empty(self.out_dims, dtype=np.int8)
        t1 = perf_counter_ns()
        job_id = self.runner.execute_async([x_int], [out_int])
        self.runner.wait(job_id)
        t2 = perf_counter_ns()

        y_fp32 = out_int.astype(np.float32) / self.out_scale
        # Host-side conversion is part of what a deployment pays, so it stays
        # inside the timed call; keeping the components separate lets a result
        # be quoted at either boundary, which is what makes the comparison
        # against the GPU path (transfers included) meaningful.
        self.last_breakdown = {
            "convert_in_ms": (t1 - t0) / 1e6,
            "execute_ms": (t2 - t1) / 1e6,
            "convert_out_ms": (perf_counter_ns() - t2) / 1e6,
        }

        if self.out_dims[-1] == 1:
            y_fp32 = y_fp32.transpose(0, 3, 1, 2)         # NHWC → NCHW
        return y_fp32

    def run_batch(self, xs: np.ndarray) -> Iterator[np.ndarray]:
        for i in range(xs.shape[0]):
            yield self.run_one(xs[i:i + 1])

    def close(self) -> None:
        # vart.Runner has no explicit close in some versions; setting to
        # None lets GC reclaim the DPU handle.
        self.runner = None
