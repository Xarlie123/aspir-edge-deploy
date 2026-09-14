"""
Loading and shape-normalisation for the ASPIR test_beam_CCH dataset.

The npz produced by the ASPIR batch test contains three FP32 arrays
of identical shape (N, 64, 64):

    originals       — clean ground-truth IR frames (the SPI source)
    reconstructions — Hadamard linear-inverse reconstructions, noisy;
                      THIS is the U-Net's input
    denoised        — the FP32 PyTorch U-Net's own output, kept here
                      as the gold reference for PSNR / SSIM vs the DPU run

This module is the single source of truth for "what is the U-Net
input at evaluation time". Both the host quantizer (calibration) and
the on-board inference scripts read through here so we cannot
accidentally feed clean originals to the model in one place and
noisy reconstructions in another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


INPUT_KEY = "reconstructions"
GOLD_KEY = "denoised"
GROUND_TRUTH_KEY = "originals"


@dataclass(frozen=True)
class TestSet:
    """A bundle of paired (input, gold, ground-truth) tensors in NCHW float32."""

    inputs: np.ndarray          # (N, 1, 64, 64), the noisy reconstructions
    gold_pt: np.ndarray         # (N, 1, 64, 64), the FP32 PT reference output
    ground_truth: np.ndarray    # (N, 1, 64, 64), the clean originals
    n: int                      # convenience: inputs.shape[0]


def _to_nchw_float32(arr: np.ndarray) -> np.ndarray:
    """Normalise an arbitrary npz array to (N, 1, 64, 64) float32.

    The ASPIR npz stores the three arrays as (N, 64, 64) float32, but
    older versions of the dataset have used (N, 64, 64, 1) channel-last
    or (N, 1, 64, 64) channel-first. Tolerate all three.
    """
    if arr.ndim == 3:                                # (N, H, W) → add channel
        arr = arr[:, None, :, :]
    elif arr.ndim == 4 and arr.shape[-1] == 1:        # (N, H, W, 1) → channel-first
        arr = arr.transpose(0, 3, 1, 2)
    elif arr.ndim == 4 and arr.shape[1] == 1:         # already (N, 1, H, W)
        pass
    else:
        raise ValueError(f"Cannot interpret npz array with shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_test_set(npz_path: Path | str) -> TestSet:
    """Open the test_images.npz and return a TestSet bundle."""
    npz = np.load(str(npz_path))
    available = list(npz.keys())
    for key in (INPUT_KEY, GOLD_KEY, GROUND_TRUTH_KEY):
        if key not in available:
            raise KeyError(
                f"Expected '{key}' in {npz_path} but found {available}. "
                f"This loader assumes the ASPIR test_beam_CCH layout."
            )

    inputs = _to_nchw_float32(npz[INPUT_KEY])
    gold = _to_nchw_float32(npz[GOLD_KEY])
    gt = _to_nchw_float32(npz[GROUND_TRUTH_KEY])
    if not (inputs.shape == gold.shape == gt.shape):
        raise ValueError(
            f"Shape mismatch in {npz_path}: inputs={inputs.shape}, "
            f"gold={gold.shape}, gt={gt.shape}"
        )
    return TestSet(inputs=inputs, gold_pt=gold, ground_truth=gt, n=inputs.shape[0])
