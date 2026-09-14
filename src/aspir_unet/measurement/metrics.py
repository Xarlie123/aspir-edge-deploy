"""
Reconstruction quality metrics for the U-Net denoising task.

PSNR / SSIM / MAE between two NCHW float32 batches. We compute against
two references from the ASPIR test_beam_CCH dataset:

    *_vs_gt   — comparison against test_images.npz["originals"], i.e.
                the clean SPI source. This is the "task accuracy" the
                paper reports — how good the denoising is regardless
                of which model produced it.
    *_vs_pt   — comparison against test_images.npz["denoised"], i.e.
                the FP32 PyTorch model's own output. This is the
                "fidelity drop" induced by INT8 quantization + DPU
                arithmetic. A perfect DPU run would reproduce gold_pt
                exactly; the gap is purely numerical.

Implementation notes:
    - Inputs are assumed clipped to roughly [0, 1] but we clip to
      [0, 1] explicitly so PSNR uses data_range=1.0 cleanly.
    - SSIM uses scikit-image's structural_similarity with default
      Gaussian window (win_size=7) which works fine for 64×64 frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAVE_SKIMAGE = True
except Exception:           # pragma: no cover  — board may not have skimage
    _HAVE_SKIMAGE = False


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)


def psnr_per_image(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-frame PSNR (dB) for two (N, 1, H, W) batches in [0, 1].

    Returns (N,) array of dB. Frames whose MSE is exactly 0 get +inf.
    """
    pred = _clip01(pred)
    ref = _clip01(ref)
    mse = ((pred - ref) ** 2).reshape(pred.shape[0], -1).mean(axis=1)
    out = np.full_like(mse, np.inf)
    nonzero = mse > 0
    out[nonzero] = 10.0 * np.log10(1.0 / mse[nonzero])
    return out


def ssim_per_image(pred: np.ndarray, ref: np.ndarray) -> np.ndarray | None:
    """Per-frame SSIM in [-1, 1] for (N, 1, H, W) batches.

    Returns None if scikit-image is not installed (e.g. on a stripped
    board image).
    """
    if not _HAVE_SKIMAGE:
        return None
    pred = _clip01(pred)
    ref = _clip01(ref)
    n = pred.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = _ssim(pred[i, 0], ref[i, 0], data_range=1.0)
    return out


def mae_per_image(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-frame mean absolute error for (N, 1, H, W) batches."""
    return np.abs(pred - ref).reshape(pred.shape[0], -1).mean(axis=1)


@dataclass
class MetricSummary:
    """Aggregate statistics over all frames of a comparison."""

    psnr_mean: float
    psnr_p05: float
    psnr_p95: float
    ssim_mean: float | None
    mae_mean: float
    n_frames: int
    # Per-frame values, kept out of the JSON but archived alongside it. With a
    # 100-frame test split the statistical power lives in the paired
    # comparison (INT8 against FP32 on the *same* frames) and in bootstrap
    # intervals over frames; neither can be recovered from a mean.
    per_frame: Dict[str, np.ndarray] | None = None

    @classmethod
    def from_pair(cls, pred: np.ndarray, ref: np.ndarray,
                  keep_per_frame: bool = False) -> "MetricSummary":
        psnr = psnr_per_image(pred, ref)
        # +inf appears when an exact match exists; cap at 100 dB for the
        # statistics so a single frame does not blow up the mean.
        psnr_capped = np.minimum(psnr, 100.0)
        ssim = ssim_per_image(pred, ref)
        mae = mae_per_image(pred, ref)
        per_frame = None
        if keep_per_frame:
            per_frame = {"psnr_db": psnr_capped.astype(np.float32),
                         "mae": mae.astype(np.float32)}
            if ssim is not None:
                per_frame["ssim"] = ssim.astype(np.float32)
        return cls(
            per_frame=per_frame,
            psnr_mean=float(psnr_capped.mean()),
            psnr_p05=float(np.percentile(psnr_capped, 5)),
            psnr_p95=float(np.percentile(psnr_capped, 95)),
            ssim_mean=float(ssim.mean()) if ssim is not None else None,
            mae_mean=float(mae.mean()),
            n_frames=int(pred.shape[0]),
        )

    def to_dict(self) -> Dict[str, float | int | None]:
        return {
            "n_frames": self.n_frames,
            "psnr_mean_db": round(self.psnr_mean, 3),
            "psnr_p05_db": round(self.psnr_p05, 3),
            "psnr_p95_db": round(self.psnr_p95, 3),
            "ssim_mean": round(self.ssim_mean, 4) if self.ssim_mean is not None else None,
            "mae_mean": round(self.mae_mean, 6),
        }
