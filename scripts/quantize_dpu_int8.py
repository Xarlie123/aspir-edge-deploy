#!/usr/bin/env python3
"""
Phase C — host-side quantization of the ASPIR U-Net (beam_gaussian_unet_8pct_cch).

Run inside the Vitis-AI 3.5 PyTorch CPU container:

    ./docker/vitis-ai-run.sh
    conda activate vitis-ai-pytorch
    python scripts/quantize_dpu_int8.py --model-dir models/<experiment>

Outputs (under models/<experiment>/phase_c_dpu/):
    UNet_int.xmodel    — architecture-neutral, INT8-quantized graph
    quant_info.json    — calibration ranges, useful for debugging
    bias_corr.pth, ... — quantizer scratch state

The .xmodel is then handed to vai_c_xir together with hardware/dpu_overlay/arch.json
to produce the board-runnable .xmodel keyed to the DPUCZDX8G B512 fingerprint.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from aspir_unet.model import UNet, UNetRes  # noqa: E402

from pytorch_nndct.apis import torch_quantizer  # noqa: E402


# Architecture name → class. Picked from test_config.json's `model_name` field
# (u-net → UNet, u-net-residual → UNetRes).
ARCHITECTURES = {
    "u-net":          UNet,
    "u-net-residual": UNetRes,
}

CALIB_N = 200          # 200 of 1000 is enough for PTQ on a tiny U-Net
CALIB_BATCH = 8

# Which array inside test_images.npz is the actual U-Net input. The npz
# carries three arrays for the test_beam_CCH dataset:
#   originals       — ground truth IR frames (the SPI source)
#   reconstructions — Hadamard linear-inverse reconstructions, noisy
#                     because the basis covers only 8% of the 4096
#                     vectors. THIS is what the U-Net was trained on
#                     and what it must denoise.
#   denoised        — the FP32 PyTorch U-Net's own output, kept here
#                     as the gold reference for PSNR/SSIM vs the DPU.
# We must calibrate quantization against the distribution the model
# actually sees in production, i.e. reconstructions. Calibrating
# against originals (range [0,1]) instead of reconstructions
# (range [0, 0.985] but with very different second-moment statistics
# — noisy textures absent from clean images) shifts the activation
# histograms and degrades INT8 accuracy.
INPUT_KEY = "reconstructions"


def _fingerprint(path: Path) -> dict:
    """sha256 and size of a file. Deliberately local rather than imported from
    aspir_unet.provenance: this script runs under the container's Python 3.7
    and the point of keeping its imports thin is that a missing dependency here
    breaks every model the campaign deploys."""
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": path.name, "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def load_state_dict(model: torch.nn.Module, pt_path: Path) -> None:
    """Tolerate both raw state_dict and {model:..., epoch:...} checkpoints.

    Backwards-compat: the Vitis-AI 3.0 docker ships PyTorch 1.12 which does
    not have the `weights_only` kwarg (added in 1.13). Modern hosts have it.
    """
    try:
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(pt_path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
    elif isinstance(blob, dict) and all(isinstance(v, torch.Tensor) for v in blob.values()):
        sd = blob
    else:
        sd = blob.state_dict() if hasattr(blob, "state_dict") else blob
    model.load_state_dict(sd, strict=True)


def normalise_to_nchw(arr: np.ndarray) -> np.ndarray:
    """Force shape (N, C, H, W) regardless of how the npz was stored."""
    if arr.ndim == 3:                       # (N, H, W)  → add channel
        arr = arr[:, None, :, :]
    elif arr.ndim == 4 and arr.shape[-1] in (1, 3):  # (N, H, W, C) → channel-first
        arr = arr.transpose(0, 3, 1, 2)
    elif arr.ndim == 4 and arr.shape[1] in (1, 3):   # already (N, C, H, W)
        pass
    else:
        raise ValueError(f"Unexpected calibration array shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path,
        required=True,
        help="Path to the model directory (must contain source/ with .pt + "
             "test_config.json + test_images.npz). Default: UNet vanilla.")
    parser.add_argument(
        "--calib-npz", type=Path, default=None,
        help="Calibration set. Defaults to source/calibration_images.npz, "
             "which must hold training-split frames disjoint from the "
             "evaluation set. The same frames must be used to calibrate the "
             "TensorRT engines, or part of any quality difference between the "
             "two INT8 paths is an artefact of the calibration data.")
    parser.add_argument(
        "--calib-n", type=int, default=CALIB_N,
        help="How many calibration frames to run.")
    parser.add_argument(
        "--allow-test-calibration", action="store_true",
        help="Calibrate on the evaluation set. Smoke tests only — never for "
             "numbers that reach the paper.")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    source = model_dir / "source"
    out_dir = model_dir / "phase_c_dpu" / "quantizer_scratch"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((source / "test_config.json").read_text())
    arch_key = cfg.get("model_name", "u-net")
    if arch_key not in ARCHITECTURES:
        raise SystemExit(
            f"Unknown architecture '{arch_key}' in test_config.json. "
            f"Known: {list(ARCHITECTURES)}.")
    Arch = ARCHITECTURES[arch_key]
    features = cfg["architecture_config"]["features"]
    in_ch = cfg.get("in_channels", 1)
    out_ch = cfg.get("out_channels", 1)
    # Infer img_size from the npz instead of hardcoding it — different
    # ASPIR experiments use different resolutions (test_beam_CCH was 64,
    # beam_gaussian_unet_8pct_cch is 32).
    npz_probe = np.load(source / "test_images.npz")
    probe_arr = npz_probe[INPUT_KEY]
    if probe_arr.ndim == 3:
        img_size = probe_arr.shape[-1]
    elif probe_arr.ndim == 4:
        img_size = probe_arr.shape[-1]
    else:
        raise ValueError(f"Unexpected probe array shape {probe_arr.shape}")
    npz_probe.close()

    print(f"[arch] {Arch.__name__} features={features}, in_channels={in_ch}, "
          f"out_channels={out_ch}, img_size={img_size}")
    print(f"[paths] model_dir = {model_dir.relative_to(REPO_ROOT)}")
    print(f"[paths] out_dir   = {out_dir.relative_to(REPO_ROOT)}")

    model = Arch(in_channels=in_ch, out_channels=out_ch, features=features).eval()
    pt_path = next(source.glob("*.pt"))
    print(f"[load] state_dict from {pt_path.name}")
    load_state_dict(model, pt_path)

    # Calibration data: the training-split frames shipped as
    # calibration_images.npz. Calibrating on the evaluation set inflates the
    # quality figures and is the objection most likely to sink the paper, so
    # the evaluation set is used only when explicitly asked for.
    calib_path = args.calib_npz
    if calib_path is None:
        default_calib = source / "calibration_images.npz"
        if default_calib.is_file():
            calib_path = default_calib
        elif args.allow_test_calibration:
            calib_path = source / "test_images.npz"
            print("[calib] WARNING calibrating on the EVALUATION set "
                  "(--allow-test-calibration). Results are not publishable; "
                  "this is a smoke-test path only.")
        else:
            raise SystemExit(
                f"no calibration set at {default_calib}.\n"
                f"Ship calibration_images.npz (training-split frames) with the "
                f"experiment, pass --calib-npz explicitly, or use "
                f"--allow-test-calibration for a throwaway smoke test."
            )

    npz = np.load(calib_path)
    available = list(npz.keys())
    print(f"[data] {Path(calib_path).name} contains arrays: {available}")
    if INPUT_KEY not in available:
        raise KeyError(
            f"Expected '{INPUT_KEY}' in {calib_path} but found {available}. "
            f"Edit INPUT_KEY at the top of this script if the dataset uses "
            f"different naming conventions."
        )
    imgs = normalise_to_nchw(npz[INPUT_KEY])
    print(f"[data] using '{INPUT_KEY}' for calibration: shape {imgs.shape} "
          f"dtype {imgs.dtype}, range [{imgs.min():.4f}, {imgs.max():.4f}]")
    assert imgs.shape[1] == in_ch and imgs.shape[2] == imgs.shape[3] == img_size, \
        f"Expected (N,{in_ch},{img_size},{img_size}), got {imgs.shape}"

    n = min(args.calib_n, len(imgs))
    dummy = torch.from_numpy(imgs[:1]).float()
    device = torch.device("cpu")

    # ---- calibration pass: collect per-tensor activation ranges ----
    print(f"[calib] running {n} images in batches of {CALIB_BATCH}")
    quantizer = torch_quantizer(
        quant_mode="calib",
        module=model,
        input_args=(dummy,),
        output_dir=str(out_dir),
        device=device,
    )
    qmodel = quantizer.quant_model
    with torch.no_grad():
        for i in range(0, n, CALIB_BATCH):
            batch = torch.from_numpy(imgs[i:i + CALIB_BATCH]).float()
            qmodel(batch)
    quantizer.export_quant_config()
    print("[calib] done")

    # ---- test pass: re-instantiate, single forward, emit the deployable xmodel ----
    print("[test ] exporting deployable xmodel")
    quantizer = torch_quantizer(
        quant_mode="test",
        module=model,
        input_args=(dummy,),
        output_dir=str(out_dir),
        device=device,
    )
    qmodel = quantizer.quant_model
    with torch.no_grad():
        _ = qmodel(dummy)
    # export_xmodel has its own output_dir argument that defaults to
    # ./quantize_result/ (cwd-relative) and ignores the output_dir we
    # gave to the torch_quantizer constructor. Pass it explicitly.
    quantizer.export_xmodel(output_dir=str(out_dir), deploy_check=True)

    # ---- our own provenance, beside the compiler's meta.json --------------
    # meta.json is written by vai_c_xir and read by VART at load time to find
    # the runner library and the model; it is a vendor artefact and we do not
    # add fields to it. This file records what produced the quantised graph.
    # The torch version matters because it is not the one that trained the
    # model: training runs on the workstation's PyTorch, while vai_q_pytorch in
    # Vitis AI 3.0 is pinned to its own, four years older. The container tag is
    # what makes that reproducible, and it is passed in from the build script
    # because nothing inside a container can see the tag it was started from.
    prov = {
        "produced_by": "quantize_dpu_int8.py",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "vitis_ai_image": os.environ.get("ASPIR_VITIS_IMAGE"),
        "architecture": Arch.__name__,
        "calibration_frames": int(n),
        "artifacts": {
            "checkpoint": _fingerprint(pt_path),
            "calibration": _fingerprint(calib_path),
        },
    }
    try:
        import pytorch_nndct
        prov["vai_q_pytorch"] = getattr(pytorch_nndct, "__version__", None)
    except Exception:
        prov["vai_q_pytorch"] = None
    # Beside the xmodel, not inside quantizer_scratch/: deploy.sh excludes the
    # scratch directory, so provenance written there would never reach the board
    # nor travel with the artefact it describes.
    prov_path = out_dir.parent / "quantizer_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2))
    print(f"[prov ] {prov_path.name} beside the xmodel  torch={prov['torch']} "
          f"image={prov['vitis_ai_image']}")

    print("\n=== Outputs in {}: ===".format(out_dir.relative_to(REPO_ROOT)))
    for p in sorted(out_dir.glob("*")):
        size = p.stat().st_size
        print(f"  {p.name:40s} {size:>12,d} bytes")

    # The .xmodel filename emitted by vai_q_pytorch follows the model
    # class name (`UNet_int.xmodel` / `UNetRes_int.xmodel`). The compiled
    # board-side xmodel name is derived from the model dir basename.
    src_xmodel = f"{Arch.__name__}_int.xmodel"
    out_name = f"{model_dir.name.replace('beam_64_gaussian_', '').replace('_8pct_cch', '')}_b512_u96v2"
    print(
        "\nNext step (still on the host, inside the same Vitis-AI container):\n"
        "    vai_c_xir \\\n"
        f"        -x {out_dir.relative_to(REPO_ROOT)}/{src_xmodel} \\\n"
        "        -a hardware/dpu_overlay/arch.json \\\n"
        f"        -o {(model_dir / 'phase_c_dpu').relative_to(REPO_ROOT)} \\\n"
        f"        -n {out_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
