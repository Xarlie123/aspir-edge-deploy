"""Model definitions: the FP32 U-Net used throughout the campaign, and a
residual variant the DPU quantiser can also load."""

from .unet import UNet
from .unet_res import UNetRes

__all__ = ["UNet", "UNetRes"]
