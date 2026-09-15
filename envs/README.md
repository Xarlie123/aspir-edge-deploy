# Environments

| Environment | Used for | Pins |
|---|---|---|
| Host workstation | Aggregation and figures | `host/requirements.txt` |
| Vitis AI 3.0 container | DPU quantisation and compilation | `vitis-ai/IMAGE` |
| Ultra96-V2, PetaLinux 2022.2 | DPU and CPU measurements | `ultra96v2/` |
| Jetson Orin NX, JetPack 6.2 | GPU and CPU measurements | `jetson/requirements.txt` |

## Host workstation

Ubuntu 24.04, Python 3.12.3; `host/requirements.txt`. The networks were trained
in ASPIR v1.2.1's own environment with PyTorch 2.6.0 (CUDA 12.4), which is not
needed here.

## Vitis AI 3.0

`vitis-ai/IMAGE` gives the container by tag and by digest. Inside it:
PyTorch 1.12.1, Python 3.7.12, `vai_q_pytorch` 3.0.0+a44284e, conda
environment `vitis-ai-pytorch`. Only `scripts/build_dpu_xmodels.sh` uses it, and
every compiled model records these versions in its
`quantizer_provenance.json`.

## Ultra96-V2 — PetaLinux 2022.2

Kernel 5.15.36-xilinx-v2022.2, Python 3.9.9, VART 3.0.0.
`ultra96v2/requirements.txt` lists the Python packages installed on the board
(`scripts/install-board-deps.sh` installs them).

`ultra96v2/petalinux/project-spec/` holds the parts of the PetaLinux project
that are specific to this work:

| Path | What it does |
|---|---|
| `meta-user/recipes-vitis-ai/` | Vitis AI 3.0 runtime recipes: `xir`, `unilog` (with a Python-path patch for PetaLinux), `target-factory`, `vart` and `vitis-ai-library` |
| `meta-user/recipes-vai-kernel/` | An additional kernel configuration fragment (`bsp.cfg`) for `linux-xlnx` |
| `meta-user/recipes-vitis-ai-route-c/vart/vart_3.0.bbappend` | Builds VART 3.0 for the vivado flow (`PACKAGECONFIG:remove = "vitis"`), so it talks to `/dev/dpu` instead of XRT |
| `meta-user/recipes-kernel/linux/linux-xlnx/route-c.cfg` | Kernel options: the DPU driver (`CONFIG_XILINX_DPU`), FPGA manager and overlays, the USB PHY options the DPU driver's dependencies otherwise disable, UIO |
| `meta-user/recipes-bsp/` | Device-tree fragments and U-Boot configuration |
| `meta-user/recipes-core/images/petalinux-image-minimal.bbappend` | Installs `aspir-load-dpu` and `aspir-unload-dpu`, the udev rule for `/dev/dpu`, the runtime environment and the `PATH` used by non-interactive SSH |
| `configs/` | PetaLinux `config` and `rootfs_config`, and small service configuration files |

The image recipe also sets up a development board: passwordless `sudo` for the
`petalinux` user, no password expiry, and SSH public keys installed from a
directory beside the project when one exists (none are shipped). Review those
choices before using the image outside a lab.

`XLNX_ENABLE_FINGERPRINT_CHECK=0` is exported because VART's runtime fingerprint
check crashed on this build of VART and XIR. It is not hiding a mismatch: the
compiled models target `DPUCZDX8G_ISA1_B512_0101000056010400`, and
`xdputil query` on the loaded overlay reports the same fingerprint.

### Building the image

The project's first user layer (`CONFIG_USER_LAYER_0`) is Avnet's `meta-avnet`
for the Ultra96-V2 (Yocto honister), which is **not included**; obtain it with
Avnet's Ultra96-V2 PetaLinux 2022.2 BSP. In outline:

```bash
petalinux-create -t project -s <Avnet Ultra96-V2 2022.2 BSP> -n ultra96v2
cp -r envs/ultra96v2/petalinux/project-spec/* ultra96v2/project-spec/
cd ultra96v2
petalinux-config --get-hw-description <path>/hardware/dpu_overlay/u96v2_sbc_base_wrapper.xsa
petalinux-build
petalinux-package --boot --u-boot --force
```

Two of the BSP's Wi-Fi recipes (`wilc3000-fw_16.2.bb`, `wilc_16.2.bb`) fetched
over `git://`, which GitHub no longer serves; they had to be switched to
`https://`. The generated SD image ships a small root partition; grow it before
installing the Python packages and datasets.

## Jetson Orin NX — JetPack 6.2

L4T R36.4.3, Ubuntu 22.04.5, kernel 5.15.148-tegra, Python 3.10.12, TensorRT
10.3.0 from JetPack; `jetson/requirements.txt`. Clocks were pinned with
`sudo jetson_clocks` before measuring, and every report records the resulting
state (CPU 1497.6 MHz, GPU 1173 MHz). The `nvpmodel` power mode is not recorded
in the reports.
