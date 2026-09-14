# meta-user template for the 2022.2 PetaLinux project

Copy these files into `project-spec/meta-user/` of the new PetaLinux 2022.2
project after `petalinux-create` runs. Designed to layer cleanly on top of
the Avnet u96v2_sbc_base BSP 2022.2 default meta-user/ tree.

## What it contains

- `recipes-kernel/linux/linux-xlnx_%.bbappend` + `bsp.cfg` — `CONFIG_SENSORS_INA2XX=y`
  for the on-board INA226 (read by ASPIR EnergyAnalyzer), plus pinning
  `CONFIG_FPGA_MGR_ZYNQMP_FPGA=y` so zocl can program the PL from xclbins.
- `recipes-bsp/device-tree/device-tree.bbappend` + `files/system-user.dtsi`
  — empty DT hook reserved for future custom IP nodes. Avnet's BSP DT
  already wires INA226 + fpga-region + DPU IRQ path.
- `recipes-core/images/petalinux-image-minimal.bbappend` — IMAGE_INSTALL
  with VART 3.0 + xir + target-factory + unilog + glog + **vitis-ai-library**
  (works clean on v3.0, unlike v4.0 where it was broken) + dev tools
  (gdb, strace, htop, perf, tmux, vim, rsync) + Python (numpy, pillow, pip).
- `conf/petalinuxbsp.conf` — `PREFERRED_VERSION_vart = "3.0"` to pick the
  Vitis-flow recipe over `vart_3.0_vivado.bb`.
- `conf/user-rootfsconfig` — menuconfig hooks for `gpio-demo`, `peekpoke`.

## How to apply (inside the 2022.2 container)

```sh
# 1. Create the project from the Avnet BSP
cd /workspace/projects
petalinux-create -t project -s /workspace/bsp/u96v2_sbc_base_2022_2.bsp -n ultra96v2_2022.2
cd ultra96v2_2022.2

# 2. Overlay our customizations on top of Avnet's defaults
cp -r /workspace/projects/ultra96v2/meta-user-2022.2-template/* project-spec/meta-user/

# 3. Link in the Vitis-AI 3.0 recipe tree (mirrors what the 2024.1 project did
#    against v4.0). The submodule is already checked out at v3.0.
cd project-spec/meta-user
ln -s ../../../../sources/Vitis-AI/src/vai_petalinux_recipes/recipes-vitis-ai recipes-vitis-ai
ln -s ../../../../sources/Vitis-AI/src/vai_petalinux_recipes/recipes-vai-kernel recipes-vai-kernel

# 4. Re-config + build
cd ../..
petalinux-config --silentconfig
petalinux-build
```

## Diff vs the 2024.1 meta-user (what we LEFT BEHIND)

- All 6 bbappends under `recipes-vai-extra/` — pure workarounds for VAI v4.0
  source bugs (merge markers, absl::log typo, duplicate add_subdirectory,
  vitis-ai-library missing CMakeLists, etc.). None of these bugs exist
  in the v3.0 tag.
- `vitis-ai-library` and `vai-benchmark` are RE-ADDED (they were excluded
  in 2024.1 because of the broken v4.0 sources).
- `PREFERRED_VERSION_vart` retargeted from "3.5" to "3.0".
- u-boot ubifs-distroboot patch and platform-top.h — TBD whether to port.
  We boot from SD on the U96v2, so distroboot QSPI path isn't required for
  Phase A/B/C measurement. Skipped from this template; add later if needed.
