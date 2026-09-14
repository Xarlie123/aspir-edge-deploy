# DPU overlay for the Ultra96-V2

The accelerator used in the article: one DPUCZDX8G v4.1 core, B512
configuration, instantiated as an IP core in a Vivado block design and driven
by VART's vivado-flow runtime through `/dev/dpu` (no XRT or zocl).

| Parameter | Value |
|---|---|
| Device | Zynq UltraScale+ XCZU3EG (`xczu3eg-sbva484-1-i`) |
| DPU | DPUCZDX8G v4.1, B512, single core, ALU_PARALLEL = 4, no softmax |
| Clocks | `clk_dpu` 300 MHz, `clk_dsp` 600 MHz |
| Fingerprint | `0x101000056010400` (`arch.json`) |
| Control base | `0x8F000000` |
| Memory ports | M_AXI_INSTR → HPC0; M_AXI_DATA0 → HP2; M_AXI_DATA1 → HP3 |
| Implementation strategy | `Performance_ExplorePostRoutePhysOpt` |
| Timing (reference build) | WNS +0.069 ns, TNS 0.000 ns |

The implementation strategy matters: B512 at 300 MHz on the XCZU3EG closes
timing with `Performance_ExplorePostRoutePhysOpt`, and did not with the
reference design's default `Performance_Explore`.

## Contents

| File | Purpose |
|---|---|
| `create_bd.tcl` | `write_bd_tcl` export of the block diagram: PS configuration, clocking, resets, DPU, address map. The source of truth for the hardware. |
| `regenerate_project.tcl` | Recreates the Vivado project from `create_bd.tcl`; with `--build` also runs synthesis, implementation, bitstream and XSA export. |
| `constraints/u96v2_sbc_base.xdc` | Board pin constraints, from the Avnet Ultra96-V2 base design. |
| `arch.json` | DPU descriptor passed to `vai_c_xir`; its fingerprint must match the bitstream. |
| `top_wrapper.bit`, `top_wrapper.xsa` | The bitstream and hardware export used for every measurement in the article. |
| `dpu_power_routed.rpt` | Vivado `report_power` on the routed design (vectorless estimate). |
| `gen_dpu_overlay.sh` | XSA → `pl.dtbo` + `.bit.bin`, for loading the overlay at runtime with `fpgautil`. |
| `fetch-dpu-trd.sh` | Downloads AMD's DPUCZDX8G_VAI_v3.0 reference design, which contains the DPU IP. |

**The DPU IP itself is not included.** AMD distributes it only inside the
reference-design archive, under its own licence; `fetch-dpu-trd.sh` downloads it
into `hardware/dpu-trd/`, which is git-ignored. No hash of the archive was
recorded when the published bitstream was built; set `DPU_TRD_SHA256` to
enforce one.

## Tool versions — read this before rebuilding

The Linux side is PetaLinux 2022.2 (kernel 5.15.36-xilinx-v2022.2), and
`regenerate_project.tcl` describes itself as producing a Vivado 2022.2
project. **The tracked `top_wrapper.xsa` and `top_wrapper.bit`, however, report
tool version 2024.1** (`TOOL_VERSION` in the XSA, `Version=` in the bitstream
header). The published measurements were taken with this bitstream and a
PetaLinux 2022.2 image, and the combination was validated end to end on the
board. A rebuild with Vivado 2022.2 is expected to be functionally equivalent
but has not been verified against these artefacts.

## Rebuilding

```bash
# 1. DPU IP source (~130 MB from AMD).
bash hardware/dpu_overlay/fetch-dpu-trd.sh

# 2. Vivado project from the block-diagram TCL (seconds).
vivado -mode batch -source hardware/dpu_overlay/regenerate_project.tcl

# 3. Optional full build: synthesis, implementation, bitstream, XSA.
vivado -mode batch -source hardware/dpu_overlay/regenerate_project.tcl -tclargs --build

# 4. Runtime overlay from an XSA.
XILINX_DIR=/path/to/Xilinx bash hardware/dpu_overlay/gen_dpu_overlay.sh
```

A rebuilt bitstream differs in timing-driven placement; what must be preserved
is the fingerprint, since every compiled model in `models/` is bound to it.

## On the board

The bitstream and overlay are loaded at runtime, not at boot, by the
`aspir-load-dpu` helper that the PetaLinux image installs (see
`envs/ultra96v2/`). The CPU runner unloads the programmable logic before
measuring, so the CPU baseline is not taken with the accelerator powered, and
the DPU runner reloads it when `/dev/dpu` is missing.
