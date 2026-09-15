# DPU overlay for the Ultra96-V2

The accelerator used in the article: one DPUCZDX8G v4.1 core, B512
configuration, instantiated as an IP core in a Vivado block design and driven
by VART's vivado-flow runtime through `/dev/dpu` (no XRT or zocl).

| Parameter | Value |
|---|---|
| Device | Zynq UltraScale+ XCZU3EG (`xczu3eg-sbva484-1-i`) |
| DPU | DPUCZDX8G v4.1, B512, single core, ALU_PARALLEL = 4, LOAD_PARALLEL = 2, no softmax |
| Clocks | `clk_dpu` 300 MHz, `clk_dsp` 600 MHz |
| Fingerprint | `0x101000056010400` |
| Tools | Vivado 2022.2 (build 3671981), the release matching the PetaLinux 2022.2 image |
| Implementation | "Vivado Implementation Defaults", 2026-05-14 |
| Timing | WNS +0.189 ns, TNS 0.000 ns, WHS +0.010 ns; no failing endpoint of 116,736 |
| Placed utilisation | 28,767 LUTs (40.8 %), 37,546 flip-flops (26.6 %), 72 BRAM tiles (33.3 %), 134 DSPs (37.2 %) |
| Power estimate | 3.558 W on-chip: 3.233 W dynamic, 0.326 W static (vectorless) |

Both figures come from the reports in `reports/` and cover the whole
programmable logic, not the DPU alone; the power estimate also includes the
processing system (PS8, 1.640 W).

## Contents

| File | Purpose |
|---|---|
| `create_bd.tcl` | `write_bd_tcl` export of the block diagram (`u96v2_sbc_base`), from the Vivado 2022.2 project that produced the measured bitstream |
| `regenerate_project.tcl` | Recreates the Vivado project from `create_bd.tcl`; with `--build` also runs synthesis and implementation with the measured build's strategy, and exports `u96v2_sbc_base_wrapper_regen.xsa` |
| `constraints/u96v2_sbc_base.xdc` | Board constraints, identical to those of the measured project; from the Avnet Ultra96-V2 base design |
| `arch.json` | DPU descriptor passed to `vai_c_xir` |
| `u96v2_sbc_base_wrapper.xsa` | Hardware export of the measured build: bitstream `u96v2_sbc_base_wrapper.bit` and hardware handoff |
| `firmware/u96v2_sbc_base_wrapper.bit.bin` | The file loaded on the board, taken from the PetaLinux image |
| `firmware/base.dtbo` | The device-tree overlay loaded with it |
| `reports/` | Timing summary, placed utilisation and routed power estimate of the measured build |
| `gen_dpu_overlay.sh` | XSA → `pl.dtbo` + `.bit.bin`, for trying a different overlay without rebuilding the image |
| `fetch-dpu-trd.sh` | Downloads AMD's DPUCZDX8G_VAI_v3.0 reference design, which contains the DPU IP |

**The DPU IP itself is not included.** AMD distributes it only inside the
reference-design archive, under its own licence; `fetch-dpu-trd.sh` downloads it
into `hardware/dpu-trd/`, which is git-ignored. No hash of the archive was
recorded when the measured bitstream was built; set `DPU_TRD_SHA256` to
enforce one.

## This is the bitstream that was measured

Identified by content, not by name or date:

- `u96v2_sbc_base_wrapper.xsa` is byte-identical to the hardware description the
  PetaLinux 2022.2 project imported.
- `firmware/u96v2_sbc_base_wrapper.bit.bin` is the file packaged into the image's
  root filesystem at `/lib/firmware/xilinx/base/` — the one `aspir-load-dpu`
  loads. After a 10,240-byte header, its configuration data is the XSA's
  bitstream byte for byte.
- The overlay loaded on the board reports fingerprint `0x101000056010400`
  (`xdputil query`), which every compiled model in `models/` targets.

## Rebuilding

```bash
# 1. DPU IP source (~130 MB from AMD).
bash hardware/dpu_overlay/fetch-dpu-trd.sh

# 2. Vivado 2022.2 project from the block-diagram TCL (seconds).
vivado -mode batch -source hardware/dpu_overlay/regenerate_project.tcl

# 3. Optional full build: synthesis, implementation, bitstream, XSA.
vivado -mode batch -source hardware/dpu_overlay/regenerate_project.tcl -tclargs --build

# 4. Runtime overlay from an XSA.
XILINX_DIR=/path/to/Xilinx bash hardware/dpu_overlay/gen_dpu_overlay.sh
```

`create_bd.tcl` refuses to run under a Vivado release other than 2022.2. A
rebuild produces a different bitstream — placement and routing vary — with the
same fingerprint, which is what binds the compiled models to the hardware.

## On the board

The overlay is loaded at runtime, not at boot, by the `aspir-load-dpu` helper
that the PetaLinux image installs (see `envs/`). The CPU runner calls
`aspir-unload-dpu` before measuring, so the CPU baseline is not taken with the
accelerator powered; `--no-unload-pl` keeps the overlay loaded, as in the
compute-plane control. The DPU runner reloads the overlay when `/dev/dpu` is
missing.
