# Route C — switch VART from vitis-flow (XRT/zocl backend) to vivado-flow
# (DNNDK backend). The upstream vart_3.0.bb adds `vitis` to
# PACKAGECONFIG by default and is explicitly designed to allow this
# override: see the comment in the recipe — "Please remove this line if
# use vivado-flow".
#
# With this override, vart links against the DPU CHARDEV interface
# (`/dev/dpu`, exposed by CONFIG_XILINX_APF=y which the Avnet kernel
# already enables) instead of the XRT/zocl xclbin path. This is the only
# way to talk to a DPU instantiated directly in the Vivado BD at
# 0x8F000000 — the v++ flow (which works on top of an xclbin) does
# not produce a functional DPU on this XCZU3EG + DPUCZDX8G v4.1.0 combo
# (see project_dpu_fingerprint_clk_dsp_ce.md for the diagnosis).
#
# Keep this bbappend isolated from the symlinked Vitis-AI submodule
# recipes-vitis-ai/ tree so the submodule stays clean for git.
PACKAGECONFIG:remove = "vitis"
