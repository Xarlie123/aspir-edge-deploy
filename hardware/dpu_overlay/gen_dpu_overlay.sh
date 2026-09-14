#!/usr/bin/env bash
# gen_dpu_overlay.sh — generate pl.dtbo + <name>.bit.bin from a fixed XSA,
# standalone (xsct + dtc + bootgen), WITHOUT running petalinux-build.
#
# Use this to try different DPU configurations at runtime: build the XSA in
# Vivado, run this, copy the two outputs to the board, then:
#     fpgautil -R                                  # drop the current overlay
#     fpgautil -b <name>.bit.bin -o pl.dtbo        # load the new one
#
# Verified equivalent: the pl.dtbo this produces is byte-for-byte identical
# to the one petalinux-build emits for the same XSA.
#
# Usage:  ./gen_dpu_overlay.sh <design.xsa> [output_dir]
#
set -euo pipefail

XSA="${1:?usage: gen_dpu_overlay.sh <design.xsa> [output_dir]}"
OUTDIR="${2:-$(pwd)/dpu_overlay_out}"
XSA="$(readlink -f "$XSA")"

# --- toolchain -------------------------------------------------------------
# xsct + dtc + bootgen all live in the Vitis 2022.2 install.
XILINX_DIR="${XILINX_DIR:?set XILINX_DIR to the Xilinx tools install root}"
# shellcheck disable=SC1091
source "${XILINX_DIR}/Vitis/2022.2/settings64.sh"

# --- DTG scripts (device-tree-xlnx) ---------------------------------------
# One-time clone from the petalinux downloads mirror. SRCREV 24d29888d0 is
# what PetaLinux 2022.2 pins; using the same revision guarantees identical
# output to the petalinux flow.
DTG_REPO="${DTG_REPO:-${HOME}/.cache/device-tree-xlnx}"
DTG_SRCREV="24d29888d0"
DTG_MIRROR="${DTG_MIRROR:-$(dirname "$0")/../../projects/ultra96v2/build/downloads/git2/github.com.Xilinx.device-tree-xlnx.git}"
if [ ! -d "${DTG_REPO}/.git" ]; then
    echo "[*] cloning device-tree-xlnx -> ${DTG_REPO}"
    git clone -q "${DTG_MIRROR}" "${DTG_REPO}"
fi
git -C "${DTG_REPO}" checkout -q "${DTG_SRCREV}"

mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

# --- 1. XSA -> overlay-form device tree sources (xsct / hsi) ---------------
cat > _gen_dtbo.tcl <<'EOF'
hsi::open_hw_design [lindex $argv 0]
hsi::set_repo_path  [lindex $argv 2]
hsi::create_sw_design dt -os device_tree -proc psu_cortexa53_0
hsi::set_property CONFIG.dt_overlay true [hsi::get_os]
hsi::generate_target -dir [lindex $argv 1]
hsi::close_hw_design [hsi::current_hw_design]
EOF
echo "[*] xsct: generating overlay device-tree from XSA"
xsct -sdx -nodisp _gen_dtbo.tcl "${XSA}" "${OUTDIR}" "${DTG_REPO}"

# --- 2. pl.dtsi -> pl.dtbo (dtc, overlay mode) -----------------------------
echo "[*] dtc: compiling pl.dtsi -> pl.dtbo"
dtc -@ -I dts -O dtb -o pl.dtbo pl.dtsi

# --- 3. bitstream inside the XSA -> <name>.bit.bin (bootgen) ---------------
# The XSA is a zip; pull the .bit out and convert to the FPGA-manager format.
BIT_NAME="$(unzip -Z1 "${XSA}" '*.bit' | head -1)"
unzip -o -q "${XSA}" "${BIT_NAME}" -d .
cat > _bit.bif <<EOF
all:
{
	${BIT_NAME}
}
EOF
echo "[*] bootgen: ${BIT_NAME} -> ${BIT_NAME}.bin"
bootgen -arch zynqmp -image _bit.bif -process_bitstream bin -w on >/dev/null

rm -f _gen_dtbo.tcl _bit.bif
echo
echo "[done] outputs in ${OUTDIR}:"
echo "  pl.dtbo            $(stat -c%s pl.dtbo) bytes"
echo "  ${BIT_NAME}.bin    $(stat -c%s "${BIT_NAME}.bin") bytes"
echo
echo "On the board:"
echo "  fpgautil -R"
echo "  fpgautil -b ${BIT_NAME}.bin -o pl.dtbo"
