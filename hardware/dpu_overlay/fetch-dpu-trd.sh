#!/bin/bash
# fetch-dpu-trd.sh — download and extract the DPUCZDX8G_VAI_v3.0 reference
# design from xilinx.com.
#
# Xilinx distributes the DPU IP source (xilinx.com:ip:dpuczdx8g:4.1) only
# inside this tarball — not via the Vitis-AI GitHub repo and not in the
# Vivado install. The Phase C Vivado project pins
#   hardware/dpu-trd/DPUCZDX8G_VAI_v3.0/dpu_ip
# as its ip_repo_paths, so the IP source must exist there before the
# project can be opened or rebuilt.
#
# Idempotent: skips the download if the extracted tree is already in place.
# Re-run any time with --force to refetch (e.g. corrupted archive).
#
# Usage:
#   bash hardware/dpu_overlay/fetch-dpu-trd.sh           # download + extract
#   bash hardware/dpu_overlay/fetch-dpu-trd.sh --force   # always refetch

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST_DIR="${REPO_ROOT}/hardware/dpu-trd"
TAR_NAME="DPUCZDX8G_VAI_v3.0.tar.gz"
URL="https://www.xilinx.com/bin/public/openDownload?filename=${TAR_NAME}"
# No expected hash is recorded: the archive behind the published bitstream was
# not hashed when it was downloaded. Set DPU_TRD_SHA256 to enforce one.
TAR_SHA256="${DPU_TRD_SHA256:-}"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"

if [ ${FORCE} -eq 0 ] && [ -d DPUCZDX8G_VAI_v3.0/dpu_ip/DPUCZDX8G ]; then
    echo "[fetch-dpu-trd] already extracted at ${DEST_DIR}/DPUCZDX8G_VAI_v3.0/, skipping."
    echo "[fetch-dpu-trd] use --force to refetch."
    exit 0
fi

if [ ${FORCE} -eq 1 ] || [ ! -f "${TAR_NAME}" ]; then
    echo "[fetch-dpu-trd] downloading ${TAR_NAME} (~130 MB) from xilinx.com …"
    rm -f "${TAR_NAME}"
    # -L follows redirects (the openDownload endpoint redirects to a CDN).
    curl -L -o "${TAR_NAME}" "${URL}"
fi

OBSERVED_SHA="$(sha256sum "${TAR_NAME}" | awk '{print $1}')"
echo "[fetch-dpu-trd] sha256: ${OBSERVED_SHA}"
if [ -n "${TAR_SHA256}" ] && [ "${OBSERVED_SHA}" != "${TAR_SHA256}" ]; then
    echo "[fetch-dpu-trd] WARNING: sha256 does not match the recorded value."
    echo "                expected: ${TAR_SHA256}"
    echo "                got:      ${OBSERVED_SHA}"
    exit 1
fi

echo "[fetch-dpu-trd] extracting …"
tar xzf "${TAR_NAME}"

echo
echo "[fetch-dpu-trd] done. Vivado picks up the IP from:"
echo "    ${DEST_DIR}/DPUCZDX8G_VAI_v3.0/dpu_ip"
echo "(this matches the ip_repo_paths recorded in the Phase C Vivado project)."
