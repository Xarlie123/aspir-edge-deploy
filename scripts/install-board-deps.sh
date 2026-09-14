#!/bin/sh
# Install the Python deps on the U96v2 board that PetaLinux doesn't ship
# in the rootfs (matplotlib for board-side plot generation, onnxruntime
# for Route A). Run as root or via the petalinux user's NOPASSWD sudo.
#
# Usage (from host):
#   ssh petalinux@<board> 'sudo sh -' < scripts/install-board-deps.sh
# or copy via SFTP and run directly:
#   sudo /home/petalinux/aspir/scripts/install-board-deps.sh
#
# This is idempotent: pip will skip already-installed packages of the
# matching version.
set -e
echo "=== install-board-deps.sh ==="

ONNX_VERSION="1.17.3"

# pip3 from PetaLinux's python3-pip package writes into /usr/lib/python3.X
# by default. That's fine; the rootfs is read-write and we want these
# available to all users, not just petalinux/root.
pip3 install --no-cache-dir \
    "numpy<2" \
    "onnxruntime==${ONNX_VERSION}" \
    "matplotlib>=3.5" \
    "scikit-image>=0.19"
# scripts/quantize_cpu_int8.py also needs the onnx package, which this script
# did not install during the campaign; it was added by hand as onnx 1.17.0
# (see envs/ultra96v2/requirements.txt for the exact environment).
pip3 install --no-cache-dir --only-binary=:all: "onnx==1.17.0"
# Pin numpy<2: onnxruntime 1.17.3 is built against the numpy 1.x C-API
# (_ARRAY_API symbol missing in numpy 2.x); installing matplotlib alone
# pulls in numpy>=1.23 which the resolver eagerly upgrades to 2.x. The
# pin keeps both happy.

echo "=== installed ==="
python3 -c "import onnxruntime, matplotlib, skimage; \
    print('onnxruntime', onnxruntime.__version__); \
    print('matplotlib', matplotlib.__version__); \
    print('skimage', skimage.__version__)"
