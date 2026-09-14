#!/bin/sh
# Quantise and compile every experiment for the DPU, in one container session.
#
#   ./scripts/build_dpu_xmodels.sh                 # all experiments, skip done
#   ./scripts/build_dpu_xmodels.sh --force         # rebuild everything
#   ./scripts/build_dpu_xmodels.sh <id> [<id> ...] # only these
#
# Two tools, both from Vitis AI 3.0, both host-side:
#
#   vai_q_pytorch  (via scripts/quantize_dpu_int8.py) traces the live
#                  PyTorch module on the experiment's calibration frames,
#                  collects per-tensor activation ranges and emits a quantised
#                  XIR graph.
#   vai_c_xir      compiles that graph against arch.json — the machine
#                  description of the accelerator actually built into the
#                  bitstream — producing the .xmodel the board executes.
#
# The compiler embeds a fingerprint of the target architecture; a model
# compiled against a different arch.json is refused at load time. That is why
# arch.json lives with the hardware project and not with the model.
#
# Everything runs in the pinned Vitis AI container, so the result does not
# depend on what the workstation happens to have installed.

set -eu

# models/<id>/source/ is a symlink into aspir-edge-data (scripts/link_data.sh);
# the container needs that checkout mounted at the same path to follow it.
: "${ASPIR_EDGE_DATA:?set ASPIR_EDGE_DATA to a checkout of aspir-edge-data}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106"
ARCH_JSON="/workspace/hardware/dpu_overlay/arch.json"

FORCE=0
EXPERIMENTS=""
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *)       EXPERIMENTS="$EXPERIMENTS $arg" ;;
    esac
done

if [ -z "$EXPERIMENTS" ]; then
    # Every experiment that has a checkpoint and a calibration set. Calibration
    # is required, not optional: quantising against the evaluation frames would
    # produce numbers that cannot go in the paper.
    EXPERIMENTS="$(cd "$REPO_ROOT/models" && for d in */; do
        [ -n "$(ls "${d}source"/*.pt 2>/dev/null)" ] || continue
        [ -f "${d}source/calibration_images.npz" ] || continue
        printf '%s ' "${d%/}"
    done)"
fi

echo "[dpu-build] image:       $IMAGE"
echo "[dpu-build] arch:        $ARCH_JSON"
echo "[dpu-build] experiments: $EXPERIMENTS"

docker run --rm \
    -v "$REPO_ROOT":/workspace -w /workspace \
    -v "$ASPIR_EDGE_DATA":"$ASPIR_EDGE_DATA":ro \
    -u "$(id -u):$(id -g)" --shm-size=8gb \
    -e EXPERIMENTS="$EXPERIMENTS" -e FORCE="$FORCE" -e ARCH_JSON="$ARCH_JSON" \
    -e ASPIR_VITIS_IMAGE="$IMAGE" \
    "$IMAGE" bash /workspace/scripts/_build_dpu_inner.sh
