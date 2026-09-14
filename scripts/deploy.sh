#!/bin/sh
# Sync the ASPIR test pipeline to the Ultra96-v2.
#
# Usage:   ./scripts/deploy.sh [user@host]
# The board address is required.
#
# Optional: EXPERIMENTS="id1 id2" ./scripts/deploy.sh
#           restricts the model sync to those experiment ids. By default every
#           experiment under models/ that has a source/ directory is sent.
#
# Phase C (DPU Vivado-flow): the bitstream lives in the PetaLinux rootfs and
# is loaded at runtime via `aspir-load-dpu`. No xclbin / /boot copy needed.
# Route A (NEON FP32) just uses the ONNX export from models/.../source/.
#
# The script also writes scripts/deploy_info.json on the board, recording the
# host commit this deployment came from. The board has no git checkout, so
# that file is what makes every run report on the board traceable to code.

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOARD="${1:?usage: scripts/deploy.sh user@board-address}"
BOARD_USER="${BOARD%@*}"
DEST="/home/${BOARD_USER}/aspir"

echo "[deploy] target: $BOARD:$DEST"

# --- code identity stamp ---------------------------------------------------
GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [ -n "$(cd "$REPO_ROOT" && git status --porcelain 2>/dev/null)" ]; then
    GIT_DIRTY=true
else
    GIT_DIRTY=false
fi
STAMP_DIR="$(mktemp -d)"
cat > "$STAMP_DIR/deploy_info.json" <<JSON
{
  "git_sha": "$GIT_SHA",
  "git_branch": "$GIT_BRANCH",
  "dirty": $GIT_DIRTY,
  "deployed_at": "$(date -Iseconds)",
  "deployed_from": "$(hostname)",
  "repo": "$(basename "$REPO_ROOT")"
}
JSON
echo "[deploy] code: $GIT_SHA ($GIT_BRANCH, dirty=$GIT_DIRTY)"

ssh -o StrictHostKeyChecking=accept-new "$BOARD" "mkdir -p $DEST/src $DEST/scripts $DEST/models $DEST/results"

# --- aspir_unet package + scripts + sweep matrices --------------------------
rsync -avz --delete \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
    "$REPO_ROOT/src/aspir_unet"   "$BOARD:$DEST/src/"
rsync -avz --delete \
    --exclude '__pycache__' --exclude '*.pyc' --exclude 'deploy_info.json' \
    "$REPO_ROOT/scripts"          "$BOARD:$DEST/"
if [ -d "$REPO_ROOT/sweeps" ]; then
    rsync -avz --delete "$REPO_ROOT/sweeps" "$BOARD:$DEST/"
fi
rsync -avz "$STAMP_DIR/deploy_info.json" "$BOARD:$DEST/scripts/deploy_info.json"
rm -rf "$STAMP_DIR"

# --- experiments ------------------------------------------------------------
# Every experiment directory travels whole except for host-side scratch and
# board-side artefacts: quantizer_scratch/ is quantiser debug output, while
# results/, jetson_trt/ and the INT8 ONNX exports are produced on the board and
# must survive a redeploy. A TensorRT engine belongs to the GPU that built it;
# the INT8 export is written by the same ONNX Runtime version that will read it,
# which is the board's, not the workstation's. Deleting either costs a rebuild
# on every single sync.
if [ -z "${EXPERIMENTS:-}" ]; then
    EXPERIMENTS="$(cd "$REPO_ROOT/models" && \
        for d in */; do [ -d "${d}source" ] && printf '%s ' "${d%/}"; done)"
fi
echo "[deploy] experiments: $EXPERIMENTS"

for M in $EXPERIMENTS; do
    if [ ! -d "$REPO_ROOT/models/$M" ]; then
        echo "[deploy] WARNING unknown experiment '$M', skipping"
        continue
    fi
    rsync -avzL --delete \
        --exclude '__pycache__' \
        --exclude 'quantizer_scratch' \
        --exclude 'results' \
        --exclude 'jetson_trt' \
        --exclude '*_int8.onnx' --exclude '*.quant.json' \
        "$REPO_ROOT/models/$M" "$BOARD:$DEST/models/"
done

cat <<EOF

[deploy] done. On the board (XLNX_ENABLE_FINGERPRINT_CHECK=0 is already set
[deploy] via /etc/environment):

  aspir-load-dpu                                  # once per boot, for route C

  # single runs — artefacts resolved from the experiment id
  python3 $DEST/scripts/run_cpu_ort.py  --list-experiments
  python3 $DEST/scripts/run_cpu_ort.py  --experiment <id> --limit 30
  python3 $DEST/scripts/run_dpu_int8.py   --experiment <id> --energy-mode
  python3 $DEST/scripts/run_recon_timing.py  --experiment <id>

  # a whole configuration matrix, resumable
  python3 $DEST/scripts/run_sweep.py $DEST/sweeps/<matrix>.json --dry-run
  python3 $DEST/scripts/run_sweep.py $DEST/sweeps/<matrix>.json

  # on a Jetson: build the engine once per experiment and precision, then run
  python3 $DEST/scripts/build_trt_engine.py --experiment <id> --precision int8
  python3 $DEST/scripts/run_gpu_trt.py     --experiment <id> --precision int8

  # reports land in $DEST/results/<experiment>/ ; pull them back with
  #   rsync -avz $BOARD:$DEST/results/ models/<id>/results/
EOF
