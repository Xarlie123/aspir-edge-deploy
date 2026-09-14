#!/bin/sh
# Link the per-experiment inputs of aspir-edge-data into models/.
#
# The runners resolve an experiment from models/<id>/source/ (checkpoint, ONNX,
# evaluation and calibration frames) and models/<id>/phase_c_dpu/ (compiled DPU
# model). This repository ships the second; aspir-edge-data ships the first.
#
#   ASPIR_EDGE_DATA=/path/to/aspir-edge-data scripts/link_data.sh
set -eu
: "${ASPIR_EDGE_DATA:?set ASPIR_EDGE_DATA to a checkout of aspir-edge-data}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$(cd "$ASPIR_EDGE_DATA" && pwd)"
for src in "$DATA"/datasets/*/source; do
    exp="$(basename "$(dirname "$src")")"
    mkdir -p "$REPO_ROOT/models/$exp"
    ln -sfn "$src" "$REPO_ROOT/models/$exp/source"
    echo "linked $exp"
done
