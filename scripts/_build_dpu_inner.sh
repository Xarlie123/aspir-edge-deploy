#!/bin/bash
# Inner half of build_dpu_xmodels.sh — runs *inside* the Vitis AI container.
# Kept as its own file rather than embedded in a quoted docker argument: the
# quoting was fragile and an unreadable one-liner is a bad place for the step
# that produces every model the campaign deploys.
#
# Expects EXPERIMENTS, FORCE and ARCH_JSON in the environment.

# No `set -u` here on purpose: conda's activation scripts reference unbound
# variables, so under `set -u` `conda activate` terminates the shell silently
# and the build appears to do nothing at all.
set +u

source /opt/vitis_ai/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate vitis-ai-pytorch 2>/dev/null || true

: "${EXPERIMENTS:?set EXPERIMENTS in the environment}"
: "${FORCE:=0}"
: "${ARCH_JSON:?set ARCH_JSON in the environment}"

for E in $EXPERIMENTS; do
    DIR=/workspace/models/$E
    OUT=$DIR/phase_c_dpu
    FINAL=$OUT/${E}_b512_u96v2.xmodel
    LOG=$OUT/build_dpu.log

    if [ -f "$FINAL" ] && [ "$FORCE" = "0" ]; then
        echo "[skip]  $E (xmodel exists)"
        continue
    fi

    mkdir -p "$OUT"
    echo "[build] $E"
    {
        echo "=== $(date -Iseconds) quantise $E ==="
        python3 /workspace/scripts/quantize_dpu_int8.py --model-dir "$DIR"
        echo "=== $(date -Iseconds) compile $E ==="
        vai_c_xir -x "$OUT/quantizer_scratch/UNet_int.xmodel" \
                  -a "$ARCH_JSON" -o "$OUT" -n "${E}_b512_u96v2"
    } > "$LOG" 2>&1

    if [ -f "$FINAL" ]; then
        SUB=$(grep -o "DPU subgraph number [0-9]*" "$LOG" | tail -1)
        echo "[done]  $E  ($SUB)"
    else
        echo "[FAIL]  $E - see $LOG"
        tail -5 "$LOG"
    fi
done
echo "[dpu-build] finished"
