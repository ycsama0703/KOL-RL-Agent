#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
BUFFER_ROOT=${BUFFER_ROOT:-data/buffer_22-24}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/sweep_expectile_beta}
IQL_STEPS=${IQL_STEPS:-200000}
BC_BATCH_SIZE=${BC_BATCH_SIZE:-256}
IQL_BATCH_SIZE=${IQL_BATCH_SIZE:-256}
EXPECTILES=${EXPECTILES:-"0.5 0.6"}
BETAS=${BETAS:-"5 8"}

if [ ! -d "$BUFFER_ROOT" ]; then
  echo "Buffer root not found: $BUFFER_ROOT"
  exit 1
fi

if [ -z "${KOLS:-}" ]; then
  KOL_LIST=()
  for d in "$BUFFER_ROOT"/*/; do
    KOL_LIST+=("$(basename "$d")")
  done
else
  IFS=',' read -ra KOL_LIST <<< "$KOLS"
fi

mkdir -p "$OUTPUT_ROOT"

for exp in $EXPECTILES; do
  for beta in $BETAS; do
    out_dir="$OUTPUT_ROOT/exp${exp}_beta${beta}"
    mkdir -p "$out_dir"
    echo "==> Sweep expectile=$exp beta=$beta"
    for kol in "${KOL_LIST[@]}"; do
      "$PYTHON" train.py \
        --kol "$kol" \
        --replay-dir "$BUFFER_ROOT" \
        --output-dir "$out_dir" \
        --iql-steps "$IQL_STEPS" \
        --expectile "$exp" \
        --temperature-beta "$beta" \
        --bc-batch-size "$BC_BATCH_SIZE" \
        --iql-batch-size "$IQL_BATCH_SIZE"
    done
  done
done
