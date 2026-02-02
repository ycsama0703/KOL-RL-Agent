#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
BUFFER_ROOT=${BUFFER_ROOT:-data/buffer_22-24}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/sweep_iql_steps}
BC_BATCH_SIZE=${BC_BATCH_SIZE:-256}
IQL_BATCH_SIZE=${IQL_BATCH_SIZE:-256}
SKIP_200K=${SKIP_200K:-1}
IQL_STEPS_LIST=${IQL_STEPS_LIST:-"50000 100000 200000 300000 500000"}

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

for step in $IQL_STEPS_LIST; do
  if [ "$SKIP_200K" = "1" ] && [ "$step" = "200000" ]; then
    echo "Skip iql_steps=$step (SKIP_200K=1)"
    continue
  fi
  out_dir="$OUTPUT_ROOT/steps_${step}"
  mkdir -p "$out_dir"
  for kol in "${KOL_LIST[@]}"; do
    "$PYTHON" train.py \
      --kol "$kol" \
      --replay-dir "$BUFFER_ROOT" \
      --output-dir "$out_dir" \
      --iql-steps "$step" \
      --bc-batch-size "$BC_BATCH_SIZE" \
      --iql-batch-size "$IQL_BATCH_SIZE"
  done
done
