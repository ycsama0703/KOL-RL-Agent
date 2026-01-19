#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
BUFFER_ROOT=${BUFFER_ROOT:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/ablation/local/bc_plus_vanilla_iql}
TEST_DIR=${TEST_DIR:-outputs/ablation/test/bc_plus_vanilla_iql}
DEVICE=${DEVICE:-cpu}
BC_BATCH_SIZE=${BC_BATCH_SIZE:-256}
IQL_BATCH_SIZE=${IQL_BATCH_SIZE:-256}

if [ -z "$BUFFER_ROOT" ]; then
  if [ -d "data/buffer_22-24_end1231" ]; then
    BUFFER_ROOT="data/buffer_22-24_end1231"
  else
    BUFFER_ROOT="data/buffer_22-24"
  fi
fi

for d in "$BUFFER_ROOT"/*/; do
  kol=$(basename "$d")
  "$PYTHON" ablation_study/04_component_ablation/train_vanilla_iql.py \
    --kol "$kol" \
    --replay-dir "$BUFFER_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --bc-batch-size "$BC_BATCH_SIZE" \
    --iql-batch-size "$IQL_BATCH_SIZE"
done

TRAIN_DIR="$OUTPUT_DIR" \
BUFFER_ROOT="$BUFFER_ROOT" \
TEST_DIR="$TEST_DIR" \
DEVICE="$DEVICE" \
bash ablation_study/04_component_ablation/batch_test_vanilla.sh
