#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
BUFFER_ROOT=${BUFFER_ROOT:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/ablation/local/bc_only}
TEST_DIR=${TEST_DIR:-outputs/ablation/test/bc_only}
DEVICE=${DEVICE:-cpu}
EXPORT_SIGNAL=${EXPORT_SIGNAL:-1}
BC_BATCH_SIZE=${BC_BATCH_SIZE:-256}
IQL_BATCH_SIZE=${IQL_BATCH_SIZE:-256}

if [ -z "$BUFFER_ROOT" ]; then
  if [ -d "data/buffer_22-24_end1231" ]; then
    BUFFER_ROOT="data/buffer_22-24_end1231"
  else
    BUFFER_ROOT="data/buffer_22-24"
  fi
fi

if [ -z "${REWARD_ROOT:-}" ]; then
  if [ "$BUFFER_ROOT" = "data/buffer_22-24_end1231" ]; then
    REWARD_ROOT="data/processed/reward_end1231"
  else
    REWARD_ROOT="data/processed/reward"
  fi
fi

for d in "$BUFFER_ROOT"/*/; do
  kol=$(basename "$d")
  "$PYTHON" train.py \
    --kol "$kol" \
    --replay-dir "$BUFFER_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --iql-steps 0 \
    --bc-batch-size "$BC_BATCH_SIZE" \
    --iql-batch-size "$IQL_BATCH_SIZE"
done

TRAIN_DIR="$OUTPUT_DIR" \
BUFFER_ROOT="$BUFFER_ROOT" \
TEST_DIR="$TEST_DIR" \
DEVICE="$DEVICE" \
REWARD_ROOT="$REWARD_ROOT" \
EXPORT_SIGNAL="$EXPORT_SIGNAL" \
bash scripts/batch_test_and_plot.sh
