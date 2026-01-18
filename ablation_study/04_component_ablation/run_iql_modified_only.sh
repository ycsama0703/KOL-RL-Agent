#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
BUFFER_ROOT=${BUFFER_ROOT:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/ablation/components/iql_modified_only}
TEST_DIR=${TEST_DIR:-outputs/ablation/components/iql_modified_only_test}
DEVICE=${DEVICE:-cpu}
EXPORT_SIGNAL=${EXPORT_SIGNAL:-1}

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
    --bc-epochs 0
done

TRAIN_DIR="$OUTPUT_DIR" \
BUFFER_ROOT="$BUFFER_ROOT" \
TEST_DIR="$TEST_DIR" \
DEVICE="$DEVICE" \
REWARD_ROOT="$REWARD_ROOT" \
EXPORT_SIGNAL="$EXPORT_SIGNAL" \
bash scripts/batch_test_and_plot.sh
