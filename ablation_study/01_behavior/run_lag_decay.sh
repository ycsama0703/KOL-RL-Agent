#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
REWARD_DIR=${REWARD_DIR:-data/processed/reward_end1231}
REWARD_ROOT=${REWARD_ROOT:-$REWARD_DIR}
BUFFER_ROOT=${BUFFER_ROOT:-ablation_study/01_behavior/buffers/lag_decay}
TICKER_VOCAB=${TICKER_VOCAB:-models/embedding/22-24_ticker_vocab.json}
TICKER_EMB=${TICKER_EMB:-models/embedding/22-24_ticker_embedding.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/ablation/behavior_lag_decay}
TEST_DIR=${TEST_DIR:-outputs/ablation/behavior_lag_decay_test}
DEVICE=${DEVICE:-cpu}
EXPORT_SIGNAL=${EXPORT_SIGNAL:-1}
SKIP_BUILD=${SKIP_BUILD:-0}

if [ "$SKIP_BUILD" != "1" ]; then
  "$PYTHON" scripts/build_replay_buffer.py \
    --reward-dir "$REWARD_DIR" \
    --output-dir "$BUFFER_ROOT" \
    --ticker-vocab "$TICKER_VOCAB" \
    --ticker-embedding "$TICKER_EMB" \
    --behavior-alpha 0.3 \
    --behavior-decay 0.2 \
    --behavior-entry-threshold 0.001
fi

for d in "$BUFFER_ROOT"/*/; do
  kol=$(basename "$d")
  "$PYTHON" train.py \
    --kol "$kol" \
    --replay-dir "$BUFFER_ROOT" \
    --output-dir "$OUTPUT_DIR"
done

TRAIN_DIR="$OUTPUT_DIR" \
BUFFER_ROOT="$BUFFER_ROOT" \
TEST_DIR="$TEST_DIR" \
DEVICE="$DEVICE" \
REWARD_ROOT="$REWARD_ROOT" \
EXPORT_SIGNAL="$EXPORT_SIGNAL" \
bash scripts/batch_test_and_plot.sh
