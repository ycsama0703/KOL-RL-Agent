#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
BUFFER_ROOT=${BUFFER_ROOT:-data/multisource_ready_22-25/08_replay_buffer}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/benchmarks/generic_rl/iql}
BC_BATCH_SIZE=${BC_BATCH_SIZE:-256}
IQL_BATCH_SIZE=${IQL_BATCH_SIZE:-256}
LOG_ROOT=${LOG_ROOT:-$OUTPUT_ROOT/_batch_logs}

if [ ! -d "$BUFFER_ROOT" ]; then
  echo "BUFFER_ROOT not found: $BUFFER_ROOT" >&2
  exit 1
fi

find "$BUFFER_ROOT" -mindepth 2 -maxdepth 2 -type d | sort | while read -r group_dir; do
  rel="${group_dir#$BUFFER_ROOT/}"
  echo "Train vanilla IQL benchmark for $rel"
  mkdir -p "$LOG_ROOT"
  safe_name="${rel//\//__}"
  "$PYTHON" benchmarks/01_generic_rl/train_generic_rl.py \
    --method iql \
    --kol "$rel" \
    --replay-dir "$BUFFER_ROOT" \
    --output-dir "$OUTPUT_ROOT" \
    --bc-epochs 0 \
    --bc-batch-size "$BC_BATCH_SIZE" \
    --iql-batch-size "$IQL_BATCH_SIZE" \
    >"$LOG_ROOT/${safe_name}.log" 2>&1
done
