#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
BUFFER_ROOT=${BUFFER_ROOT:-data/multisource_ready_22-25/08_replay_buffer}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/benchmarks/generic_rl/bc}
BC_BATCH_SIZE=${BC_BATCH_SIZE:-256}
BC_EPOCHS=${BC_EPOCHS:-10}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-logs/$(basename "$OUTPUT_ROOT")}
DEVICE=${DEVICE:-cuda}

if [ ! -d "$BUFFER_ROOT" ]; then
  echo "BUFFER_ROOT not found: $BUFFER_ROOT" >&2
  exit 1
fi

find "$BUFFER_ROOT" -mindepth 2 -maxdepth 2 -type d | sort | while read -r group_dir; do
  rel="${group_dir#$BUFFER_ROOT/}"
  echo "Train BC benchmark for $rel"
  mkdir -p "$LOG_ROOT"
  safe_name="${rel//\//_}_${RUN_TAG}"
  "$PYTHON" train.py \
    --kol "$rel" \
    --replay-dir "$BUFFER_ROOT" \
    --output-dir "$OUTPUT_ROOT" \
    --device "$DEVICE" \
    --bc-epochs "$BC_EPOCHS" \
    --bc-fit-behavior \
    --bc-anchor-lambda 0.0 \
    --iql-steps 0 \
    --no-hard-intent-constraints \
    --fidelity-lambda 0.0 \
    --actor-align-lambda 0.0 \
    --entry-penalty-lambda 0.0 \
    --reversal-penalty-lambda 0.0 \
    --bc-batch-size "$BC_BATCH_SIZE" \
    --iql-batch-size "$BC_BATCH_SIZE" \
    --no-progress-bar \
    >"$LOG_ROOT/${safe_name}.log" 2>&1
done
