#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PYTHON=${PYTHON:-python}
BUFFER_ROOT=${BUFFER_ROOT:-data/multisource_ready_22-25/08_replay_buffer}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/benchmarks/generic_rl/cql}
CQL_BATCH_SIZE=${CQL_BATCH_SIZE:-256}
CQL_STEPS=${CQL_STEPS:-200000}
CQL_ALPHA=${CQL_ALPHA:-1.0}
CQL_TEMP=${CQL_TEMP:-1.0}
CQL_N_ACTIONS=${CQL_N_ACTIONS:-10}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-logs/$(basename "$OUTPUT_ROOT")}
MAX_JOBS=${MAX_JOBS:-8}

if [ ! -d "$BUFFER_ROOT" ]; then
  echo "BUFFER_ROOT not found: $BUFFER_ROOT" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"

TASKS=()
while IFS= read -r group_dir; do
  rel="${group_dir#$BUFFER_ROOT/}"
  source="${rel%%/*}"
  kol="${rel#*/}"
  TASKS+=("${source}|${kol}")
done < <(find "$BUFFER_ROOT" -mindepth 2 -maxdepth 2 -type d | sort)

if [ "${#TASKS[@]}" -eq 0 ]; then
  echo "No source/KOL folders found under $BUFFER_ROOT" >&2
  exit 1
fi

i=0
for task in "${TASKS[@]}"; do
  source="${task%%|*}"
  kol="${task#*|}"
  rel="${source}/${kol}"
  safe_name="${source}_${kol}_${RUN_TAG}"
  log_file="$LOG_ROOT/${safe_name}.log"
  out_dir="${OUTPUT_ROOT}/${source}"
  mkdir -p "$out_dir"

  echo "Launch CQL benchmark for $rel -> $log_file"
  nohup "$PYTHON" benchmarks/01_generic_rl/train_cql.py \
    --kol "$rel" \
    --replay-dir "$BUFFER_ROOT" \
    --output-dir "$out_dir" \
    --batch-size "$CQL_BATCH_SIZE" \
    --steps "$CQL_STEPS" \
    --cql-alpha "$CQL_ALPHA" \
    --cql-temp "$CQL_TEMP" \
    --cql-n-actions "$CQL_N_ACTIONS" \
    --no-progress-bar \
    >"$log_file" 2>&1 &

  ((i += 1))
  while (( $(jobs -pr | wc -l | tr -d ' ') >= MAX_JOBS )); do
    sleep 1
  done
done

wait
