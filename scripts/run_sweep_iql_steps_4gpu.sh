#!/usr/bin/env bash
set -euo pipefail

BUFFER_ROOT=${BUFFER_ROOT:-data/replay_buffer/22-24}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/sweep/iql_steps}
LOG_ROOT=${LOG_ROOT:-outputs/sweep/logs}
STEPS_LIST=(50000 100000)
GPUS=(0 1 2 3)

KOLS=()
for d in "$BUFFER_ROOT"/*/; do
  kol=$(basename "$d")
  if [[ "$kol" == "Everything_Money" ]]; then
    continue
  fi
  KOLS+=("$kol")
done

mkdir -p "$OUTPUT_ROOT"

for steps in "${STEPS_LIST[@]}"; do
  log_dir="$LOG_ROOT/steps_${steps}"
  mkdir -p "$log_dir"
  i=0
  for kol in "${KOLS[@]}"; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    ts=$(date +%Y%m%d_%H%M%S)
    log_file="$log_dir/${kol}_${ts}.log"
    CUDA_VISIBLE_DEVICES=$gpu python train.py \
      --kol "$kol" \
      --replay-dir "$BUFFER_ROOT" \
      --output-dir "$OUTPUT_ROOT/steps_${steps}" \
      --iql-steps "$steps" >"$log_file" 2>&1 &

    ((i++))
    if (( i % ${#GPUS[@]} == 0 )); then
      wait
    fi
  done
  wait
done
