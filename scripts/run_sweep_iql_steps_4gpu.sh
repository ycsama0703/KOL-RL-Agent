#!/usr/bin/env bash
set -euo pipefail

BUFFER_ROOT=${BUFFER_ROOT:-data/replay_buffer/22-24}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/sweep/iql_steps}
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
  i=0
  for kol in "${KOLS[@]}"; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$gpu python train.py \
      --kol "$kol" \
      --replay-dir "$BUFFER_ROOT" \
      --output-dir "$OUTPUT_ROOT/steps_${steps}" \
      --iql-steps "$steps" &

    ((i++))
    if (( i % ${#GPUS[@]} == 0 )); then
      wait
    fi
  done
  wait
done
