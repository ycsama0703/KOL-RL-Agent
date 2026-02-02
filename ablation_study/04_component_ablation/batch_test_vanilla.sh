#!/usr/bin/env bash
set -u

TRAIN_DIR=${TRAIN_DIR:-outputs/ablation/local/iql_vanilla_only}
TEST_DIR=${TEST_DIR:-outputs/ablation/test/iql_vanilla_only}
BUFFER_ROOT=${BUFFER_ROOT:-}
DEVICE=${DEVICE:-cpu}
ACTION_THRESHOLD=${ACTION_THRESHOLD:-0.02}
DAILY_PRICE_UPDATE=${DAILY_PRICE_UPDATE:-1}

if [ -z "$BUFFER_ROOT" ]; then
  if [ -d "data/buffer_22-24_end1231" ]; then
    BUFFER_ROOT="data/buffer_22-24_end1231"
  else
    BUFFER_ROOT="data/buffer_22-24"
  fi
fi

export MPLBACKEND=Agg

for d in "$BUFFER_ROOT"/*/; do
  kol=$(basename "$d")
  run=$(ls -td "$TRAIN_DIR/${kol}_"* 2>/dev/null | head -n1)
  if [ -z "$run" ]; then
    echo "Skip $kol (no run found)"
    continue
  fi

  run_name=$(basename "$run")
  out_dir="$TEST_DIR/$run_name"
  event_dir="$out_dir/event"
  daily_dir="$out_dir/daily"
  mkdir -p "$event_dir" "$daily_dir"

  ckpt="$run/checkpoints/policy.pt"
  buffer="$BUFFER_ROOT/$kol/test.pt"
  if [ ! -f "$ckpt" ] || [ ! -f "$buffer" ]; then
    echo "Skip $kol (missing checkpoint or test.pt)"
    continue
  fi

  python ablation_study/04_component_ablation/evaluate_vanilla_run.py \
    --checkpoint "$ckpt" \
    --buffer "$buffer" \
    --device "$DEVICE" \
    --output-dir "$event_dir" \
    --daily-output-dir "$daily_dir" \
    --action-threshold "$ACTION_THRESHOLD" \
    $([ "$DAILY_PRICE_UPDATE" = "1" ] && echo "--daily-price-update") \
    --plot || {
      echo "Eval failed for $kol"
      continue
    }
done
