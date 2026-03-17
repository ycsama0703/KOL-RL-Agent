#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=${TRAIN_ROOT:-outputs/benchmarks/generic_rl/iql}
BUFFER_ROOT=${BUFFER_ROOT:-data/multisource_ready_22-25/08_replay_buffer}
TEST_ROOT=${TEST_ROOT:-outputs/benchmarks/generic_rl/iql_test}
DEVICE=${DEVICE:-cpu}
ACTION_THRESHOLD=${ACTION_THRESHOLD:-0.02}
DAILY_PRICE_UPDATE=${DAILY_PRICE_UPDATE:-1}

export MPLBACKEND=Agg

if [ ! -d "$TRAIN_ROOT" ]; then
  echo "TRAIN_ROOT not found: $TRAIN_ROOT" >&2
  exit 1
fi

if [ ! -d "$BUFFER_ROOT" ]; then
  echo "BUFFER_ROOT not found: $BUFFER_ROOT" >&2
  exit 1
fi

find "$BUFFER_ROOT" -mindepth 2 -maxdepth 2 -type d | sort | while read -r group_dir; do
  rel="${group_dir#$BUFFER_ROOT/}"
  source_name="${rel%%/*}"
  kol="${rel#*/}"

  run=$(ls -td "$TRAIN_ROOT/$source_name/${kol}_"* 2>/dev/null | head -n1 || true)
  if [ -z "$run" ]; then
    echo "Skip $rel (no run found)"
    continue
  fi

  run_name=$(basename "$run")
  out_dir="$TEST_ROOT/$source_name/$run_name"
  event_dir="$out_dir/event"
  daily_dir="$out_dir/daily"
  mkdir -p "$event_dir" "$daily_dir"

  ckpt="$run/checkpoints/policy.pt"
  buffer="$BUFFER_ROOT/$rel/test.pt"
  if [ ! -f "$ckpt" ] || [ ! -f "$buffer" ]; then
    echo "Skip $rel (missing checkpoint or test.pt)"
    continue
  fi

  cmd=(
    python ablation_study/04_component_ablation/evaluate_vanilla_run.py
    --checkpoint "$ckpt"
    --buffer "$buffer"
    --device "$DEVICE"
    --output-dir "$event_dir"
    --daily-output-dir "$daily_dir"
    --action-threshold "$ACTION_THRESHOLD"
    --plot
  )

  if [ "$DAILY_PRICE_UPDATE" = "1" ]; then
    cmd+=(--daily-price-update)
  fi

  echo "Test $rel -> $out_dir"
  "${cmd[@]}" || {
    echo "Eval failed for $rel"
    continue
  }
done
