#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

TRAIN_ROOT=${TRAIN_ROOT:-outputs/benchmarks/generic_rl/iql}
BUFFER_ROOT=${BUFFER_ROOT:-data/multisource_ready_22-25/08_replay_buffer}
TEST_ROOT=${TEST_ROOT:-outputs/benchmarks/generic_rl/iql_test}
DEVICE=${DEVICE:-cpu}
ACTION_THRESHOLD=${ACTION_THRESHOLD:-0.02}
DAILY_PRICE_UPDATE=${DAILY_PRICE_UPDATE:-1}
HARD_INTENT_CONSTRAINTS=${HARD_INTENT_CONSTRAINTS:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-logs/$(basename "$TEST_ROOT")}
MAX_JOBS=${MAX_JOBS:-8}

export MPLBACKEND=Agg

if [ ! -d "$TRAIN_ROOT" ]; then
  echo "TRAIN_ROOT not found: $TRAIN_ROOT" >&2
  exit 1
fi

if [ ! -d "$BUFFER_ROOT" ]; then
  echo "BUFFER_ROOT not found: $BUFFER_ROOT" >&2
  exit 1
fi

find_latest_run_dir() {
  local train_root="$1"
  local source_name="$2"
  local kol="$3"
  local nested_root="$train_root/$source_name/$source_name"
  local flat_root="$train_root/$source_name"
  local run=""

  if [ -d "$nested_root" ]; then
    run=$(find "$nested_root" -mindepth 1 -maxdepth 1 -type d -name "${kol}_*" | sort -r | head -n1 || true)
  fi
  if [ -z "$run" ] && [ -d "$flat_root" ]; then
    run=$(find "$flat_root" -mindepth 1 -maxdepth 1 -type d -name "${kol}_*" | sort -r | head -n1 || true)
  fi

  printf "%s" "$run"
}

mkdir -p "$LOG_ROOT" "$TEST_ROOT"

TASKS=()
while IFS= read -r group_dir; do
  rel="${group_dir#$BUFFER_ROOT/}"
  source_name="${rel%%/*}"
  kol="${rel#*/}"
  TASKS+=("${source_name}|${kol}")
done < <(find "$BUFFER_ROOT" -mindepth 2 -maxdepth 2 -type d | sort)

if [ "${#TASKS[@]}" -eq 0 ]; then
  echo "No source/KOL folders found under $BUFFER_ROOT" >&2
  exit 1
fi

i=0
for task in "${TASKS[@]}"; do
  source_name="${task%%|*}"
  kol="${task#*|}"
  rel="${source_name}/${kol}"

  run=$(find_latest_run_dir "$TRAIN_ROOT" "$source_name" "$kol")
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
    python scripts/evaluate_run.py
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
  if [ "$HARD_INTENT_CONSTRAINTS" = "1" ]; then
    cmd+=(--hard-intent-constraints)
  else
    cmd+=(--no-hard-intent-constraints)
  fi

  safe_name="${source_name}_${kol}_${RUN_TAG}"
  log_file="$LOG_ROOT/${safe_name}.log"

  echo "Launch test $rel -> $out_dir (log: $log_file)"
  nohup "${cmd[@]}" >"$log_file" 2>&1 &

  ((i += 1))
  while (( $(jobs -pr | wc -l | tr -d ' ') >= MAX_JOBS )); do
    sleep 1
  done
done

wait
