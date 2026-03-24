#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Hard-constraint scope ablation launcher.
#
# 4 variants:
#   1) hard_both        : hard constraints in train + inference (reference)
#   2) hard_train_only  : hard constraints in train only
#   3) hard_infer_only  : hard constraints in inference only
#   4) hard_none        : no hard constraints in train/inference
#
# Modes:
#   MODE=train  -> launch training for all selected variants
#   MODE=test   -> launch testing for all selected variants
#   MODE=report -> build compare report (anchor = hard_both)
#   MODE=all    -> train + test + report

PYTHON=${PYTHON:-python}
MODE=${MODE:-all}

SOURCE_FILTER=${SOURCE_FILTER:-all}     # youtube | x | youtube,x | all
BUFFER_ROOT=${BUFFER_ROOT:-data/multisource_ready_22-25/08_replay_buffer}
KOL_LIST_FILE=${KOL_LIST_FILE:-}        # optional, each line: source/kol

TRAIN_ROOT_BASE=${TRAIN_ROOT_BASE:-outputs/ablation/hard_scope_train}
TEST_ROOT_BASE=${TEST_ROOT_BASE:-outputs/ablation/hard_scope_test}
LOG_ROOT_BASE=${LOG_ROOT_BASE:-logs/ablation_hard_scope}
REPORT_ROOT=${REPORT_ROOT:-benchmarks/compare/ablation_hard_scope}
FULL_TEST_ROOT=${FULL_TEST_ROOT:-$TEST_ROOT_BASE/hard_both}

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
MAX_JOBS=${MAX_JOBS:-8}

# training
BC_EPOCHS=${BC_EPOCHS:-10}
IQL_STEPS=${IQL_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-256}

# testing
DEVICE=${DEVICE:-cpu}
ACTION_THRESHOLD=${ACTION_THRESHOLD:-0.02}
DAILY_PRICE_UPDATE=${DAILY_PRICE_UPDATE:-1}

# report
EVENT_CURVE_MODE=${EVENT_CURVE_MODE:-daily_mtm}
PLOT_FORMAT=${PLOT_FORMAT:-png}

VARIANTS=(hard_both hard_train_only hard_infer_only hard_none)
VARIANTS_FILTER=${VARIANTS_FILTER:-}

usage() {
  cat <<'USAGE'
Usage:
  MODE=<train|test|report|all> bash benchmarks/01_generic_rl/run_ablation_hard_scope.sh

Key env vars:
  SOURCE_FILTER     youtube | x | youtube,x | all
  BUFFER_ROOT       replay buffer root
  KOL_LIST_FILE     optional allow-list file (one source/kol per line)
  TRAIN_ROOT_BASE   outputs for training runs
  TEST_ROOT_BASE    outputs for test results
  LOG_ROOT_BASE     logs root
  MAX_JOBS          concurrent background jobs
  DEVICE            evaluate_run device (cpu/cuda)
USAGE
}

source_allowed() {
  local src="$1"
  if [[ "$SOURCE_FILTER" == "all" ]]; then
    return 0
  fi
  IFS=',' read -r -a arr <<< "$SOURCE_FILTER"
  for s in "${arr[@]}"; do
    if [[ "$src" == "$s" ]]; then
      return 0
    fi
  done
  return 1
}

select_variants() {
  local -n selected_ref=$1
  selected_ref=()
  if [[ -z "$VARIANTS_FILTER" ]]; then
    selected_ref=("${VARIANTS[@]}")
    return
  fi

  IFS=',' read -r -a want <<< "$VARIANTS_FILTER"
  for w in "${want[@]}"; do
    local found=0
    for v in "${VARIANTS[@]}"; do
      if [[ "$w" == "$v" ]]; then
        selected_ref+=("$v")
        found=1
        break
      fi
    done
    if [[ "$found" -eq 0 ]]; then
      echo "Unknown variant in VARIANTS_FILTER: $w" >&2
      exit 1
    fi
  done
}

collect_tasks() {
  local -n tasks_ref=$1
  tasks_ref=()

  if [[ -n "$KOL_LIST_FILE" ]]; then
    if [[ ! -f "$KOL_LIST_FILE" ]]; then
      echo "KOL_LIST_FILE not found: $KOL_LIST_FILE" >&2
      exit 1
    fi
    while IFS= read -r rel || [[ -n "$rel" ]]; do
      rel="${rel%%#*}"
      rel="${rel%"${rel##*[![:space:]]}"}"
      rel="${rel#"${rel%%[![:space:]]*}"}"
      [[ -z "$rel" ]] && continue
      local src="${rel%%/*}"
      if ! source_allowed "$src"; then
        continue
      fi
      if [[ ! -d "$BUFFER_ROOT/$rel" ]]; then
        echo "Skip allowlist item (not in buffer): $rel" >&2
        continue
      fi
      tasks_ref+=("${src}|${rel#*/}")
    done < "$KOL_LIST_FILE"
  else
    while IFS= read -r group_dir; do
      local rel="${group_dir#$BUFFER_ROOT/}"
      local src="${rel%%/*}"
      local kol="${rel#*/}"
      if source_allowed "$src"; then
        tasks_ref+=("${src}|${kol}")
      fi
    done < <(find "$BUFFER_ROOT" -mindepth 2 -maxdepth 2 -type d | sort)
  fi
}

find_latest_run_dir() {
  local train_root="$1"
  local source_name="$2"
  local kol="$3"
  local nested_root="$train_root/$source_name/$source_name"
  local flat_root="$train_root/$source_name"
  local run=""

  if [[ -d "$nested_root" ]]; then
    run=$(find "$nested_root" -mindepth 1 -maxdepth 1 -type d -name "${kol}_*" | sort -r | head -n1 || true)
  fi
  if [[ -z "$run" && -d "$flat_root" ]]; then
    run=$(find "$flat_root" -mindepth 1 -maxdepth 1 -type d -name "${kol}_*" | sort -r | head -n1 || true)
  fi
  printf "%s" "$run"
}

variant_train_extra() {
  local variant="$1"
  case "$variant" in
    hard_both|hard_train_only) echo "" ;;
    hard_infer_only|hard_none) echo "--no-hard-intent-constraints" ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 1
      ;;
  esac
}

variant_test_hard_flag() {
  local variant="$1"
  case "$variant" in
    hard_both|hard_infer_only) echo "1" ;;
    hard_train_only|hard_none) echo "0" ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 1
      ;;
  esac
}

run_train() {
  [[ -d "$BUFFER_ROOT" ]] || { echo "BUFFER_ROOT not found: $BUFFER_ROOT" >&2; exit 1; }

  local tasks=()
  collect_tasks tasks
  [[ "${#tasks[@]}" -gt 0 ]] || { echo "No tasks found under $BUFFER_ROOT" >&2; exit 1; }

  local selected_variants=()
  select_variants selected_variants

  for variant in "${selected_variants[@]}"; do
    local variant_train_root="$TRAIN_ROOT_BASE/$variant"
    local variant_log_root="$LOG_ROOT_BASE/train/$variant"
    mkdir -p "$variant_train_root" "$variant_log_root"
    local extra
    extra=$(variant_train_extra "$variant")

    echo "== Train variant: $variant =="
    for task in "${tasks[@]}"; do
      local source_name="${task%%|*}"
      local kol="${task#*|}"
      local rel="${source_name}/${kol}"
      local out_dir="$variant_train_root/$source_name"
      local safe_name="${source_name}_${kol}_${RUN_TAG}"
      local log_file="$variant_log_root/${safe_name}.log"
      mkdir -p "$out_dir"

      cmd=(
        "$PYTHON" train.py
        --kol "$rel"
        --replay-dir "$BUFFER_ROOT"
        --output-dir "$out_dir"
        --bc-epochs "$BC_EPOCHS"
        --iql-steps "$IQL_STEPS"
        --bc-batch-size "$BATCH_SIZE"
        --iql-batch-size "$BATCH_SIZE"
        --no-progress-bar
      )
      if [[ -n "$extra" ]]; then
        # shellcheck disable=SC2206
        extra_arr=($extra)
        cmd+=("${extra_arr[@]}")
      fi

      echo "Launch train $variant $rel -> $log_file"
      nohup "${cmd[@]}" >"$log_file" 2>&1 &

      while (( $(jobs -pr | wc -l | tr -d ' ') >= MAX_JOBS )); do
        sleep 1
      done
    done
    wait
  done
}

run_test() {
  [[ -d "$BUFFER_ROOT" ]] || { echo "BUFFER_ROOT not found: $BUFFER_ROOT" >&2; exit 1; }

  local tasks=()
  collect_tasks tasks
  [[ "${#tasks[@]}" -gt 0 ]] || { echo "No tasks found under $BUFFER_ROOT" >&2; exit 1; }

  local selected_variants=()
  select_variants selected_variants

  for variant in "${selected_variants[@]}"; do
    local variant_train_root="$TRAIN_ROOT_BASE/$variant"
    local variant_test_root="$TEST_ROOT_BASE/$variant"
    local variant_log_root="$LOG_ROOT_BASE/test/$variant"
    local hard_flag
    hard_flag=$(variant_test_hard_flag "$variant")
    mkdir -p "$variant_test_root" "$variant_log_root"

    echo "== Test variant: $variant (hard=$hard_flag) =="
    for task in "${tasks[@]}"; do
      local source_name="${task%%|*}"
      local kol="${task#*|}"
      local rel="${source_name}/${kol}"

      local run
      run=$(find_latest_run_dir "$variant_train_root" "$source_name" "$kol")
      if [[ -z "$run" ]]; then
        echo "Skip $variant $rel (no run found)"
        continue
      fi

      local run_name
      run_name=$(basename "$run")
      local ckpt="$run/checkpoints/policy.pt"
      local buffer="$BUFFER_ROOT/$rel/test.pt"
      if [[ ! -f "$ckpt" || ! -f "$buffer" ]]; then
        echo "Skip $variant $rel (missing checkpoint or test.pt)"
        continue
      fi

      local out_dir="$variant_test_root/$source_name/$run_name"
      local event_dir="$out_dir/event"
      local daily_dir="$out_dir/daily"
      local safe_name="${source_name}_${kol}_${RUN_TAG}"
      local log_file="$variant_log_root/${safe_name}.log"
      mkdir -p "$event_dir" "$daily_dir"

      cmd=(
        "$PYTHON" scripts/evaluate_run.py
        --checkpoint "$ckpt"
        --buffer "$buffer"
        --device "$DEVICE"
        --output-dir "$event_dir"
        --daily-output-dir "$daily_dir"
        --action-threshold "$ACTION_THRESHOLD"
        --plot
      )
      if [[ "$DAILY_PRICE_UPDATE" == "1" ]]; then
        cmd+=(--daily-price-update)
      fi
      if [[ "$hard_flag" == "1" ]]; then
        cmd+=(--hard-intent-constraints)
      else
        cmd+=(--no-hard-intent-constraints)
      fi

      echo "Launch test $variant $rel -> $log_file"
      nohup "${cmd[@]}" >"$log_file" 2>&1 &

      while (( $(jobs -pr | wc -l | tr -d ' ') >= MAX_JOBS )); do
        sleep 1
      done
    done
    wait
  done
}

run_report() {
  [[ -d "$FULL_TEST_ROOT" ]] || { echo "FULL_TEST_ROOT not found: $FULL_TEST_ROOT" >&2; exit 1; }
  local report_suffix
  report_suffix=$(echo "$SOURCE_FILTER" | tr ',' '_')
  local out_dir="$REPORT_ROOT/$report_suffix"
  mkdir -p "$out_dir"

  cmd=(
    "$PYTHON" benchmarks/01_generic_rl/build_compare_report.py
    --ours-root "$FULL_TEST_ROOT"
    --ours-name HARD_BOTH
    --method TRAIN_ONLY="$TEST_ROOT_BASE/hard_train_only"
    --method INFER_ONLY="$TEST_ROOT_BASE/hard_infer_only"
    --method HARD_NONE="$TEST_ROOT_BASE/hard_none"
    --output-dir "$out_dir"
    --mode anchor_ours
    --event-curve-mode "$EVENT_CURVE_MODE"
    --plot-format "$PLOT_FORMAT"
    --no-include-baseline
    --highlight-method HARD_BOTH
  )

  echo "== Build report -> $out_dir =="
  "${cmd[@]}"
}

echo "MODE=$MODE SOURCE_FILTER=$SOURCE_FILTER RUN_TAG=$RUN_TAG MAX_JOBS=$MAX_JOBS"
echo "VARIANTS_FILTER=${VARIANTS_FILTER:-<all>}"
echo "BUFFER_ROOT=$BUFFER_ROOT"
echo "KOL_LIST_FILE=${KOL_LIST_FILE:-<none>}"
echo "TRAIN_ROOT_BASE=$TRAIN_ROOT_BASE"
echo "TEST_ROOT_BASE=$TEST_ROOT_BASE"
echo "REPORT_ROOT=$REPORT_ROOT"

case "$MODE" in
  train) run_train ;;
  test) run_test ;;
  report) run_report ;;
  all)
    run_train
    run_test
    run_report
    ;;
  help|-h|--help) usage ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    usage
    exit 1
    ;;
esac

