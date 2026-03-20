#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Unified launcher for KICL ablation study.
# Supports:
#   MODE=train   -> train all ablation variants
#   MODE=test    -> test all ablation variants
#   MODE=report  -> build compare report for all ablation variants
#   MODE=all     -> train + test + report
#
# Default scope is YouTube only.
#
# Example:
#   MODE=all \
#   SOURCE_FILTER=youtube \
#   BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
#   DEVICE=cuda \
#   MAX_JOBS=8 \
#   bash benchmarks/01_generic_rl/run_ablation_kicl.sh

PYTHON=${PYTHON:-python}
MODE=${MODE:-all}

# Source scope: "youtube", "x", "youtube,x", or "all"
SOURCE_FILTER=${SOURCE_FILTER:-youtube}

BUFFER_ROOT=${BUFFER_ROOT:-data/multisource_ready_22-25/08_replay_buffer}
TRAIN_ROOT_BASE=${TRAIN_ROOT_BASE:-outputs/ablation/kicl_train}
TEST_ROOT_BASE=${TEST_ROOT_BASE:-outputs/ablation/kicl_test}
REPORT_ROOT=${REPORT_ROOT:-benchmarks/ablation_compare}
LOG_ROOT_BASE=${LOG_ROOT_BASE:-logs/ablation_kicl}

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
MAX_JOBS=${MAX_JOBS:-8}

# Training
BC_EPOCHS=${BC_EPOCHS:-10}
IQL_STEPS=${IQL_STEPS:-200000}
BATCH_SIZE=${BATCH_SIZE:-256}

# Testing
DEVICE=${DEVICE:-cpu}
ACTION_THRESHOLD=${ACTION_THRESHOLD:-0.02}
DAILY_PRICE_UPDATE=${DAILY_PRICE_UPDATE:-1}

# Report
EVENT_CURVE_MODE=${EVENT_CURVE_MODE:-daily_mtm}
PLOT_FORMAT=${PLOT_FORMAT:-png}

VARIANTS=(
  "full"
  "w_no_hard"
  "w_no_soft"
  "w_no_bc_anchor"
  "w_no_rl_completion"
)
VARIANTS_FILTER=${VARIANTS_FILTER:-}

usage() {
  cat <<'USAGE'
Usage:
  MODE=<train|test|report|all> bash benchmarks/01_generic_rl/run_ablation_kicl.sh

Key env vars:
  SOURCE_FILTER     youtube | x | youtube,x | all    (default: youtube)
  BUFFER_ROOT       replay buffer root
  TRAIN_ROOT_BASE   ablation train output root
  TEST_ROOT_BASE    ablation test output root
  REPORT_ROOT       ablation compare output root
  LOG_ROOT_BASE     log root
  MAX_JOBS          concurrent jobs
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

collect_tasks() {
  local -n tasks_ref=$1
  tasks_ref=()
  while IFS= read -r group_dir; do
    rel="${group_dir#$BUFFER_ROOT/}"
    source_name="${rel%%/*}"
    kol="${rel#*/}"
    if source_allowed "$source_name"; then
      tasks_ref+=("${source_name}|${kol}")
    fi
  done < <(find "$BUFFER_ROOT" -mindepth 2 -maxdepth 2 -type d | sort)
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

variant_train_args() {
  local variant="$1"
  case "$variant" in
    full)
      echo ""
      ;;
    w_no_hard)
      echo "--no-hard-intent-constraints"
      ;;
    w_no_soft)
      echo "--bc-anchor-lambda 0.0 --fidelity-lambda 0.0 --actor-align-lambda 0.0 --entry-penalty-lambda 0.0 --reversal-penalty-lambda 0.0"
      ;;
    w_no_bc_anchor)
      echo "--bc-anchor-lambda 0.0"
      ;;
    w_no_rl_completion)
      echo "--iql-steps 0 --no-bc-fit-behavior --bc-anchor-lambda 0.0"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 1
      ;;
  esac
}

variant_test_hard_flag() {
  local variant="$1"
  case "$variant" in
    w_no_hard) echo "0" ;;
    *) echo "1" ;;
  esac
}

run_train() {
  if [[ ! -d "$BUFFER_ROOT" ]]; then
    echo "BUFFER_ROOT not found: $BUFFER_ROOT" >&2
    exit 1
  fi

  local tasks=()
  collect_tasks tasks
  if [[ "${#tasks[@]}" -eq 0 ]]; then
    echo "No tasks found for SOURCE_FILTER=$SOURCE_FILTER under $BUFFER_ROOT" >&2
    exit 1
  fi

  local selected_variants=()
  select_variants selected_variants
  for variant in "${selected_variants[@]}"; do
    local variant_train_root="$TRAIN_ROOT_BASE/$variant"
    local variant_log_root="$LOG_ROOT_BASE/train/$variant"
    mkdir -p "$variant_train_root" "$variant_log_root"
    local extra
    extra=$(variant_train_args "$variant")

    echo "== Train variant: $variant =="
    for task in "${tasks[@]}"; do
      source_name="${task%%|*}"
      kol="${task#*|}"
      rel="${source_name}/${kol}"
      safe_name="${source_name}_${kol}_${RUN_TAG}"
      log_file="$variant_log_root/${safe_name}.log"
      out_dir="$variant_train_root/$source_name"
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
  if [[ ! -d "$BUFFER_ROOT" ]]; then
    echo "BUFFER_ROOT not found: $BUFFER_ROOT" >&2
    exit 1
  fi

  local tasks=()
  collect_tasks tasks
  if [[ "${#tasks[@]}" -eq 0 ]]; then
    echo "No tasks found for SOURCE_FILTER=$SOURCE_FILTER under $BUFFER_ROOT" >&2
    exit 1
  fi

  local selected_variants=()
  select_variants selected_variants
  for variant in "${selected_variants[@]}"; do
    local hard_flag
    hard_flag=$(variant_test_hard_flag "$variant")
    local variant_train_root="$TRAIN_ROOT_BASE/$variant"
    local variant_test_root="$TEST_ROOT_BASE/$variant"
    local variant_log_root="$LOG_ROOT_BASE/test/$variant"
    mkdir -p "$variant_test_root" "$variant_log_root"

    echo "== Test variant: $variant (hard=$hard_flag) =="
    for task in "${tasks[@]}"; do
      source_name="${task%%|*}"
      kol="${task#*|}"
      rel="${source_name}/${kol}"

      run=$(find_latest_run_dir "$variant_train_root" "$source_name" "$kol")
      if [[ -z "$run" ]]; then
        echo "Skip $variant $rel (no run found)"
        continue
      fi
      run_name=$(basename "$run")
      out_dir="$variant_test_root/$source_name/$run_name"
      event_dir="$out_dir/event"
      daily_dir="$out_dir/daily"
      mkdir -p "$event_dir" "$daily_dir"

      ckpt="$run/checkpoints/policy.pt"
      buffer="$BUFFER_ROOT/$rel/test.pt"
      if [[ ! -f "$ckpt" || ! -f "$buffer" ]]; then
        echo "Skip $variant $rel (missing checkpoint or test.pt)"
        continue
      fi

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

      safe_name="${source_name}_${kol}_${RUN_TAG}"
      log_file="$variant_log_root/${safe_name}.log"
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
  local report_suffix
  report_suffix=$(echo "$SOURCE_FILTER" | tr ',' '_')
  local out_dir="$REPORT_ROOT/$report_suffix"
  mkdir -p "$out_dir"

  # Report currently expects all 5 variants to exist.
  cmd=(
    "$PYTHON" benchmarks/01_generic_rl/build_compare_report.py
    --ours-root "$TEST_ROOT_BASE/full"
    --ours-name FULL
    --method WO_HARD="$TEST_ROOT_BASE/w_no_hard"
    --method WO_SOFT="$TEST_ROOT_BASE/w_no_soft"
    --method WO_BC_ANCHOR="$TEST_ROOT_BASE/w_no_bc_anchor"
    --method WO_RL_COMPLETION="$TEST_ROOT_BASE/w_no_rl_completion"
    --output-dir "$out_dir"
    --mode anchor_ours
    --event-curve-mode "$EVENT_CURVE_MODE"
    --plot-format "$PLOT_FORMAT"
    --no-include-baseline
    --highlight-method FULL
  )

  echo "== Build report -> $out_dir =="
  "${cmd[@]}"
}

echo "MODE=$MODE SOURCE_FILTER=$SOURCE_FILTER MAX_JOBS=$MAX_JOBS RUN_TAG=$RUN_TAG"
echo "VARIANTS_FILTER=${VARIANTS_FILTER:-<all>}"
echo "BUFFER_ROOT=$BUFFER_ROOT"
echo "TRAIN_ROOT_BASE=$TRAIN_ROOT_BASE"
echo "TEST_ROOT_BASE=$TEST_ROOT_BASE"
echo "REPORT_ROOT=$REPORT_ROOT"

case "$MODE" in
  train)
    run_train
    ;;
  test)
    run_test
    ;;
  report)
    run_report
    ;;
  all)
    run_train
    run_test
    run_report
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    usage
    exit 1
    ;;
esac
