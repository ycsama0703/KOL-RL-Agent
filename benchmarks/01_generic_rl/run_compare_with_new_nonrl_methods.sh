#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}

BUFFER_ROOT=${BUFFER_ROOT:-data/multisource_ready_22-25/08_replay_buffer}
OURS_ROOT=${OURS_ROOT:-outputs/multisource_test_mainline}
OURS_NAME=${OURS_NAME:-KICL}

# Existing benchmark roots
BC_ROOT=${BC_ROOT:-outputs/benchmarks/generic_rl/bc_mainline_test}
IQL_ROOT=${IQL_ROOT:-outputs/benchmarks/generic_rl/iql_mainline_test}
TD3BC_ROOT=${TD3BC_ROOT:-outputs/benchmarks/generic_rl/td3bc_mainline_test}
CQL_ROOT=${CQL_ROOT:-outputs/benchmarks/generic_rl/cql_mainline_test}
AWAC_ROOT=${AWAC_ROOT:-outputs/benchmarks/generic_rl/awac_mainline_test}

# New non-RL-ish method roots
RMB_ROOT=${RMB_ROOT:-outputs/benchmarks/generic_rl/risk_managed_baseline_test}
HAP_ROOT=${HAP_ROOT:-outputs/benchmarks/generic_rl/heuristic_allocation_test}
SUPDELTA_ROOT=${SUPDELTA_ROOT:-outputs/benchmarks/generic_rl/supervised_delta_test}

OUTPUT_DIR=${OUTPUT_DIR:-benchmarks/compare/with_new_nonrl}
MODE=${MODE:-intersection}
EVENT_CURVE_MODE=${EVENT_CURVE_MODE:-daily_mtm}

"$PYTHON" benchmarks/01_generic_rl/build_compare_report.py \
  --ours-root "$OURS_ROOT" \
  --ours-name "$OURS_NAME" \
  --method BC="$BC_ROOT" \
  --method IQL="$IQL_ROOT" \
  --method CQL="$CQL_ROOT" \
  --method TD3BC="$TD3BC_ROOT" \
  --method AWAC="$AWAC_ROOT" \
  --method RMB="$RMB_ROOT" \
  --method HAP="$HAP_ROOT" \
  --method SUP_DELTA="$SUPDELTA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --mode "$MODE" \
  --event-curve-mode "$EVENT_CURVE_MODE"
