#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=${TRAIN_ROOT:-outputs/benchmarks/generic_rl/risk_managed_baseline}
BUFFER_ROOT=${BUFFER_ROOT:-data/multisource_ready_22-25/08_replay_buffer}
TEST_ROOT=${TEST_ROOT:-outputs/benchmarks/generic_rl/risk_managed_baseline_test}
LOG_ROOT=${LOG_ROOT:-logs/$(basename "$TEST_ROOT")}
DEVICE=${DEVICE:-cpu}
ACTION_THRESHOLD=${ACTION_THRESHOLD:-0.02}
DAILY_PRICE_UPDATE=${DAILY_PRICE_UPDATE:-1}
MAX_JOBS=${MAX_JOBS:-8}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

TRAIN_ROOT="$TRAIN_ROOT" \
BUFFER_ROOT="$BUFFER_ROOT" \
TEST_ROOT="$TEST_ROOT" \
LOG_ROOT="$LOG_ROOT" \
DEVICE="$DEVICE" \
ACTION_THRESHOLD="$ACTION_THRESHOLD" \
DAILY_PRICE_UPDATE="$DAILY_PRICE_UPDATE" \
HARD_INTENT_CONSTRAINTS=1 \
REGIME_SPLIT=1 \
ZERO_MARKET_FACTORS=0 \
MAX_JOBS="$MAX_JOBS" \
RUN_TAG="$RUN_TAG" \
bash benchmarks/01_generic_rl/batch_test_vanilla_multisource.sh

