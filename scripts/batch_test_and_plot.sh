#!/usr/bin/env bash
set -u

TRAIN_DIR=${TRAIN_DIR:-outputs/train_v2}
TEST_DIR=${TEST_DIR:-outputs/test_v2}
BUFFER_ROOT=${BUFFER_ROOT:-data/buffer_22-24}
DEVICE=${DEVICE:-cpu}
ACTION_THRESHOLD=${ACTION_THRESHOLD:-0.02}
BENCHMARK_TICKER=${BENCHMARK_TICKER:-SPY}
BENCHMARK_LABEL=${BENCHMARK_LABEL:-"SPY (market)"}
DAILY_BENCHMARK_TICKER=${DAILY_BENCHMARK_TICKER:-$BENCHMARK_TICKER}
DAILY_BENCHMARK_LABEL=${DAILY_BENCHMARK_LABEL:-$BENCHMARK_LABEL}
EXPORT_SIGNAL=${EXPORT_SIGNAL:-1}
DAILY_PRICE_UPDATE=${DAILY_PRICE_UPDATE:-1}
REWARD_ROOT=${REWARD_ROOT:-data/processed/reward}
TICKER_VOCAB=${TICKER_VOCAB:-models/embedding/22-24_ticker_vocab.json}
TICKER_EMB=${TICKER_EMB:-models/embedding/22-24_ticker_embedding.pt}
ENTRY_THRESHOLD=${ENTRY_THRESHOLD:-0.001}
CLAMP_DELTA=${CLAMP_DELTA:-1.0}

export KMP_DUPLICATE_LIB_OK=TRUE
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

  python ./scripts/evaluate_run.py \
    --checkpoint "$ckpt" \
    --buffer "$buffer" \
    --device "$DEVICE" \
    --output-dir "$event_dir" \
    --daily-output-dir "$daily_dir" \
    --daily-benchmark-ticker "$DAILY_BENCHMARK_TICKER" \
    --daily-benchmark-label "$DAILY_BENCHMARK_LABEL" \
    --action-threshold "$ACTION_THRESHOLD" \
    $([ "$DAILY_PRICE_UPDATE" = "1" ] && echo "--daily-price-update") || {
      echo "Eval failed for $kol"
      continue
    }

  if [ "$EXPORT_SIGNAL" = "1" ]; then
    reward_csv="$REWARD_ROOT/$kol/test.csv"
    if [ -f "$reward_csv" ]; then
      python ./scripts/export_signal_decisions.py \
        --checkpoint "$ckpt" \
        --reward-csv "$reward_csv" \
        --buffer "$buffer" \
        --vocab-path "$TICKER_VOCAB" \
        --embedding-path "$TICKER_EMB" \
        --output "$event_dir/signal_decisions_test.csv" \
        --entry-threshold "$ENTRY_THRESHOLD" \
        --clamp-delta "$CLAMP_DELTA" || echo "Signal export failed for $kol"
    else
      echo "Skip signal export for $kol (missing $reward_csv)"
    fi
  fi

  plot_cmd=(
    python ./scripts/plot_equity_curve.py
    --checkpoint "$ckpt"
    --buffer "$buffer"
    --output-figure "$event_dir/equity_test.png"
    --device "$DEVICE"
  )
  if [ -n "$BENCHMARK_TICKER" ]; then
    plot_cmd+=(--benchmark-ticker "$BENCHMARK_TICKER" --benchmark-label "$BENCHMARK_LABEL")
  fi
  "${plot_cmd[@]}" || echo "Plot failed for $kol"
done
