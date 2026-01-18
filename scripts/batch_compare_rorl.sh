#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="${TRAIN_DIR:-outputs/train_v2}"
BUFFER_ROOT="${BUFFER_ROOT:-}"
RORL_ROOT="${RORL_ROOT:-bencmarks/RORL}"
DEVICE="${DEVICE:-cuda}"
ALIGN="${ALIGN:-intersection}"
ACTION_THRESHOLD="${ACTION_THRESHOLD:-0.01}"

if [ -z "$BUFFER_ROOT" ]; then
  if [ -d "data/buffer_22-24_end1231" ]; then
    BUFFER_ROOT="data/buffer_22-24_end1231"
  else
    BUFFER_ROOT="data/buffer_22-24"
  fi
fi

if [ ! -d "$RORL_ROOT" ]; then
  echo "RORL root not found: $RORL_ROOT"
  exit 1
fi

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

for rorl_dir in "$RORL_ROOT"/*/; do
  kol="$(basename "$rorl_dir")"
  perf="$rorl_dir/performance.csv"
  if [ ! -f "$perf" ]; then
    echo "Skip $kol (missing performance.csv)"
    continue
  fi

  run="$(ls -td "$TRAIN_DIR"/${kol}_* 2>/dev/null | head -n1 || true)"
  if [ -z "$run" ]; then
    echo "Skip $kol (no run found)"
    continue
  fi

  ckpt="$run/checkpoints/policy.pt"
  buffer="$BUFFER_ROOT/$kol/test.pt"
  if [ ! -f "$ckpt" ] || [ ! -f "$buffer" ]; then
    echo "Skip $kol (missing checkpoint or buffer)"
    continue
  fi

  python scripts/compare_rorl.py \
    --checkpoint "$ckpt" \
    --buffer "$buffer" \
    --rorl-performance "$perf" \
    --align "$ALIGN" \
    --action-threshold "$ACTION_THRESHOLD" \
    --output-csv "$rorl_dir/compare_rorl.csv" \
    --output-figure "$rorl_dir/compare_rorl.png" || echo "Compare failed for $kol"
done
