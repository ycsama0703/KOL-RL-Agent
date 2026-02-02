#!/usr/bin/env bash
set -euo pipefail

BUFFER_ROOT=${BUFFER_ROOT:-data/buffer_22-24}
SWEEP_ROOT=${SWEEP_ROOT:-outputs/sweep_iql_steps}
TRAIN_200K=${TRAIN_200K:-outputs/train_v2}
TEST_ROOT=${TEST_ROOT:-outputs/sweep_iql_steps/test}
STEPS_LIST=${STEPS_LIST:-"50000 100000 200000 300000 500000"}
DEVICE=${DEVICE:-cpu}
EXPORT_SIGNAL=${EXPORT_SIGNAL:-0}

if [ ! -d "$BUFFER_ROOT" ]; then
  echo "Buffer root not found: $BUFFER_ROOT"
  exit 1
fi

mkdir -p "$TEST_ROOT"

for step in $STEPS_LIST; do
  if [ "$step" = "200000" ]; then
    train_dir="$TRAIN_200K"
  else
    train_dir="$SWEEP_ROOT/steps_${step}"
  fi
  test_dir="$TEST_ROOT/steps_${step}"
  if [ ! -d "$train_dir" ]; then
    echo "Skip iql_steps=$step (missing train dir: $train_dir)"
    continue
  fi
  TRAIN_DIR="$train_dir" \
  TEST_DIR="$test_dir" \
  BUFFER_ROOT="$BUFFER_ROOT" \
  DEVICE="$DEVICE" \
  EXPORT_SIGNAL="$EXPORT_SIGNAL" \
  bash scripts/batch_test_and_plot.sh
done

python - <<'PY'
import json
import os
import re
from pathlib import Path

import pandas as pd

use_daily = os.getenv("USE_DAILY", "0") == "1"
test_root = Path(os.getenv("TEST_ROOT", "outputs/sweep_iql_steps/test"))
pattern = re.compile(r"^(.*)_\d{8}_\d{6}$")

records = []
for step_dir in sorted(test_root.glob("steps_*")):
    step = step_dir.name.replace("steps_", "")
    for run in step_dir.iterdir():
        if not run.is_dir():
            continue
        m = pattern.match(run.name)
        if not m:
            continue
        kol = m.group(1)
        metrics_path = run / "event" / "metrics_test.json"
        if not metrics_path.exists():
            metrics_path = run / "metrics_test.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = data.get("daily_metrics", {}) if use_daily else data
        records.append(
            {
                "kol": kol,
                "iql_steps": int(step),
                "cumulative_return": metrics.get("cumulative_return"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
            }
        )

df = pd.DataFrame(records)
if df.empty:
    raise SystemExit("No metrics_test.json found under sweep test root.")

out_dir = test_root.parent / "summary"
out_dir.mkdir(parents=True, exist_ok=True)
suffix = "daily" if use_daily else "event"
csv_path = out_dir / f"sweep_metrics_{suffix}.csv"
md_path = out_dir / f"sweep_metrics_{suffix}.md"
df.sort_values(["kol", "iql_steps"]).to_csv(csv_path, index=False)
md_path.write_text(df.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
print(f"Saved: {csv_path}")
print(f"Saved: {md_path}")
PY
