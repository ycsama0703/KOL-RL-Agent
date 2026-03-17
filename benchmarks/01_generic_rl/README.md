# Generic RL Benchmarks (Mainline Entrypoints)

This folder runs benchmark baselines using the same mainline entrypoints as your current pipeline:

- training: `train.py`
- evaluation: `scripts/evaluate_run.py`

No custom benchmark trainer/evaluator is used.

## Methods

- `BC` benchmark:
  - `iql_steps=0`
  - hard constraints disabled
  - intent auxiliary losses disabled
- `vanilla IQL` benchmark:
  - `bc_epochs=0`
  - hard constraints disabled
  - intent auxiliary losses disabled
- `TD3+BC` benchmark:
  - twin-critic TD3 target + BC actor regularization
  - hard constraints disabled
  - intent auxiliary losses disabled

## Training

Launch style is aligned with your recent `ours` batch runs:

- one `nohup train.py` per KOL
- flat logs under `LOG_ROOT` named as `<source>_<kol>_<RUN_TAG>.log`
- output split by source (`OUTPUT_ROOT/youtube`, `OUTPUT_ROOT/x`)
- parallelism controlled by `MAX_JOBS`
- testing follows the same launch/logging style

### BC (all sources/all KOLs)

```bash
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
OUTPUT_ROOT=outputs/benchmarks/generic_rl/bc_mainline \
LOG_ROOT=logs/benchmark_bc_mainline \
MAX_JOBS=8 \
bash benchmarks/01_generic_rl/run_bc_multisource.sh
```

### Vanilla IQL (all sources/all KOLs)

```bash
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
OUTPUT_ROOT=outputs/benchmarks/generic_rl/iql_mainline \
LOG_ROOT=logs/benchmark_iql_mainline \
MAX_JOBS=8 \
IQL_STEPS=200000 \
bash benchmarks/01_generic_rl/run_iql_multisource.sh
```

### TD3+BC (all sources/all KOLs)

```bash
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
OUTPUT_ROOT=outputs/benchmarks/generic_rl/td3bc_mainline \
LOG_ROOT=logs/benchmark_td3bc_mainline \
MAX_JOBS=8 \
TD3BC_STEPS=200000 \
bash benchmarks/01_generic_rl/run_td3bc_multisource.sh
```

If you want to pin visible GPUs, set it outside:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ...
```

## Testing

### BC test (event + daily outputs)

```bash
TRAIN_ROOT=outputs/benchmarks/generic_rl/bc_mainline \
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
TEST_ROOT=outputs/benchmarks/generic_rl/bc_mainline_test \
LOG_ROOT=logs/benchmark_bc_mainline_test \
DEVICE=cuda \
HARD_INTENT_CONSTRAINTS=0 \
DAILY_PRICE_UPDATE=1 \
MAX_JOBS=8 \
bash benchmarks/01_generic_rl/batch_test_vanilla_multisource.sh
```

### Vanilla IQL test (event + daily outputs)

```bash
TRAIN_ROOT=outputs/benchmarks/generic_rl/iql_mainline \
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
TEST_ROOT=outputs/benchmarks/generic_rl/iql_mainline_test \
LOG_ROOT=logs/benchmark_iql_mainline_test \
DEVICE=cuda \
HARD_INTENT_CONSTRAINTS=0 \
DAILY_PRICE_UPDATE=1 \
MAX_JOBS=8 \
bash benchmarks/01_generic_rl/batch_test_vanilla_multisource.sh
```

### TD3+BC test (event + daily outputs)

```bash
TRAIN_ROOT=outputs/benchmarks/generic_rl/td3bc_mainline \
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
TEST_ROOT=outputs/benchmarks/generic_rl/td3bc_mainline_test \
LOG_ROOT=logs/benchmark_td3bc_mainline_test \
DEVICE=cuda \
HARD_INTENT_CONSTRAINTS=0 \
DAILY_PRICE_UPDATE=1 \
MAX_JOBS=8 \
bash benchmarks/01_generic_rl/batch_test_vanilla_multisource.sh
```

## Dailyline

Replace `BUFFER_ROOT` with:

- `data/multisource_ready_22-25/08_replay_buffer_daily`

## Comparison / Reporting

Use the compare builder to merge `KICL + BC + IQL` results into one reporting folder:

```bash
python benchmarks/01_generic_rl/build_compare_report.py \
  --mode anchor_ours \
  --event-curve-mode daily_mtm
```

Default output:

- `benchmarks/compare`

Important:

- With `--event-curve-mode daily_mtm`, `event_equity_compare` should be interpreted with `daily_metrics_compare` (not `event_metrics_compare`).
- If you need strict event-step curve/metric alignment, switch to `--event-curve-mode signal_step`.

Detailed protocol:

- `docs/benchmark_compare_protocol.md`
