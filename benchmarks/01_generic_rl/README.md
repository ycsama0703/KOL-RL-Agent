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

## Training

### BC (all sources/all KOLs)

```bash
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
OUTPUT_ROOT=outputs/benchmarks/generic_rl/bc_mainline \
LOG_ROOT=logs/benchmark_bc_mainline \
bash benchmarks/01_generic_rl/run_bc_multisource.sh
```

### Vanilla IQL (all sources/all KOLs)

```bash
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
OUTPUT_ROOT=outputs/benchmarks/generic_rl/iql_mainline \
LOG_ROOT=logs/benchmark_iql_mainline \
IQL_STEPS=200000 \
bash benchmarks/01_generic_rl/run_iql_multisource.sh
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
bash benchmarks/01_generic_rl/batch_test_vanilla_multisource.sh
```

## Dailyline

Replace `BUFFER_ROOT` with:

- `data/multisource_ready_22-25/08_replay_buffer_daily`
