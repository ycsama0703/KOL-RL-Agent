# Generic RL Benchmarks

This folder contains benchmark scripts for generic methods that do **not** use the repository's intent-preserving completion design.

Current methods:

- `BC`: vanilla single-head behavior cloning
- `IQL`: vanilla single-head IQL without BC warm start

These scripts are intended for **benchmark comparison**, not ablation.

## Data requirement

No new data generation is required if you already have replay buffers.

Supported replay roots:

- mainline: `data/multisource_ready_22-25/08_replay_buffer`
- dailyline: `data/multisource_ready_22-25/08_replay_buffer_daily`

You only need to rebuild data if you change:

- state definition
- reward definition
- replay construction logic

## Training

### BC benchmark

```bash
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
OUTPUT_ROOT=outputs/benchmarks/generic_rl/bc_mainline \
bash benchmarks/01_generic_rl/run_bc_multisource.sh
```

### Vanilla IQL benchmark

```bash
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
OUTPUT_ROOT=outputs/benchmarks/generic_rl/iql_mainline \
bash benchmarks/01_generic_rl/run_iql_multisource.sh
```

## Testing

### BC benchmark

```bash
TRAIN_ROOT=outputs/benchmarks/generic_rl/bc_mainline \
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
TEST_ROOT=outputs/benchmarks/generic_rl/bc_mainline_test \
DEVICE=cuda \
bash benchmarks/01_generic_rl/batch_test_vanilla_multisource.sh
```

### Vanilla IQL benchmark

```bash
TRAIN_ROOT=outputs/benchmarks/generic_rl/iql_mainline \
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer \
TEST_ROOT=outputs/benchmarks/generic_rl/iql_mainline_test \
DEVICE=cuda \
bash benchmarks/01_generic_rl/batch_test_vanilla_multisource.sh
```

## Dailyline example

BC:

```bash
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer_daily \
OUTPUT_ROOT=outputs/benchmarks/generic_rl/bc_daily \
bash benchmarks/01_generic_rl/run_bc_multisource.sh
```

```bash
TRAIN_ROOT=outputs/benchmarks/generic_rl/bc_daily \
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer_daily \
TEST_ROOT=outputs/benchmarks/generic_rl/bc_daily_test \
DEVICE=cuda \
bash benchmarks/01_generic_rl/batch_test_vanilla_multisource.sh
```

IQL:

```bash
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer_daily \
OUTPUT_ROOT=outputs/benchmarks/generic_rl/iql_daily \
bash benchmarks/01_generic_rl/run_iql_multisource.sh
```

```bash
TRAIN_ROOT=outputs/benchmarks/generic_rl/iql_daily \
BUFFER_ROOT=data/multisource_ready_22-25/08_replay_buffer_daily \
TEST_ROOT=outputs/benchmarks/generic_rl/iql_daily_test \
DEVICE=cuda \
bash benchmarks/01_generic_rl/batch_test_vanilla_multisource.sh
```
