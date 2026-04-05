# Training Config Supplement (for Appendix)

This file records the actual training/testing configuration used in this repo, aligned with current scripts and code.

## 1. Experiment Scope

- Main code path:
  - Training: `train.py` (KICL + several train.py-based baselines)
  - Evaluation: `scripts/evaluate_run.py`
  - Batch test launcher: `benchmarks/01_generic_rl/batch_test_vanilla_multisource.sh`
- Main replay-buffer roots:
  - Selected-20 (10 X + 10 YouTube): `data/multisource_selected20/08_replay_buffer`
  - Full multisource: `data/multisource_ready_22-25/08_replay_buffer`

## 2. Data Split Setting

- Split style: chronological by `trading_day`, same-day samples stay in same split.
- Split ratio: train/val/test = `0.6 / 0.2 / 0.2`.
- Relevant script: `scripts/prepare_step03_daily_split.py`.

## 3. KICL Model Architecture

Defined in `src/training/models.py`.

- Actor (dual-head, deterministic):
  - Backbone MLP: `state_dim -> 512 -> 512 -> 256` (ReLU)
  - `delta_signal` head: `256 -> 1` + `Tanh`
  - `delta_decay` head: `256 -> 1` + `Tanh`
- Critic (Q):
  - Input is residual-aware: `[state, baseline_action, delta_action]`
  - Effective MLP input dim: `state_dim + 2`
  - Hidden: `512 -> 512 -> 256`, output `1`
- Value:
  - Input is residual-aware: `[state, baseline_action]`
  - Effective MLP input dim: `state_dim + 1`
  - Hidden: `512 -> 512 -> 256`, output `1`

## 4. KICL Full Training Hyperparameters

From `train.py` defaults (unless overridden by launcher):

- BC stage:
  - `bc_epochs=10`
  - `bc_batch_size=256`
  - `bc_lr=3e-4`
  - `bc_fit_behavior=True`
  - `bc_anchor_lambda=0.03`
- IQL stage:
  - `iql_steps=200000` (default)
  - `iql_batch_size=256`
  - `actor_lr=3e-4`
  - `critic_lr=3e-4`
  - `value_lr=3e-4`
  - `gamma=0.99`
  - `expectile=0.7`
  - `temperature_beta=3.0`
- Intent/constraint-related:
  - `fidelity_lambda=0.03`
  - `actor_align_lambda=0.04`
  - `entry_penalty_lambda=0.02`
  - `reversal_penalty_lambda=0.05`
  - `entry_threshold=5e-4`
  - `clamp_delta=1.8`
  - `hard_intent_constraints=True`
  - `regime_split=True`
  - `zero_market_factors=False`
  - `market_factor_dim=6`
- Logging/runtime:
  - `log_interval=200`
  - `write_iql_csv=True`
  - `progress_bar=True`
  - Device: auto (`cuda` if available, else `cpu`)

Notes:
- In selected-20 hard-scope runs, launcher default is `iql_steps=100000` (`run_ablation_hard_scope.sh`).
- In some ablation/benchmark runs, explicit flags override the defaults below.

## 5. Ablation: KICL Variant Overrides

Script: `benchmarks/01_generic_rl/run_ablation_kicl.sh`.

- `full`: no override
- `w_no_hard`: `--no-hard-intent-constraints`
- `w_no_soft`: `--bc-anchor-lambda 0.0 --fidelity-lambda 0.0 --actor-align-lambda 0.0 --entry-penalty-lambda 0.0 --reversal-penalty-lambda 0.0`
- `w_no_bc_anchor`: `--bc-anchor-lambda 0.0`
- `w_no_rl_completion`: `--iql-steps 0 --no-bc-fit-behavior --bc-anchor-lambda 0.0`
- `w_no_fidelity`: `--fidelity-lambda 0.0`
- `w_no_reversal_penalty`: `--reversal-penalty-lambda 0.0`
- `w_no_entry_penalty`: `--entry-penalty-lambda 0.0`
- `w_no_market_factors`: `--zero-market-factors --market-factor-dim 6`
- `w_single_head_no_regime_split`: `--no-regime-split`

## 6. Ablation: Hard-Scope Variants

Script: `benchmarks/01_generic_rl/run_ablation_hard_scope.sh`.

- `hard_both`:
  - Train: hard on
  - Test: hard on
- `hard_train_only`:
  - Train: hard on
  - Test: hard off
- `hard_infer_only`:
  - Train: hard off
  - Test: hard on
- `hard_none`:
  - Train: hard off
  - Test: hard off

Default launcher params for this hard-scope script:
- `BC_EPOCHS=10`
- `IQL_STEPS=100000`
- `BATCH_SIZE=256`

## 7. Benchmark Method Configs

### 7.1 train.py-based methods

- BC (`run_bc_multisource.sh`)
  - `bc_epochs=10`, `iql_steps=0`
  - `bc_fit_behavior=True`
  - hard off
  - soft penalties off
  - `bc_anchor_lambda=0.0`

- Vanilla IQL (`run_iql_multisource.sh`)
  - `bc_epochs=0`, `iql_steps=200000` (script default)
  - hard off
  - soft penalties off

- RMB: Risk-Managed Baseline (`run_risk_managed_baseline_multisource.sh`)
  - `bc_epochs=10`, `iql_steps=0`
  - `no-bc-fit-behavior`
  - hard on
  - `entry_threshold=0.01`
  - `clamp_delta=0.6`
  - soft penalties off

- HAP: Heuristic Allocation Proxy (`run_heuristic_allocation_multisource.sh`)
  - `bc_epochs=10`, `iql_steps=0`
  - hard off
  - `no-regime-split`
  - `zero-market-factors` on (`market_factor_dim=6`)
  - soft penalties off
  - `bc_anchor_lambda=0.0`

- SUP_DELTA: Supervised Delta (`run_supervised_delta_multisource.sh`)
  - `bc_epochs=10`, `iql_steps=0`
  - hard on
  - `bc_anchor_lambda=0.03`
  - `entry_penalty_lambda=0.02`
  - `reversal_penalty_lambda=0.05`
  - `fidelity_lambda=0.0`
  - `actor_align_lambda=0.0`

### 7.2 dedicated benchmark trainers

- CQL (`run_cql_multisource.sh` -> `benchmarks/01_generic_rl/train_cql.py`)
  - `batch_size=256`
  - `steps=200000`
  - `cql_alpha=1.0`
  - `cql_temp=1.0`
  - `cql_n_actions=10`

- AWAC (`run_awac_multisource.sh` -> `benchmarks/01_generic_rl/train_awac.py`)
  - `batch_size=256`
  - `steps=200000`
  - `awac_beta=1.0`
  - `awac_max_weight=20.0`

- TD3BC (`run_td3bc_multisource.sh` -> `benchmarks/01_generic_rl/train_td3bc.py`)
  - `batch_size=256`
  - `steps=200000`

## 8. Unified Test/Evaluation Config

Core evaluation script: `scripts/evaluate_run.py`.

Common test settings in batch scripts:
- `action_threshold=0.02`
- `daily_price_update=True`
- `plot=True`
- `DEVICE` usually `cuda` on server
- `MAX_JOBS` typically 8~10 according to GPU memory budget

Method-specific test flags (passed via wrapper scripts):
- Hard on/off: `--hard-intent-constraints` or `--no-hard-intent-constraints`
- Regime split on/off: `--regime-split` or `--no-regime-split`
- Market factor ablation: `--zero-market-factors --market-factor-dim 6`

## 9. Logging and Artifacts

- Each run writes:
  - `run_summary.json`
  - `logs/training.log`
  - `logs/iql_metrics.csv` (if enabled)
  - `checkpoints/policy.pt` (plus critic/value ckpts)
- Each test run writes:
  - `event/metrics_test.json`
  - `event/positions_test.csv`
  - `daily/performance.csv`
  - event/daily equity plots

## 10. Reproducibility Notes

- No fixed global random seed is enforced in `train.py` by default.
- Concurrency is controlled in launchers via `MAX_JOBS`.
- For paper appendix, report both:
  1) code defaults (this file Sections 4/7/8), and
  2) run-time overrides used by each experiment group (mainline, selected-20, ablation, hard-scope).
