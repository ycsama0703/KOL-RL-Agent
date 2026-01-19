04_component_ablation - component ablation set

This folder contains a single ablation group focused on components:
- bc_only
- iql_modified_only
- iql_vanilla_only
- bc_plus_vanilla_iql
- full_modified (BC + modified IQL, current method)

Usage (run from repo root):
  bash ablation_study/04_component_ablation/run_full_modified.sh
  bash ablation_study/04_component_ablation/run_bc_only.sh
  bash ablation_study/04_component_ablation/run_iql_modified_only.sh
  bash ablation_study/04_component_ablation/run_iql_vanilla_only.sh
  bash ablation_study/04_component_ablation/run_bc_plus_vanilla_iql.sh

Notes:
- Vanilla IQL runs use parallel scripts in this folder and do not touch main training/eval scripts.
- Vanilla runs use `batch_test_vanilla.sh` for evaluation (no signal_decisions export).

Optional env vars:
- PYTHON: Python executable (default: python)
- BUFFER_ROOT: replay buffer root (default: data/buffer_22-24_end1231 if exists)
- REWARD_ROOT: reward CSV root (auto from BUFFER_ROOT if unset)
- OUTPUT_DIR / TEST_DIR: output roots
- DEVICE: evaluation device
- EXPORT_SIGNAL: 1 to export signal_decisions_test.csv
- BC_BATCH_SIZE / IQL_BATCH_SIZE: override batch sizes (default 256)
