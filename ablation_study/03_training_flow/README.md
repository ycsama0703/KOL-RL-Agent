03_training_flow - BC vs IQL

Compares:
- bc_only: behavior cloning only (iql_steps=0)
- iql_only: IQL only (bc_epochs=0)
- bc_iql: default full training

Usage:
  bash ablation_study/03_training_flow/run_bc_only.sh
  bash ablation_study/03_training_flow/run_iql_only.sh
  bash ablation_study/03_training_flow/run_bc_iql.sh

Optional env vars:
- BUFFER_ROOT (default: data/buffer_22-24_end1231 if exists, else data/buffer_22-24)
- OUTPUT_DIR, TEST_DIR, DEVICE, EXPORT_SIGNAL
