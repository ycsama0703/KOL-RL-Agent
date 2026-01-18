Ablation study workspace

This folder contains per-ablation subfolders with runnable scripts and notes.
Each subfolder focuses on one dimension:

- 01_behavior: behavior policy construction (baseline vs lag/decay).
- 02_intent_constraints: soft intent penalties on/off.
- 03_training_flow: BC-only vs IQL-only vs BC+IQL.
- 04_component_ablation: component-level ablation (BC / IQL variants).

Common environment variables (all optional):
- PYTHON: Python executable (default: python)
- BUFFER_ROOT: replay buffer root for training/testing
- REWARD_DIR: reward CSV root for building buffers
- OUTPUT_DIR: training output root
- TEST_DIR: test output root
- DEVICE: evaluation device (default: cpu)
- EXPORT_SIGNAL: 1 to export signal_decisions_test.csv

Run a subfolder script from repo root, e.g.:
  bash ablation_study/03_training_flow/run_bc_only.sh
