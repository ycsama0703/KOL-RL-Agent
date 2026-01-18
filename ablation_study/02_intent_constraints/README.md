02_intent_constraints - soft intent penalties

Compares:
- constraints_on: default entry/reversal penalty terms
- constraints_off: disable soft penalties and relax thresholds

Note: evaluation still uses hard sign constraints in apply_intent_constraints.

Usage:
  bash ablation_study/02_intent_constraints/run_constraints_on.sh
  bash ablation_study/02_intent_constraints/run_constraints_off.sh

Optional env vars:
- BUFFER_ROOT (default: data/buffer_22-24_end1231 if exists, else data/buffer_22-24)
- OUTPUT_DIR, TEST_DIR, DEVICE, EXPORT_SIGNAL
