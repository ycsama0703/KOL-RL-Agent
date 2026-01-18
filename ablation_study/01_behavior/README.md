01_behavior - behavior policy construction

Compares:
- baseline behavior (alpha=1.0, decay=1.0) -> behavior equals baseline
- lag/decay behavior (alpha=0.3, decay=0.2)

Scripts build a dedicated replay buffer, train all KOLs, then run batch tests.

Usage:
  bash ablation_study/01_behavior/run_baseline.sh
  bash ablation_study/01_behavior/run_lag_decay.sh

Optional env vars:
- REWARD_DIR (default: data/processed/reward_end1231)
- TICKER_VOCAB, TICKER_EMB
- BUFFER_ROOT, OUTPUT_DIR, TEST_DIR
- DEVICE, EXPORT_SIGNAL
- SKIP_BUILD=1 to skip buffer rebuild
