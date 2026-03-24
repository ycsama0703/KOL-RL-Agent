# Five-Point Ablation Story (Full + Baseline + 3 Key Ablations)

## What Is Included
- `BASELINE`
- `KICL` (full model)
- `WO_HARD` (remove hard intent constraints)
- `WO_RL_COMPLETION` (disable RL completion; near-baseline proxy)
- `WO_REGIME_SPLIT` (single-head without regime split)

## Why These 3 Ablations
- `WO_HARD`: tests feasibility layer necessity (hard-violation control).
- `WO_RL_COMPLETION`: tests whether RL completion contributes beyond baseline imitation.
- `WO_REGIME_SPLIT`: tests whether signal/silence architecture contributes structurally.

## High-Level Observations
### X
- `KICL`: event return=0.108, sharpe=0.473, UER=0.000, DRR=0.000, BD=0.007.
- `WO_HARD`: event return=0.014, sharpe=-0.029, UER=0.191, DRR=0.204, BD=0.025.
- `WO_RL_COMPLETION`: event return=0.021, sharpe=0.295, UER=0.000, DRR=0.000, BD=0.000.
- `WO_REGIME_SPLIT`: event return=0.066, sharpe=0.408, UER=0.000, DRR=0.000, BD=0.007.
- Daily win vs baseline (count out of 10): `KICL`=8, `WO_HARD`=8, `WO_RL_COMPLETION`=5, `WO_REGIME_SPLIT`=6.

### YOUTUBE
- `KICL`: event return=0.287, sharpe=2.560, UER=0.000, DRR=0.000, BD=0.008.
- `WO_HARD`: event return=0.100, sharpe=2.135, UER=0.259, DRR=0.235, BD=0.036.
- `WO_RL_COMPLETION`: event return=0.270, sharpe=2.520, UER=0.000, DRR=0.000, BD=0.000.
- `WO_REGIME_SPLIT`: event return=0.289, sharpe=2.465, UER=0.000, DRR=0.000, BD=0.010.
- Daily win vs baseline (count out of 10): `KICL`=9, `WO_HARD`=2, `WO_RL_COMPLETION`=8, `WO_REGIME_SPLIT`=7.

## How To Tell the Story (Paper-Friendly)
1. Constraint necessity: removing hard constraints (`WO_HARD`) sharply increases hard betrayal (UER/DRR), and weakens robust gains.
2. Completion necessity: without RL completion (`WO_RL_COMPLETION`), behavior stays close to baseline but incremental gains over `KICL` shrink.
3. Structural contribution: removing regime split (`WO_REGIME_SPLIT`) degrades performance consistency, indicating value of signal/silence decomposition.

## Files
- `summary_by_kol.csv`: per-KOL raw compare table from five-point run
- `summary_by_method_mean_by_source.csv`: method means by source (4 methods)
- `five_point_summary_by_source.csv`: story-ready 5-point table (adds BASELINE row)
- `five_point_summary_overall.csv`: averaged across sources
- `five_point_win_vs_baseline_by_source.csv`: win/tie/lose vs baseline
