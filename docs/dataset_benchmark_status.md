# Dataset & Benchmark Status

Last updated: 2026-03-21 (Selected-20 stats + signal/silence refreshed)

## 1) Dataset Snapshot (Current)

### Source-level discourse statistics (Table 1)

This table is the **current paper scope** (20 KOLs total: 10 YouTube + 10 X).

Source file: `outputs/analysis/company_stats_selected20_20260321/table1_selected20_youtube_vs_x.md`

| Statistic | YouTube | X |
|---|---:|---:|
| Coverage: Mentioned companies | 4,871 | 2,960 |
| Single-mention ratio | 60.71% | 45.37% |
| No follow-up within 90 days | 82.20% | 73.38% |
| Silence duration (p90, days) | 125.76 | 58.00 |
| Sentiment reversal rate | 62.08% | 64.09% |
| Median time to first reversal (days) | 189.83 | 161.00 |
| Positive ratio | 49.84% | 62.47% |
| Negative ratio | 31.50% | 29.74% |
| Neutral ratio | 18.66% | 7.78% |
| Rows | 32,620 | 49,542 |
| KOLs | 10 | 10 |
| Date range (UTC) | 2022-01-01 -> 2025-12-31 | 2022-01-03 -> 2025-12-31 |

### Pipeline inventory (selected-20 snapshot)

Selected subset roots:
- YouTube 10: `data/multisource_ready_22-25` (KOL subset from `benchmarks/compare/meta/kicl_top10_vs_baseline_youtube.csv`)
- X 10: `data/multisource_ready_22-25_xrefresh_20260320_144701` (subset from `benchmarks/compare/meta/kicl_top10_vs_baseline_x.csv`)

| Stage | YouTube (10 KOLs) | X (10 KOLs) |
|---|---:|---:|
| 00_mapped_full (rows / KOLs) | 32,620 / 10 | 49,542 / 10 |
| 04_clean (train+val+test rows / KOLs) | 31,644 / 10 | 41,625 / 10 |
| 06_enriched (train+val+test rows / KOLs) | 17,670 / 10 | 33,409 / 10 |
| Enriched retention vs clean | 55.84% | 80.26% |

### Replay buffer inventory (selected-20 snapshot)

Subset replay root: `data/multisource_selected20/08_replay_buffer`

| Buffer | Source | KOLs | Train samples | Val samples | Test samples |
|---|---|---:|---:|---:|---:|
| `data/multisource_selected20/08_replay_buffer` | youtube | 10 | 128,948 | 31,879 | 33,306 |
| `data/multisource_selected20/08_replay_buffer` | x | 10 | 728,289 | 137,219 | 169,380 |

### Signal vs Silence statistics (Selected-20 replay subset)

Scope:
- 20 KOLs (YouTube 10 + X 10), replay subset root: `data/multisource_selected20/08_replay_buffer`
- Signal definition: `abs(baseline_actions) > 1e-8`
- Script output: `outputs/signal_silence_stats_selected20`

Global:
- \|D_sig\| = **45,626**
- \|D_sil\| = **1,183,395**
- \|D_sil\| / \|D_sig\| = **25.936856**

By source:

| source | d_sig | d_sil | d_total | rho_sig | rho_sil | sil_sig_ratio |
|---|---:|---:|---:|---:|---:|---:|
| x | 30,711 | 1,004,177 | 1,034,888 | 0.029676 | 0.970324 | 32.697633 |
| youtube | 14,915 | 179,218 | 194,133 | 0.076829 | 0.923171 | 12.015957 |

By split:

| split | d_sig | d_sil | d_total | rho_sig | rho_sil | sil_sig_ratio |
|---|---:|---:|---:|---:|---:|---:|
| train | 25,174 | 832,063 | 857,237 | 0.029366 | 0.970634 | 33.052475 |
| val | 9,730 | 159,368 | 169,098 | 0.057541 | 0.942459 | 16.379034 |
| test | 10,722 | 191,964 | 202,686 | 0.052900 | 0.947100 | 17.903749 |

Ticker-level concentration:
- #tickers = 1,845
- pct(\|D_sil^(i)\| > \|D_sig^(i)\|) = 0.997290
- median(\|D_sil^(i)\| / \|D_sig^(i)\|, finite) = 90.0
- p25 / p75 (finite) = 39.960317 / 163.5

## 2) Benchmark Summary (Mainline Package)

Package root: `benchmarks/benchmark_package/mainline`  
Source tables:
- `benchmarks/benchmark_package/mainline/tables/benchmark_event_by_source.csv`
- `benchmarks/benchmark_package/mainline/tables/benchmark_betrayal_by_source.csv`
- `benchmarks/benchmark_package/mainline/tables/benchmark_ranking_event_return.csv`
- `benchmarks/benchmark_package/mainline/tables/benchmark_ranking_betrayal.csv`

### 2.1 Event-level performance (mean over KOLs)

| source | method | n_kols | event_mean_cumulative_return | event_mean_sharpe | event_mean_max_drawdown |
|---|---|---:|---:|---:|---:|
| x | KICL | 11 | 0.047351 | 0.332001 | 0.344728 |
| x | BC | 11 | -0.023725 | -0.158124 | 0.132923 |
| x | IQL | 11 | -0.022477 | -0.176892 | 0.127357 |
| x | TD3BC | 11 | -0.012327 | -0.235952 | 0.079881 |
| x | CQL | 11 | -0.008809 | -0.072899 | 0.085579 |
| x | AWAC | 11 | -0.000557 | -0.050203 | 0.122947 |
| youtube | KICL | 17 | 0.326655 | 2.264758 | 0.171590 |
| youtube | BC | 17 | 0.142780 | 1.751556 | 0.071887 |
| youtube | IQL | 17 | 0.180790 | 2.255437 | 0.073327 |
| youtube | TD3BC | 17 | 0.017211 | 0.320946 | 0.056013 |
| youtube | CQL | 17 | 0.064585 | 0.987029 | 0.047978 |
| youtube | AWAC | 17 | 0.200311 | 2.290908 | 0.072913 |

### 2.2 Betrayal / intent-deviation metrics (mean over KOLs)

| source | method | n_kols | reversal_rate | entry_violation_rate | mean_abs_deviation | mean_normalized_deviation | sign_agreement_rate | baseline_policy_corr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| x | KICL | 11 | 0.000000 | 0.000000 | 0.006658 | 0.602416 | 0.771089 | 0.820293 |
| x | BC | 11 | 0.204395 | 0.200004 | 0.022533 | 0.724292 | 0.795605 | 0.838333 |
| x | IQL | 11 | 0.218645 | 0.176836 | 0.024805 | 0.735109 | 0.781355 | 0.835648 |
| x | TD3BC | 11 | 0.399061 | 0.792042 | 0.180967 | 1.992841 | 0.600939 | 0.428746 |
| x | CQL | 11 | 0.431061 | 0.857508 | 0.103604 | 1.462592 | 0.568939 | 0.732403 |
| x | AWAC | 11 | 0.205911 | 0.174271 | 0.024451 | 0.706508 | 0.794089 | 0.827135 |
| youtube | KICL | 17 | 0.000000 | 0.000000 | 0.007986 | 0.592949 | 0.773321 | 0.831828 |
| youtube | BC | 17 | 0.241142 | 0.244231 | 0.033054 | 0.816593 | 0.758858 | 0.807021 |
| youtube | IQL | 17 | 0.221392 | 0.221172 | 0.028023 | 0.764211 | 0.778608 | 0.826651 |
| youtube | TD3BC | 17 | 0.450326 | 0.942119 | 0.208655 | 2.023022 | 0.549674 | 0.563324 |
| youtube | CQL | 17 | 0.287087 | 0.906420 | 0.214441 | 1.913783 | 0.712913 | 0.405597 |
| youtube | AWAC | 17 | 0.203355 | 0.235964 | 0.031197 | 0.742247 | 0.796645 | 0.815943 |

### 2.3 Rankings

Event return ranking (`rank_event_return`, lower is better rank):
- x: KICL (1), AWAC (2), CQL (3), TD3BC (4), IQL (5), BC (6)
- youtube: KICL (1), AWAC (2), IQL (3), BC (4), CQL (5), TD3BC (6)

Betrayal abs-dev ranking (`rank_betrayal_absdev`, lower is better rank):
- x: KICL (1), BC (2), AWAC (3), IQL (4), CQL (5), TD3BC (6)
- youtube: KICL (1), IQL (2), AWAC (3), BC (4), TD3BC (5), CQL (6)

## 3) Figures (benchmark package)

- `benchmarks/benchmark_package/mainline/figures/event_return_by_source.png`
- `benchmarks/benchmark_package/mainline/figures/betrayal_absdev_by_source.png`
- `benchmarks/benchmark_package/mainline/figures/tradeoff_return_vs_intent_by_source.png`

## 4) Notes for interpretation

- Table 1 (Section 1) is now anchored to the selected paper subset: `10 YouTube + 10 X`.
- Legacy benchmark package (`benchmarks/benchmark_package/mainline`) still reports the earlier canonical intersection (`11 X + 17 YouTube`).
- When writing the paper, keep each reported table tied to its explicit KOL scope.

## 5) Selected-20 Benchmark Subset (Paper Set)

This section tracks the paper-facing subset: **Top-10 per source** (YouTube + X), selected by:
- `uplift_vs_baseline = trained_cumulative_return - baseline_cumulative_return` (KICL row in `daily_metrics_compare.csv`)

Selection sources:
- `benchmarks/compare/meta/kicl_top10_vs_baseline_selection.md`
- `benchmarks/compare/meta/kicl_top10_vs_baseline_youtube.csv`
- `benchmarks/compare/meta/kicl_top10_vs_baseline_x.csv`

### 5.1 Selected KOLs

YouTube (10):
- Financial_Education
- The_Maverick_of_Wall_Street
- Ale_s_World_of_Stocks
- Invest_with_Henry
- Daniel_Pronk
- Unrivaled_Investing
- Dividend_Data
- Sven_Carlin_Ph.D.
- Investing_with_Tom
- Humbled_Trader

X (10):
- Stocktwits
- Jake__Wujastyk
- goldseek
- traderstewie
- intocryptoverse
- bespokeinvest
- EliteOptions2
- Mr_Derivatives
- GoldTelegraph_
- Stephanie_Link

### 5.2 Method Summary on Selected-20 (vs Baseline)

Source: `benchmarks/compare/meta/selected20_method_summary_vs_baseline.csv`

| source | method | n_kols | mean_event_return | mean_event_sharpe | mean_event_mdd | mean_daily_return | mean_daily_sharpe | mean_daily_mdd | mean_uplift_return | mean_uplift_sharpe | mean_mdd_improve | win_rate_vs_baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x | KICL | 10 | 0.108450 | 0.472568 | 0.316003 | 0.116484 | 0.424075 | 0.317445 | 0.090544 | 0.166737 | 0.014302 | 0.800000 |
| x | BC | 10 | 0.012833 | -0.017254 | 0.106768 | 0.193345 | 0.649350 | 0.244631 | 0.167427 | 0.392038 | 0.087116 | 0.800000 |
| x | IQL | 10 | 0.013391 | -0.025917 | 0.101540 | 0.192693 | 0.690116 | 0.244644 | 0.169700 | 0.440336 | 0.088885 | 0.800000 |
| x | CQL | 10 | -0.010204 | -0.093378 | 0.088283 | 0.167327 | 0.353045 | 0.267812 | 0.142208 | 0.097641 | 0.064014 | 0.500000 |
| x | TD3BC | 10 | -0.021023 | -0.391251 | 0.084121 | -0.095745 | -0.605289 | 0.303239 | -0.121684 | -0.862626 | 0.028508 | 0.300000 |
| x | AWAC | 10 | 0.028994 | 0.062023 | 0.100706 | 0.202544 | 0.701472 | 0.239592 | 0.176547 | 0.443725 | 0.092058 | 0.700000 |
| youtube | KICL | 10 | 0.286662 | 2.560002 | 0.159200 | 0.341863 | 1.288860 | 0.189160 | 0.024966 | 0.040138 | -0.003968 | 0.900000 |
| youtube | BC | 10 | 0.088105 | 1.941166 | 0.073595 | 0.241258 | 1.121222 | 0.180442 | -0.075640 | -0.127500 | 0.004751 | 0.300000 |
| youtube | IQL | 10 | 0.121355 | 2.427473 | 0.074564 | 0.246134 | 1.233284 | 0.178103 | -0.070763 | -0.015438 | 0.007090 | 0.300000 |
| youtube | CQL | 10 | 0.012612 | 0.511559 | 0.050205 | 0.002057 | 0.142368 | 0.173029 | -0.316401 | -1.110213 | 0.011603 | 0.100000 |
| youtube | TD3BC | 10 | -0.005631 | 0.033582 | 0.064493 | -0.078188 | -0.932426 | 0.236131 | -0.395085 | -2.181149 | -0.050938 | 0.100000 |
| youtube | AWAC | 10 | 0.123546 | 2.413181 | 0.071933 | 0.248084 | 1.211018 | 0.183872 | -0.068813 | -0.037704 | 0.001320 | 0.400000 |

### 5.3 Detailed per-KOL/per-method table (including BASELINE rows)

For full reproducibility and paper appendix:
- `benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed.csv`

This file contains, for each of the 20 selected KOLs and each method (`KICL, BC, IQL, CQL, TD3BC, AWAC, BASELINE`):
- event metrics (`event_cumulative_return`, `event_sharpe`, `event_max_drawdown`)
- daily metrics (`daily_trained_*`, `daily_baseline_*`)
- betrayal metrics (`reversal_rate`, `entry_violation_rate`, `mean_abs_deviation`, `sign_agreement_rate`, `baseline_policy_corr`)
- uplift vs baseline (`cumret_uplift_vs_baseline`, `sharpe_uplift_vs_baseline`, `mdd_improvement_vs_baseline`)
