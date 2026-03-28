# Benchmark Upgrade Brief (for GPT-assisted LaTeX editing)

This note summarizes the **latest benchmark upgrade** and provides explicit guidance for updating the paper LaTeX.

## 1) Scope and Evaluation Protocol

- Dataset scope for benchmark section: **Selected-20 KOL subset** (X: 10, YouTube: 10).
- Main benchmark protocol:
  - **Event-conditioned training**
  - **Event-level primary metrics** (used in main tables)
  - Daily curves are only for qualitative visualization in appendix/case-study.

## 2) Result Files to Use (Authoritative)

### Main benchmark (existing + new 3 methods)
- `benchmarks/compare/selected20_plus_new3/summary_by_method_mean.csv`
- `benchmarks/compare/selected20_plus_new3/summary_by_method_mean_by_source.csv`
- `benchmarks/compare/selected20_plus_new3/summary_by_kol.csv`

### With trading metrics and baseline-relative summary
- `benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed_benchtest_plus_trading.csv`
- `benchmarks/compare/meta/selected20_method_summary_vs_baseline_benchtest_plus_trading.csv`

> Note: Baseline is used as a **reference column** in meta files, but not treated as a peer method row in the upgraded benchmark method taxonomy.

## 3) Method Taxonomy (for cleaner presentation)

Use the following 4 groups:

1. **Intent-Constrained Policy**
   - `KICL`
2. **Generic Offline RL**
   - `IQL`, `CQL`, `TD3BC`, `AWAC`
3. **Imitation / Supervised**
   - `BC`, `SUP_DELTA`
4. **Heuristic / Rule-based**
   - `RMB`, `HAP`

Recommended row order in tables/figures:

`RMB, HAP | BC, SUP_DELTA | IQL, AWAC, CQL, TD3BC | KICL`

## 4) Method Description + Citation Policy (important)

### 4.1 Which methods are literature baselines vs implementation-defined baselines

- **Literature baselines (cite papers):**
  - `BC`, `IQL`, `CQL`, `TD3BC`, `AWAC`
- **Implementation-defined engineering baselines (do not pretend external SOTA):**
  - `RMB`, `HAP`, `SUP_DELTA`

### 4.2 One-line description for each new method (ready to use)

- `RMB` (Risk-Managed Baseline):
  - BC-only anchor-following policy with hard constraints; no RL refinement (`iql_steps=0`), intended as a conservative risk-managed control.
- `HAP` (Heuristic Allocation Policy):
  - BC-only heuristic-style policy with no hard constraints, no regime split, and zeroed market-factor branch; used to represent weakly structured allocation behavior.
- `SUP_DELTA` (Supervised Delta):
  - BC-only residual policy (`iql_steps=0`) under hard constraints with soft alignment penalties; represents supervised intent-conditioned adjustment without value-based RL.

### 4.3 Ready-to-paste paper wording (English)

```text
In addition to standard offline RL baselines (BC, IQL, CQL, TD3BC, AWAC), we include three implementation-defined non-RL controls to improve diagnostic coverage: Risk-Managed Baseline (RMB), Heuristic Allocation Policy (HAP), and Supervised Delta (SUP_DELTA). These are engineered from the same training/evaluation pipeline via ablated training switches (e.g., disabling IQL updates), and are reported as internal controls rather than externally standardized algorithms.
```

### 4.4 中文解释模板（给 GPT 或写作时参考）

```text
除文献中的离线RL基线（BC/IQL/CQL/TD3BC/AWAC）外，我们额外加入了三个工程化对照方法（RMB/HAP/SUP_DELTA）。这三者并非外部标准算法，而是在同一训练框架下通过关闭或调整训练模块构造的内部对照，用于更细致地诊断“收益提升是否来自意图保持下的执行补全”。
```

## 5) Metric Taxonomy (for cleaner columns)

Use 3 metric groups:

1. **Performance**
   - Return (event cumulative return, higher better)
   - Sharpe (higher better)
   - MDD (event max drawdown, lower better)
2. **Betrayal / Intent Violation**
   - UER (unsupported entry rate, lower better)
   - DRR (direction reversal rate, lower better)
   - BD (mean absolute deviation from baseline action, lower better)
3. **Trading Behavior** (newly added)
   - Turnover
   - Rebalance Frequency
   - Active Exposure Ratio

### Main text vs appendix recommendation

- Main table: keep compact with
  - `Return, Sharpe, MDD, UER, DRR, BD` (+ optional WinRate if needed).
- Appendix / supplementary table:
  - add `Turnover, RebalanceFreq, ActiveExposure`.

## 6) Newly Added Methods (3) and New Metrics (3)

### Newly added methods
- `RMB` (Risk-Managed Baseline)
- `HAP` (Heuristic Allocation Policy)
- `SUP_DELTA` (Supervised Delta)

### Newly added metrics
- `trading_mean_turnover`
- `trading_mean_rebalance_freq`
- `trading_mean_active_exposure_ratio`

All are already present in:
- `summary_by_method_mean.csv` (prefixed as `trading_mean_*`)
- `summary_by_method_mean_by_source.csv`
- `...plus_trading.csv` files in `benchmarks/compare/meta`

## 7) Key Overall Numbers (Selected-20, all methods)

From `benchmarks/compare/selected20_plus_new3/summary_by_method_mean_by_source.csv`:

### X (10 KOLs)

| Method | Return | Sharpe | MDD | UER | DRR | BD | Turnover | RebalanceFreq | ActiveExposure | WinRate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KICL | 0.108 | 0.473 | 0.316 | 0.000 | 0.000 | 0.0070 | 1.407 | 0.921 | 0.987 | 0.80 |
| SUP_DELTA | 0.021 | 0.304 | 0.321 | 0.000 | 0.000 | 0.0066 | 1.397 | 0.920 | 0.980 | 0.40 |
| RMB | 0.022 | 0.295 | 0.331 | 0.000 | 0.000 | 0.0000 | 1.476 | 0.921 | 0.992 | 0.70 |
| AWAC | 0.029 | 0.062 | 0.101 | 0.188 | 0.194 | 0.0262 | 0.671 | 0.979 | 1.000 | 0.70 |
| IQL | 0.013 | -0.026 | 0.102 | 0.190 | 0.208 | 0.0266 | 0.692 | 0.977 | 1.000 | 0.80 |
| HAP | 0.002 | -0.070 | 0.108 | 0.185 | 0.194 | 0.0222 | 0.699 | 0.997 | 1.000 | 0.70 |
| BC | 0.013 | -0.017 | 0.107 | 0.216 | 0.193 | 0.0241 | 0.655 | 0.997 | 1.000 | 0.80 |
| CQL | -0.010 | -0.093 | 0.088 | 0.844 | 0.380 | 0.1017 | 0.331 | 0.997 | 1.000 | 0.50 |
| TD3BC | -0.021 | -0.391 | 0.084 | 0.791 | 0.371 | 0.1795 | 0.446 | 0.997 | 1.000 | 0.30 |

### YouTube (10 KOLs)

| Method | Return | Sharpe | MDD | UER | DRR | BD | Turnover | RebalanceFreq | ActiveExposure | WinRate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KICL | 0.287 | 2.560 | 0.159 | 0.000 | 0.000 | 0.0084 | 1.468 | 0.998 | 0.918 | 0.90 |
| SUP_DELTA | 0.289 | 2.473 | 0.162 | 0.000 | 0.000 | 0.0097 | 1.479 | 0.998 | 0.919 | 0.70 |
| RMB | 0.270 | 2.519 | 0.155 | 0.000 | 0.000 | 0.0001 | 1.527 | 0.998 | 0.921 | 0.50 |
| AWAC | 0.124 | 2.413 | 0.072 | 0.270 | 0.182 | 0.0350 | 0.778 | 1.000 | 1.000 | 0.40 |
| IQL | 0.121 | 2.427 | 0.075 | 0.253 | 0.192 | 0.0309 | 0.787 | 1.000 | 1.000 | 0.30 |
| HAP | 0.118 | 2.431 | 0.065 | 0.354 | 0.122 | 0.0369 | 0.728 | 1.000 | 1.000 | 0.30 |
| BC | 0.088 | 1.941 | 0.074 | 0.274 | 0.239 | 0.0377 | 0.764 | 1.000 | 1.000 | 0.30 |
| CQL | 0.013 | 0.512 | 0.050 | 0.923 | 0.331 | 0.2374 | 0.624 | 1.000 | 1.000 | 0.10 |
| TD3BC | -0.006 | 0.034 | 0.064 | 0.935 | 0.504 | 0.2055 | 0.502 | 1.000 | 1.000 | 0.10 |

> WinRate definition used here: fraction of KOLs where `daily_trained_cumulative_return > daily_baseline_cumulative_return`, computed from `summary_by_kol.csv`.

## 8) Source-level Notes (X vs YouTube)

Use `summary_by_method_mean_by_source.csv` for platform-specific rows.

Minimal interpretation support:
- KICL keeps `UER=0` and `DRR=0` on both sources.
- `SUP_DELTA` and `RMB` also keep hard-violation rates near zero.
- Generic RL methods generally show higher betrayal rates.
- Trading behavior metrics distinguish “active but constrained” (`KICL/SUP_DELTA/RMB`) from “always-active” behavior (`ActiveExposure ~ 1.0` for many unconstrained baselines).

## 9) What GPT Should Change in LaTeX

1. **Benchmark methods paragraph**
   - Replace old method list with grouped taxonomy (4 groups above).
2. **Main benchmark table**
   - Keep primary event-level metrics.
   - Reorder rows using recommended method order.
   - Optionally include group separators.
3. **Add one supplementary benchmark table**
   - Include trading metrics: Turnover / RebalanceFreq / ActiveExposure.
4. **Figure/table captions**
   - Explicitly state event-level is primary; daily is for trajectory visualization only.
5. **Metric naming consistency**
   - Use consistent abbreviations: `UER`, `DRR`, `BD`.

## 10) Ready-to-copy Method Group Mapping

```text
Intent-Constrained Policy: KICL
Generic Offline RL: IQL, CQL, TD3BC, AWAC
Imitation/Supervised: BC, SUP_DELTA
Heuristic/Rule-based: RMB, HAP
```

## 11) Ready-to-copy Metric Group Mapping

```text
Performance: Return, Sharpe, MDD
Betrayal: UER, DRR, BD
Trading behavior: Turnover, RebalanceFreq, ActiveExposure
```
