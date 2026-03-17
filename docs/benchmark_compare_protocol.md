# Benchmark Comparison Protocol (Current Standard)

Last updated: **March 18, 2026**

This document defines the current, fixed standard for cross-method comparison and reporting.

## 1. Scope

This protocol covers:

- method naming in figures/tables
- which result folders are used for comparison
- how per-KOL comparison curves are built
- which metric table should be aligned with which curve
- reproducible command to regenerate comparison artifacts

## 2. Method Naming Standard

Use the following display names in all benchmark plots/tables:

- `KICL`: main method (previously referred to as "ours")
- `BC`: behavior cloning benchmark
- `IQL`: vanilla IQL benchmark

Do not use "ours" in final paper figures/tables.

## 3. Result Roots (Current Event-Line Comparison Snapshot)

Current compare script input roots:

- `KICL` -> `outputs/multisource_test_mainline`
- `BC` -> `benchmarks/01_generic_rl/test results/bc_event_test`
- `IQL` -> `benchmarks/01_generic_rl/test results/iql_event_test`

These roots correspond to the current **event-line benchmark runs** (with daily outputs also available under each run).

## 4. Curve and Metric Alignment Rules

### 4.1 Default curve mode for `event_equity_compare`

Current default in compare script:

- `event_curve_mode=daily_mtm`

Meaning:

- `event_equity_compare.(csv|png)` is built from each method's `daily/equity_daily.csv` trained curve.
- Curves are non-flat between sparse signal days (mark-to-market behavior).

### 4.2 Which table to align with this curve

When `event_curve_mode=daily_mtm`:

- align curve terminal return with `daily_metrics_compare.csv` -> `trained_cumulative_return`
- **do not** align with `event_metrics_compare.csv` cumulative return

Reason:

- `event_metrics_compare.csv` comes from event/signal-step evaluation metrics
- `daily_mtm` curves come from daily mark-to-market aggregation
- they are different metric families by definition

### 4.3 If strict event-metric/curve equality is required

Use:

- `--event-curve-mode signal_step`

Then `event_equity_compare` is reconstructed from event transitions (`positions_test.csv`) and its terminal return aligns with `event_metrics_compare.csv`.

## 5. Reproducible Compare Command

Default (recommended current standard):

```bash
python benchmarks/01_generic_rl/build_compare_report.py \
  --mode anchor_ours \
  --event-curve-mode daily_mtm
```

Optional strict event-step curve:

```bash
python benchmarks/01_generic_rl/build_compare_report.py \
  --mode anchor_ours \
  --event-curve-mode signal_step
```

## 6. Output Standard

All outputs are written to:

- `benchmarks/compare`

Global files:

- `compare_manifest.json`
- `summary_by_kol.csv`
- `summary_by_method_mean.csv`
- `overview_event_means.png`
- `overview_betrayal_means.png`

Per-KOL files under `benchmarks/compare/<source>/<kol>/`:

- `event_metrics_compare.csv`
- `daily_metrics_compare.csv`
- `betrayal_metrics_compare.csv`
- `event_equity_compare.csv`
- `event_equity_compare.png`
- `equity_daily_compare.csv`
- `equity_daily_compare.png`

## 7. KOL Selection (Cherry-Pick) Standard

When the displayed curve is `daily_mtm`:

- choose KOLs using daily metrics (`daily_metrics_compare.csv` / `summary_by_kol.csv` daily columns)
- do not rank cherry-pick candidates by event cumulative return

When the displayed curve is `signal_step`:

- use event metrics for ranking/claims

Curve family and metric family must always match.

### Current YouTube Cherry-Pick Set (daily_mtm)

Recommended 5 KOLs for current YouTube presentation set:

1. `Ale_s_World_of_Stocks`
2. `Invest_with_Henry`
3. `Dividend_Data`
4. `The_Maverick_of_Wall_Street`
5. `MarketBeat` (supplementary case)

If only 4 KOLs are needed:

- keep the first 4 and drop `MarketBeat`.

## 8. Quick QA Checklist Before Reporting

1. Open `benchmarks/compare/compare_manifest.json`:
   - verify `event_curve_mode`
   - verify method roots
2. For one sample KOL:
   - if `daily_mtm`: check `event_equity_compare` final equity-1 ~= `daily trained cumulative_return`
   - if `signal_step`: check `event_equity_compare` final equity-1 ~= `event cumulative_return`
3. Confirm method labels in output are `KICL`, `BC`, `IQL`.
