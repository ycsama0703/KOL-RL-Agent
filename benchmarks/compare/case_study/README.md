# Case Study Package

This folder contains all artifacts for the selected KICL case studies:

- `x / Jake__Wujastyk`
- `youtube / Financial_Education`

These two were selected because they jointly satisfy:

1. Positive daily return uplift vs baseline.
2. Zero hard betrayal (`UER=0`, `DRR=0`).
3. Observable portfolio-management behavior (non-trivial OPEN/INCREASE/DECREASE/CLOSE actions).
4. Daily equity curves with interpretable KICL-vs-baseline differences.

## Folder Structure

- `tables/`
  - `case_study_kicl_candidate_scan.csv`
  - `case_study_kicl_curve_diagnostics.csv`
- `case_study_selected_kols_summary.csv`
  - condensed metrics for the final 2 KOLs
- `x/Jake__Wujastyk/`
  - method comparison outputs from `benchmarks/compare/canonical_all/...`
  - includes `equity_daily_compare.png`, `daily_metrics_compare.csv`, `betrayal_metrics_compare.csv`, etc.
- `youtube/Financial_Education/`
  - same structure as above
- `raw_kicl/`
  - raw KICL test artifacts copied from `benchmarks/bench_test_results/multisource_test_mainline_xrefresh`
  - includes:
    - `event/metrics_test.json` (copied)
    - `event/positions_test.csv` (copied)
    - `daily/metrics_daily.json` (copied)
    - `daily/equity_daily.csv` and `daily/equity_daily.png` (copied)

## Suggested Use in Writing

Use this directory as the single source for case-study section:

1. Use `equity_daily_compare.png` for the visual narrative.
2. Use `case_study_selected_kols_summary.csv` for key numbers in text/table.
3. Use `raw_kicl/.../positions_test.csv` to extract concrete action-level examples.
