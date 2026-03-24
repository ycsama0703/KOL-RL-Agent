# Case Study Notes (Auto-generated)

## Scope
- Cases: `x/Jake__Wujastyk`, `youtube/Financial_Education`
- Focus: whether KICL gains are achieved under hard intent constraints while improving portfolio execution.

## Key Observations
- `x/Jake__Wujastyk`: KICL event return `0.462` (rank #1), daily uplift vs baseline `0.181`, `UER=0.000`, `DRR=0.000`, `BD=0.004`.
  Behavior: non-hold ratio `0.076`, active baseline ratio `0.069` -> active policy ratio `0.038`, mean |policy-baseline| `0.0038`.
  Best event-return method on this case is `KICL` (`0.462`); KICL remains hard-consistent (`UER=DRR=0`).
- `youtube/The_Maverick_of_Wall_Street`: KICL event return `0.446` (rank #1), daily uplift vs baseline `0.054`, `UER=0.000`, `DRR=0.000`, `BD=0.003`.
  Behavior: non-hold ratio `0.078`, active baseline ratio `0.081` -> active policy ratio `0.035`, mean |policy-baseline| `0.0031`.
  Best event-return method on this case is `KICL` (`0.446`); KICL remains hard-consistent (`UER=DRR=0`).

## Files
- `tables/case_study_method_snapshot.csv`
- `tables/case_study_kicl_behavior_breakdown.csv`
- `tables/case_study_kicl_top_uplift_windows.csv`

## Suggested Figures
- `x/Jake__Wujastyk/equity_daily_compare.png`
- `youtube/Financial_Education/equity_daily_compare.png`
- `x/Jake__Wujastyk/event_equity_compare.png`
- `youtube/Financial_Education/event_equity_compare.png`

## Stage Uplift Highlights (Top windows)
- `x/Jake__Wujastyk` top 20-day uplift window: `2024-11-21 -> 2024-12-20`, uplift `0.2631` (trained `0.3543` vs baseline `0.0912`).
- `youtube/The_Maverick_of_Wall_Street` top 20-day uplift window: `2024-11-01 -> 2024-12-02`, uplift `0.1379` (trained `0.2301` vs baseline `0.0922`).

