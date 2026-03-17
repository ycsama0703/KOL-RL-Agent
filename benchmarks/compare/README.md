# Compare Output Folder

Generated comparison outputs are written here by:

`python benchmarks/01_generic_rl/build_compare_report.py`

Key files:

- `summary_by_kol.csv`
- `summary_by_method_mean.csv`
- `overview_event_means.png`
- `overview_betrayal_means.png`

Per-KOL outputs are under:

- `youtube/<KOL>/...`
- `x/<KOL>/...`

Each KOL folder includes:

- `event_equity_compare.csv`
- `event_equity_compare.png`
- `equity_daily_compare.csv`
- `equity_daily_compare.png`

`event_equity_compare` curve mode is configurable:

- `daily_mtm` (default): non-flat mark-to-market daily equity.
- `signal_step`: signal-step cumulative equity from positions.
