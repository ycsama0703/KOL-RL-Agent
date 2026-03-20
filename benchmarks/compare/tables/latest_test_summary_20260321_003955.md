# Latest Test Summary (Canonical)

- Generated at: 2026-03-21T00:39:55
- Scope: youtube from `benchmarks/compare`; x from `benchmarks/compare/xrefresh`
- Methods: KICL, BC, IQL, TD3BC, CQL, AWAC

## Canonical Source Mapping
- YouTube (17 KOL): `benchmarks/compare/`
- X refresh (11 KOL): `benchmarks/compare/xrefresh/`
- Old X-4 compare (legacy): `benchmarks/compare/x/`

## Key Metrics (mean by source/method)

| source | method | n_kols | event_mean_cumulative_return | event_mean_sharpe | event_mean_max_drawdown | betrayal_mean_reversal_rate | betrayal_mean_entry_violation_rate | betrayal_mean_mean_abs_deviation | daily_trained_mean_cumulative_return | daily_baseline_mean_cumulative_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| x | KICL | 11 | 0.0474 | 0.332 | 0.3447 | 0.0 | 0.0 | 0.0067 | 0.0551 | 0.0055 |
| x | BC | 11 | -0.0237 | -0.1581 | 0.1329 | 0.2044 | 0.2 | 0.0225 | 0.13 | 0.0055 |
| x | IQL | 11 | -0.0225 | -0.1769 | 0.1274 | 0.2186 | 0.1768 | 0.0248 | 0.1299 | 0.0028 |
| x | TD3BC | 11 | -0.0123 | -0.236 | 0.0799 | 0.3991 | 0.792 | 0.181 | -0.0905 | 0.0055 |
| x | CQL | 11 | -0.0088 | -0.0729 | 0.0856 | 0.4311 | 0.8575 | 0.1036 | 0.0913 | 0.0047 |
| x | AWAC | 11 | -0.0006 | -0.0502 | 0.1229 | 0.2059 | 0.1743 | 0.0245 | 0.1429 | 0.0055 |
| youtube | KICL | 17 | 0.3267 | 2.2648 | 0.1716 | 0.0 | 0.0 | 0.008 | 0.313 | 0.3337 |
| youtube | BC | 17 | 0.1428 | 1.7516 | 0.0719 | 0.2411 | 0.2442 | 0.0331 | 0.2904 | 0.3337 |
| youtube | IQL | 17 | 0.1808 | 2.2554 | 0.0733 | 0.2214 | 0.2212 | 0.028 | 0.314 | 0.3337 |
| youtube | TD3BC | 17 | 0.0172 | 0.3209 | 0.056 | 0.4503 | 0.9421 | 0.2087 | -0.0554 | 0.3337 |
| youtube | CQL | 17 | 0.0646 | 0.987 | 0.048 | 0.2871 | 0.9064 | 0.2144 | 0.088 | 0.3346 |
| youtube | AWAC | 17 | 0.2003 | 2.2909 | 0.0729 | 0.2034 | 0.236 | 0.0312 | 0.3188 | 0.3337 |