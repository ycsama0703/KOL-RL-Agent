# Evaluation Metrics (Current Implementation Standard)

Last updated: **March 18, 2026**

This file summarizes the exact metrics currently used in the codebase for model evaluation and benchmark comparison.

Primary code references:

- `train.py` -> `compute_metrics(...)`
- `scripts/evaluate_run.py` -> `compute_betrayal_metrics(...)`

## 1) Return/Risk Metrics

These three metrics are computed by `compute_metrics(daily_returns)` and are used in:

- `event/metrics_test.json`
- `daily/metrics_daily.json`
- `daily_metrics` field inside `event/metrics_test.json`

### 1.1 Cumulative Return

\[
\text{cumulative\_return} = \prod_t (1 + r_t) - 1
\]

- where \(r_t\) is daily portfolio return.
- higher is better.

### 1.2 Sharpe Ratio (annualized)

\[
\text{sharpe} =
\begin{cases}
\frac{\mathbb{E}[r_t]}{\sigma(r_t)}\sqrt{252}, & \sigma(r_t) > 0 \\
0, & \text{otherwise}
\end{cases}
\]

- annualization factor is fixed to 252 trading days.
- higher is better.

### 1.3 Maximum Drawdown (MDD)

Given equity curve \(E_t=\prod_{i \le t}(1+r_i)\), peak \(P_t=\max_{i \le t} E_i\):

\[
\text{MDD} = \max_t \frac{P_t - E_t}{P_t + 10^{-8}}
\]

- lower is better.

## 2) Betrayal Metrics (Intent-Deviation Diagnostics)

Computed in `scripts/evaluate_run.py::compute_betrayal_metrics(...)`, stored in:

- `event/metrics_test.json` -> `betrayal_metrics`

Notation:

- \(a^{base}\): baseline action
- \(a^\pi\): policy action
- `entry_threshold`: signal activation threshold (baseline side)
- `action_threshold`: non-zero action threshold (policy side)

Signal split:

- `has_signal`: \(|a^{base}| \ge \text{entry_threshold}\)
- `no_signal`: complement of `has_signal`

### 2.1 Hard-rule violation rates

- `reversal_rate`: fraction of `has_signal` samples where \(a^{base} \cdot a^\pi < 0\)
- `entry_violation_rate`: fraction of `no_signal` samples where \(|a^\pi| > \text{action_threshold}\)

Lower is better for both.

### 2.2 Violation magnitude

- `reversal_mean_abs_action`: mean \(|a^\pi|\) on reversal samples
- `reversal_mean_abs_delta`: mean \(|a^\pi-a^{base}|\) on reversal samples
- `entry_violation_mean_abs_action`: mean \(|a^\pi|\) on entry-violation samples

Lower is better.

### 2.3 Global deviation

- `mean_abs_deviation`:
  \[
  \mathbb{E}[|a^\pi-a^{base}|]
  \]
- `mean_normalized_deviation` (on `has_signal`):
  \[
  \mathbb{E}\left[\frac{|a^\pi-a^{base}|}{|a^{base}|+10^{-8}}\right]
  \]

Lower is better.

### 2.4 Direction and consistency

- `sign_agreement_rate`: on `has_signal`, fraction with same direction (\(a^{base} \cdot a^\pi > 0\)); higher is better.
- `baseline_policy_corr`: correlation between \(a^{base}\) and \(a^\pi\) on `has_signal`; higher is better.

## 3) Event vs Daily Metric Families

The project uses two metric families:

### 3.1 Event family (`event/metrics_test.json`)

- computed from event/signal-step replay logic.
- includes:
  - `cumulative_return`, `sharpe`, `max_drawdown`
  - `daily_metrics` (a daily aggregation field embedded in event output)
  - `betrayal_metrics`

### 3.2 Daily family (`daily/metrics_daily.json`)

- stored as:
  - `trained: {cumulative_return, sharpe, max_drawdown}`
  - `baseline: {cumulative_return, sharpe, max_drawdown}`
- in current benchmark test runs, daily curves are typically generated with daily mark-to-market (`--daily-price-update`).

## 4) Curve-to-Metric Alignment Rule (Important)

In current compare workflow (`benchmarks/compare`), `event_equity_compare` can be built in two modes:

- `daily_mtm` (default): aligns with `daily_metrics_compare.csv` trained metrics.
- `signal_step`: aligns with `event_metrics_compare.csv` metrics.

Do not mix these when writing claims.

For full compare protocol, see:

- `docs/benchmark_compare_protocol.md`

## 5) Recommended Reporting Bundle (Paper)

Minimum set for each method:

1. Return/Risk: `cumulative_return`, `sharpe`, `max_drawdown`
2. Intent fidelity: `reversal_rate`, `entry_violation_rate`, `mean_abs_deviation`, `baseline_policy_corr`
3. Explicitly state curve mode used in plots (`daily_mtm` or `signal_step`)

