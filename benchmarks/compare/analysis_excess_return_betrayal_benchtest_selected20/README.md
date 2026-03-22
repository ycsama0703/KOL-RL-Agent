# Excess-Return vs Betrayal (Experiment 2)

- Pair mode: `intersection`
- Condition mode: `profit_event`
- Universe pairs in detailed CSV: `20`
- Common pairs actually used: `20`

Definition:
- `event_return = weight * reward` (or `policy_action * reward` if `weight` is absent)
- `excess_return_proxy = (policy_action - baseline_action) * reward`
- Condition event: `event_return > 0`
- `betrayal_any = reversal OR entry_violation OR (dev >= 0.200)`

Key files:
- `coverage_summary.csv`
- `excess_return_betrayal_pooled.csv`
- `excess_return_betrayal_bootstrap_ci.csv`
- `excess_return_betrayal_probability_shift.png`