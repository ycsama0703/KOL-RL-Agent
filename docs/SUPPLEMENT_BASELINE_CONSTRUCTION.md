# Supplement Note: Baseline Construction (Code-Aligned)

This note defines the **baseline policy construction** used in the project, aligned with the current implementation.

Primary code references:
- `scripts/add_baseline_action.py`
- `src/pipeline/replay_utils.py`
- `scripts/build_replay_buffer.py`
- `src/portfolio/layer.py`
- `train.py`
- `scripts/evaluate_run.py`


## 1. Goal of the baseline

The baseline is the executable proxy of KOL intent. It is used in three places:

1. As an anchor action in training/inference (`baseline_actions`).
2. As a comparator in evaluation (`baseline_event_metrics`).
3. As the anchor for betrayal metrics (UER/DRR/BD, etc.).


## 2. Input fields for baseline construction

Per row (ticker-event sample), baseline construction uses:
- `sentiment`
- `confidence`
- `ticker`
- `published_at`


## 3. Baseline raw score

First, construct a bounded score from sentiment and confidence:

\[
\text{baseline\_raw\_score} = \tanh(2 \cdot \text{sentiment} \cdot \text{confidence})
\]

Implementation:
- `scripts/add_baseline_action.py` (lines around raw score creation).


## 4. Day-level signal assembly

For each day, ticker-level raw signal dictionary is assembled as:

\[
\text{raw\_dict}[i] = \text{baseline\_raw\_score}_i \cdot \operatorname{sign}(\text{sentiment}_i)
\]

Implementation:
- `src/pipeline/replay_utils.py`, `annotate_positions()` (raw dict assembly).

Note:
- This is the **as-implemented** formula and should be reported exactly for reproducibility.


## 5. Mapping to executable baseline weights

`raw_dict` is mapped by `PortfolioLayer.allocate(...)` into executable weights.

As implemented, this layer includes:
- previous-day carry (`prev_weights`)
- absolute exposure normalization
- long/short caps
- post-cap re-normalization

Default portfolio constraints (from `PortfolioConfig`):
- `max_long = 0.2`
- `max_short = 0.2`
- `hold_decay = 1.0`

Outputs written back into annotated data:
- `baseline_weight`
- `last_position`
- `silence_days`
- `has_signal`

Implementation:
- `src/pipeline/replay_utils.py`, `annotate_positions()`
- `src/portfolio/layer.py`


## 6. Baseline anchor vs behavior action (important distinction)

Replay buffer stores both baseline anchor and behavior action:

1. **Baseline anchor**:
   - `baseline_actions = baseline_weight`
   - used as intent anchor in model logic.

2. **Behavior action**:
   - `actions = behavior_weight`
   - built by lag/decay smoothing:
     \[
     w^{beh} = w_{t-1} + \alpha (w^{base} - w_{t-1})
     \]
   - where:
     - `alpha = alpha_entry` if `|w_base| >= entry_threshold`
     - else `alpha = decay_exit`

Default behavior parameters:
- `alpha_entry = 0.3`
- `decay_exit = 0.2`
- `entry_threshold = 1e-3`

Implementation:
- `src/pipeline/replay_utils.py`, `build_behavior_weights()`
- `scripts/build_replay_buffer.py` (buffer packing).


## 7. How baseline is used by the policy

Policy is residual around baseline:

\[
a_t^{policy} = a_t^{base} + \delta_t
\]

With hard intent constraints enabled, two rules are applied:
1. No new entry when \(|a_t^{base}| < \text{entry\_threshold}\).
2. No directional reversal against baseline sign.

Default hard-constraint parameters:
- `entry_threshold = 5e-4`
- `clamp_delta = 1.8`

Implementation:
- `train.py`, `apply_intent_constraints()`


## 8. How baseline is evaluated

In evaluation, a zero-delta actor is run under the same decode pipeline:
- `delta = 0`
- policy reduces to baseline-anchored action path
- event metrics saved as `baseline_event_metrics`

Implementation:
- `scripts/evaluate_run.py` (zero actor + baseline metrics block).


## 9. Reproducibility checklist (for appendix)

When reporting baseline in supplementary materials, include:

1. Raw score formula: `tanh(2*sentiment*confidence)`.
2. Day-level signal assembly formula used in code.
3. Portfolio allocation constraints (`max_long/max_short/hold_decay`).
4. Distinction between `baseline_actions` and `actions` (behavior).
5. Residual policy form `a_base + delta`.
6. Hard constraint settings (`entry_threshold`, reversal rule).
7. Baseline evaluation protocol (`delta=0` zero actor).


## 10. Ready-to-paste short paragraph (English)

We construct a KOL-aligned baseline action in two stages. First, each event-level signal is mapped to a bounded raw score as \(\tanh(2 \cdot s \cdot c)\), where \(s\) and \(c\) denote sentiment and confidence. Second, day-level ticker signals are passed through the portfolio allocation layer with carry and exposure constraints to obtain executable baseline weights. The replay buffer stores both baseline anchor actions and smoothed behavior actions; the former serves as the intent anchor, while the latter supports offline fitting. Our policy is residual over baseline, \(a^{\pi}=a^{base}+\delta\), with optional hard admissibility rules (no unsupported entry and no reversal against baseline direction). Baseline evaluation uses the same decoding pipeline with a zero-delta actor to ensure fair event-level comparison.

