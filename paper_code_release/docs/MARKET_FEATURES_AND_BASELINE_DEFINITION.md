# Market Information Exposure and Baseline Method Definition

This note documents the **actual implemented definitions** in the current codebase for:

1. The market information indicators exposed to the model
2. The baseline method (from sentiment/confidence to executable baseline actions)

All definitions below are code-aligned with:
- `scripts/augment_with_market_data.py`
- `scripts/add_baseline_action.py`
- `src/pipeline/replay_utils.py`
- `scripts/build_replay_buffer.py`
- `train.py`
- `scripts/evaluate_run.py`


## 1) Market Information Exposed to the Model

### 1.1 Exposed market factor list (6 dimensions)

The model is exposed to the following 6 compact market factors (column names kept exactly):

1. `ret_1d`
2. `ret_5d`
3. `vol_5d`
4. `vol_20d`
5. `volu_z_20d`
6. `dist_sma20`

Source of truth:
- `scripts/augment_with_market_data.py` (`MARKET_FACTOR_COLS`)
- `src/pipeline/replay_utils.py` (`MARKET_FEATURE_COLS`)


### 1.2 Exact formula definitions

Let `P_t` be close price at the sample cutoff day `t`, and `r_t = P_t / P_{t-1} - 1`.

1. `ret_1d`
- `P_t / P_{t-1} - 1`

2. `ret_5d`
- `P_t / P_{t-5} - 1`

3. `vol_5d`
- Standard deviation of last 5 daily returns (population std, `ddof=0`)

4. `vol_20d`
- Standard deviation of last 20 daily returns (population std, `ddof=0`)

5. `volu_z_20d`
- Volume z-score over last 20 days:
- `(V_t - mean(V_{t-19:t})) / std(V_{t-19:t})`
- If volume std is ~0, fallback to `0.0`

6. `dist_sma20`
- Distance to 20-day SMA:
- `P_t / SMA20_t - 1`, where `SMA20_t = mean(P_{t-19:t})`


### 1.3 No-future-leakage rule

Factors are computed strictly with historical data up to the sample day cutoff:

- Publish timestamp is converted to date cutoff (same day close series cutoff)
- Historical slice used is `series.loc[:cutoff]`
- Rows are dropped if historical warmup is insufficient:
  - Need at least 21 close points and at least 20 return points for factor construction

Implementation:
- `append_market_factors()` and `compute_market_factors()` in `scripts/augment_with_market_data.py`


### 1.4 How factors enter the state vector

State is built as:

`state = [text_embedding || ticker_embedding || core_scalar_features || market_factors]`

Where:
- `core_scalar_features = [sentiment, confidence, last_position, silence_days]`
- `market_factors = [ret_1d, ret_5d, vol_5d, vol_20d, volu_z_20d, dist_sma20]`

Implementation:
- `build_states()` in `src/pipeline/replay_utils.py`

Important:
- Factors are appended at the tail of the state.
- Ablation can zero out these trailing dims via:
  - `--zero-market-factors`
  - `--market-factor-dim 6`
  in `train.py` and `scripts/evaluate_run.py`.


## 2) Baseline Method: Exact Definition

This section defines baseline at three levels:
- baseline raw score
- executable baseline weight/action
- baseline strategy used in evaluation and comparison


### 2.1 Baseline raw score from sentiment/confidence

For each row:

- `baseline_raw_score = tanh(2 * sentiment * confidence)`

Implementation:
- `scripts/add_baseline_action.py`

This produces a bounded score in `[-1, 1]` before portfolio mapping.


### 2.2 From raw score to executable baseline weight

In replay construction (`annotate_positions()`), per-day raw ticker signal is formed as:

- `raw_dict[ticker] = baseline_raw_score * sign(sentiment)`

Then mapped to executable portfolio weights by `PortfolioLayer.allocate(...)`:

- Uses previous-day weights (`prev_weights`) for continuity
- Supports carry via `hold_decay` (default `1.0`)
- Normalizes by absolute exposure
- Applies per-asset caps (`max_long`, `max_short`)
- Re-normalizes after capping

Key default config (`PortfolioConfig`):
- `max_long = 0.2`
- `max_short = 0.2`
- `hold_decay = 1.0`
- `capital = 10000`

Outputs generated in annotated table:
- `last_position`
- `baseline_weight`
- `silence_days`
- `has_signal`

Implementation:
- `src/pipeline/replay_utils.py` (`annotate_positions`)
- `src/portfolio/layer.py` (`PortfolioLayer`)


### 2.3 Baseline vs behavior action in replay buffer

Replay buffer stores both:

1. `baseline_actions`
- From `baseline_weight`
- This is the intent anchor used by the policy

2. `actions`
- Behavior action (`behavior_weight`) produced by lag/decay smoothing:
  - Entry smoothing: `alpha_entry` (default `0.3`)
  - Exit decay: `decay_exit` (default `0.2`)
  - Signal threshold: `behavior_entry_threshold` (default `1e-3`)

This means:
- Baseline is the anchor intent action.
- Behavior is the dataset action used for BC/IQL fitting.

Implementation:
- `build_behavior_weights()` in `src/pipeline/replay_utils.py`
- buffer packing in `scripts/build_replay_buffer.py`


### 2.4 Baseline inside training/inference policy form

Policy is residual around baseline:

- `a_policy = a_base + delta`

with optional hard intent constraints:

1. No new entry when `|a_base| < entry_threshold` (default `5e-4`)
2. No reversal against baseline direction

Implementation:
- `apply_intent_constraints()` in `train.py`


### 2.5 Baseline curve in evaluation

When evaluating baseline metrics in `scripts/evaluate_run.py`:

- A zero actor (`delta = 0`) is run with the same decoding/constraint pipeline
- Resulting policy action is effectively baseline-anchored action
- Baseline event metrics are saved as `baseline_event_metrics`

So in benchmark tables/plots:
- “Baseline” means the executable KOL-aligned anchor policy under the same replay/evaluation mechanics, not a random or external heuristic.


## 3) Practical Clarifications

1. Price/reward data are still retained for reward and portfolio backtest.
- The 6 market factors are the compact subset exposed as market context to the model.

2. Market factors are fixed-window (5/20) in current code.
- `--price-days` is kept for CLI compatibility but ignored in factor mode.

3. If a row cannot get valid ticker mapping or sufficient market history for factor computation, that row is dropped in enrichment.


## 4) Reproducibility Checklist (what to cite in paper/supplement)

If you need a concise reproducible definition section, include:

1. Factor set and formulas (`ret_1d`, `ret_5d`, `vol_5d`, `vol_20d`, `volu_z_20d`, `dist_sma20`)
2. No-leakage cutoff rule (`hist <= sample day`)
3. State composition (`text emb + ticker emb + 4 core scalars + 6 factors`)
4. Baseline score (`tanh(2 * sentiment * confidence)`)
5. Baseline action mapping via `PortfolioLayer`
6. Residual policy form (`a_base + delta`) and hard constraints
7. Baseline evaluation protocol (`delta=0` baseline actor path)
