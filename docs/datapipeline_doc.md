# Data Pipeline

This document is the current source of truth for the experiment-ready data pipeline used in this repository.

It covers:
- the upstream YouTube route
- the upstream X/Twitter route
- the unified multi-source training pipeline under `data/multisource_ready_22-25`
- both evaluation branches:
  - `mainline` / event-driven reward-buffer line
  - `dailyline` / daily mark-to-market reward-buffer line

---

## 1. Current experiment-ready root

The current working root for training and testing is:

- `data/multisource_ready_22-25`

Its numbered subfolders represent the pipeline stages:

1. `00_mapped_full`
2. `01_raw_aligned`
3. `02_unified_schema`
4. `03_splits`
5. `04_clean`
6. `05_embeddings`
7. `06_enriched`
8. `07_reward`
9. `07_reward_daily`
10. `08_replay_buffer`
11. `08_replay_buffer_daily`

All stages keep source separation:

- `.../youtube/...`
- `.../x/...`

---

## 2. Upstream source routes

### 2.1 YouTube route

Current YouTube source data lives in:

- `data/22-25_youtube/*.csv`

These CSVs are already structured KOL-company mention data. Typical fields include:

- `channel_name`
- `video_id`
- `publishedAt` or `published_at`
- `title`
- `company`
- `text` or `excerpt`
- `confidence`
- `sentiment`

This means the current experiment pipeline starts from already-extracted YouTube mention tables, not from raw transcripts.

### 2.2 X/Twitter route

Current raw shards live in:

- `data/x_data/raw/fin1-45.jsonl`
- `data/x_data/raw/fin46-85_5y.jsonl`
- `data/x_data/raw/fin86-140_5y.jsonl`
- `data/x_data/raw/fin141-178_5y.jsonl`

Current seed list:

- `config/x_kol_seed_list.txt`

Current raw-X preparation scripts include:

- `scripts/split_x_raw_by_kol_type.py`
- `scripts/process_x_data_topk.py`
- `scripts/clean_x_data.py`
- `scripts/aggregate_x_original_to_trading_day.py`

The X route is now partially separated into two layers:

1. raw / cleaned / by-type preprocessing for Twitter-native analysis
2. a summarized "YouTube-like" table used by the RL pipeline

The RL-facing X input currently used by the multi-source pipeline is:

- `data/x_data/youtube_like_22-25`

That directory contains X data already converted into the same row-wise schema style as the YouTube mention tables.

---

## 3. Trading-day mapping rule

The shared mapping rule is implemented by:

- `scripts/prepare_multisource_with_trading_day.py`

Market timezone:

- `America/New_York`

Mapping logic:

1. `pre_market`: `t < 09:30 ET` -> same trading day
2. `intraday`: `09:30 <= t < 16:00 ET` -> same trading day
3. `after_hours`: `t >= 16:00 ET` -> next trading day
4. `non_trading_day` -> next NYSE trading day

This rule is used to create `trading_day`, which is the canonical date key for downstream splitting, reward generation, and replay construction.

---

## 4. Multi-source pipeline: stage-by-stage

### Stage 00: map both sources to `trading_day`

Script:

- `scripts/prepare_multisource_with_trading_day.py`

Typical command:

```bash
python scripts/prepare_multisource_with_trading_day.py \
  --youtube-input data/22-25_youtube \
  --x-input data/x_data/youtube_like_22-25 \
  --output-root data/multisource_ready_22-25
```

Current output folder:

- `data/multisource_ready_22-25/00_mapped_full`

Purpose:

- add a unified `trading_day`
- normalize `event_id`
- keep YouTube and X separated but aligned to the same date convention

Manifest:

- `data/multisource_ready_22-25/00_mapped_full/manifest_00_mapped_full.json`

### Stage 01: clip to the common experiment window

Current output folder:

- `data/multisource_ready_22-25/01_raw_aligned`

Current clipping rule:

- keep rows with `trading_day <= 2025-12-31`

Purpose:

- remove overhang rows outside the shared experiment window
- make YouTube and X temporally comparable before schema unification

Manifest:

- `data/multisource_ready_22-25/01_raw_aligned/manifest_01_raw_aligned.json`

Note:

- this is currently an existing materialized step in the repo outputs
- the clipping logic is straightforward, but there is no dedicated standalone checked-in script for this step yet

### Stage 02: unify schema

Script:

- `scripts/prepare_step02_unified_schema.py`

Typical command:

```bash
python scripts/prepare_step02_unified_schema.py \
  --input-root data/multisource_ready_22-25/01_raw_aligned \
  --output-root data/multisource_ready_22-25/02_unified_schema
```

Unified columns:

- `source_file`
- `platform`
- `event_id`
- `channel_name`
- `published_at`
- `title`
- `text`
- `company`
- `confidence`
- `sentiment`
- `trading_day`
- `ticker`

Purpose:

- make YouTube and X use one common tabular interface
- ensure all later scripts can recurse over the same directory structure

Manifest:

- `data/multisource_ready_22-25/02_unified_schema/manifest_02_unified_schema.json`

### Stage 03: chronological split by trading day

Script:

- `scripts/prepare_step03_daily_split.py`

Typical command:

```bash
python scripts/prepare_step03_daily_split.py \
  --input-root data/multisource_ready_22-25/02_unified_schema \
  --output-root data/multisource_ready_22-25/03_splits \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2
```

Current split rule:

- chronological split by `trading_day`
- all rows from the same `trading_day` stay in the same split

Output layout:

- `data/multisource_ready_22-25/03_splits/<source>/<KOL>/train.csv`
- `data/multisource_ready_22-25/03_splits/<source>/<KOL>/val.csv`
- `data/multisource_ready_22-25/03_splits/<source>/<KOL>/test.csv`

Manifest:

- `data/multisource_ready_22-25/03_splits/manifest_03_splits.json`

### Stage 04: clean text and de-duplicate rows

Script:

- `scripts/clean_dataset.py`

Typical command:

```bash
python scripts/clean_dataset.py \
  --input data/multisource_ready_22-25/03_splits \
  --output data/multisource_ready_22-25/04_clean \
  --min_length 50
```

Current behavior:

- clean `text`
- normalize `company`
- remove stop companies
- drop duplicates on available subset of:
  - `event_id`
  - `video_id`
  - `company`
  - `text`

Purpose:

- remove low-information text
- standardize company strings before ticker mapping

### Stage 05: text embeddings

Script:

- `scripts/generate_embeddings.py`

Current experiment setting:

- model: `Qwen/Qwen3-Embedding-4B`
- output dimension: `1024`

Typical command:

```bash
python scripts/generate_embeddings.py \
  --model Qwen/Qwen3-Embedding-4B \
  --input data/multisource_ready_22-25/04_clean \
  --output data/multisource_ready_22-25/05_embeddings \
  --batch-size 2 \
  --device cuda \
  --trust-remote-code \
  --padding-side left \
  --torch-dtype float16 \
  --max-length 8192 \
  --normalize \
  --output-dim 1024 \
  --log-file logs/embed_qwen4b_d1024.log \
  --log-level INFO \
  --no-progress-bar
```

Output:

- mirrored `.pt` tensors under `data/multisource_ready_22-25/05_embeddings/<source>/<KOL>/...`

### Stage 06: enrich with ticker mapping and market factors

Script:

- `scripts/augment_with_market_data.py`

Typical commands:

```bash
python scripts/augment_with_market_data.py \
  --input data/multisource_ready_22-25/04_clean/youtube \
  --embeddings data/multisource_ready_22-25/05_embeddings/youtube \
  --output data/multisource_ready_22-25/06_enriched/youtube

python scripts/augment_with_market_data.py \
  --input data/multisource_ready_22-25/04_clean/x \
  --embeddings data/multisource_ready_22-25/05_embeddings/x \
  --output data/multisource_ready_22-25/06_enriched/x
```

Current responsibilities:

- map `company -> ticker`
- attach embedding columns
- download / align market data
- keep price data for reward generation
- expose a compact market-information window to the model

Current model-exposed market factors:

- `ret_1d`
- `ret_5d`
- `vol_5d`
- `vol_20d`
- `volu_z_20d`
- `dist_sma20`

Important note:

- the model no longer uses the old raw 5-day price window as its explicit market input
- price data is still fetched because reward computation and portfolio accounting require it

### Stage 07A: event-driven reward branch (`mainline`)

Scripts:

- `scripts/generate_reward.py`
- `scripts/add_baseline_action.py`

Typical commands:

```bash
python scripts/generate_reward.py \
  --input data/multisource_ready_22-25/06_enriched \
  --output data/multisource_ready_22-25/07_reward

python scripts/add_baseline_action.py \
  --input data/multisource_ready_22-25/07_reward \
  --output data/multisource_ready_22-25/07_reward
```

Purpose:

- compute event-driven rewards
- add `baseline_raw_score`
- preserve `next_date` and `done`

This is the default event-driven line used by the current mainline training/testing route.

### Stage 07B: daily mark-to-market reward branch (`dailyline`)

Scripts:

- `scripts/generate_reward_daily.py`
- `scripts/add_baseline_action.py`

Typical commands:

```bash
python scripts/generate_reward_daily.py \
  --input data/multisource_ready_22-25/06_enriched \
  --output data/multisource_ready_22-25/07_reward_daily

python scripts/add_baseline_action.py \
  --input data/multisource_ready_22-25/07_reward_daily \
  --output data/multisource_ready_22-25/07_reward_daily
```

Purpose:

- compute next-trading-day return rewards
- support the daily mark-to-market branch
- later pair with replay construction using `--next-state-mode next_date`

### Stage 08A: event-driven replay buffer (`mainline`)

Scripts:

- `scripts/build_ticker_embedding.py`
- `scripts/build_replay_buffer.py`

Typical commands:

```bash
python scripts/build_ticker_embedding.py \
  --input data/multisource_ready_22-25/07_reward \
  --vocab-path models/embedding/multisource_22-25_ticker_vocab.json \
  --embedding-path models/embedding/multisource_22-25_ticker_embedding.pt

python scripts/build_replay_buffer.py \
  --reward-dir data/multisource_ready_22-25/07_reward \
  --output-dir data/multisource_ready_22-25/08_replay_buffer \
  --ticker-vocab models/embedding/multisource_22-25_ticker_vocab.json \
  --ticker-embedding models/embedding/multisource_22-25_ticker_embedding.pt \
  --next-state-mode next_event
```

### Stage 08B: daily replay buffer (`dailyline`)

Scripts:

- `scripts/build_ticker_embedding.py`
- `scripts/build_replay_buffer.py`

Typical commands:

```bash
python scripts/build_ticker_embedding.py \
  --input data/multisource_ready_22-25/07_reward_daily \
  --vocab-path models/embedding/multisource_22-25_daily_ticker_vocab.json \
  --embedding-path models/embedding/multisource_22-25_daily_ticker_embedding.pt

python scripts/build_replay_buffer.py \
  --reward-dir data/multisource_ready_22-25/07_reward_daily \
  --output-dir data/multisource_ready_22-25/08_replay_buffer_daily \
  --ticker-vocab models/embedding/multisource_22-25_daily_ticker_vocab.json \
  --ticker-embedding models/embedding/multisource_22-25_daily_ticker_embedding.pt \
  --next-state-mode next_date
```

Purpose:

- build the dailyline replay buffer
- connect state transitions by the next trading date rather than the next mention event

Current replay buffer keys:

- `states`
- `rewards`
- `portfolio_rewards`
- `actions`
- `baseline_actions`
- `next_baseline_action`
- `next_states`
- `dones`
- `meta`

Current `meta` keys:

- `baseline_raw_score`
- `event_id`
- `published_at`
- `ticker`

---

## 5. What the two training branches mean

### `mainline`

Root:

- `data/multisource_ready_22-25/08_replay_buffer`

Meaning:

- event-driven transitions
- the next state follows the next mention event
- this is the current non-daily baseline branch

### `dailyline`

Root:

- `data/multisource_ready_22-25/08_replay_buffer_daily`

Meaning:

- daily mark-to-market reward branch
- replay links use `next_date`
- designed to support daily portfolio-value updates between sparse KOL signals

---

## 6. Notes on current implementation choices

### 6.1 Model-visible market information

Current market factors exposed to the model are intentionally compact:

- `ret_1d`
- `ret_5d`
- `vol_5d`
- `vol_20d`
- `volu_z_20d`
- `dist_sma20`

The design goal is:

- market information should assist KOL-intent completion
- market information should not dominate the policy and turn it into a purely market-driven trader

### 6.2 Baseline action

`baseline_raw_score` is added after reward generation and is used later by:

- replay construction
- regime split
- signal / silence statistics
- training and evaluation

### 6.3 Source mixing

At the replay stage, YouTube and X are kept in separate subdirectories:

- `.../08_replay_buffer/youtube/<KOL>/...`
- `.../08_replay_buffer/x/<KOL>/...`

This is the correct current behavior. The training scripts can recurse over both sources from the same multi-source root.

---

## 7. Legacy and non-primary docs

The following documents are still useful, but they are no longer the authoritative description of the current experiment-ready pipeline:

- `docs/x_data_pipeline.md`
- `docs/report_prep.md`
- `README.md`

For the latest end-to-end multi-source experiment pipeline, use this file first.
