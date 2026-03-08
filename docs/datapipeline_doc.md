# Data Pipeline (YouTube + X)

This document is the **single source of truth** for the current dual-source data workflow:
- YouTube KOL data route
- X/Twitter KOL data route

Both routes can feed downstream analysis and, after transformation, the RL pipeline.

---

## 1) YouTube Route

### 1.1 Upstream collection and extraction

1. Use YouTube keyword search (e.g., `US stock analysis`) to build candidate KOL list.
2. Use `video_count.py` to filter active channels in a time range.
3. Use `summary_pipeline.fetch_video_ids` + `summary_pipeline.fetch_video_details` to collect metadata.
4. Use `summary_pipeline.download_transcript` to get transcript text.
5. Use LLM extraction (`summary_pipeline.find_company_llm`) to output:
   - `company`
   - `excerpt`
   - `confidence`
   - `sentiment`

Typical generated raw table fields:
- `channel_name`, `video_id`, `publishedAt`, `title`, `company`, `excerpt`, `confidence`, `sentiment`

Current dataset example:
- `data/22-25_youtube/*.csv`

### 1.2 Training-side preprocessing (legacy RL route)

Main route used by training scripts:

1. Split by time/video:
   - `scripts/split_by_video_time.py`
   - or `scripts/split_by_global_time.py`
2. Clean text/company:
   - `scripts/clean_dataset.py`
3. Generate embeddings:
   - `scripts/generate_embeddings.py`
4. Map `company -> ticker`, append price history:
   - `scripts/augment_with_market_data.py`
5. Generate reward:
   - `scripts/generate_reward.py`
6. Build replay artifacts:
   - `scripts/run_replay_pipeline.py`
   - `scripts/build_replay_buffer.py`

Important note:
- **Ticker completion is done in `augment_with_market_data.py` via `CompanyTickerMapper`**, not in `clean_dataset.py`.

---

## 2) X/Twitter Route

### 2.1 Seed KOL list (fixed list mode)

Seed list file:
- `config/x_kol_seed_list.txt`

Current seed list (20):
- `ACInvestorBlog`
- `Jake__Wujastyk`
- `bespokeinvest`
- `Stephanie_Link`
- `ripster47`
- `StockCats_2009`
- `MrZackMorris`
- `StockSavvyShay`
- `goldseek`
- `TheLastDegree`
- `intocryptoverse`
- `Trader_Dante`
- `GoldTelegraph_`
- `EliteOptions2`
- `Stocktwits`
- `traderstewie`
- `Mr_Derivatives`
- `WOLF_Financial`
- `CryptoCapo_`
- `davidfaber`

### 2.2 Raw input

Current raw shards:
- `data/x_data/raw/fin1-45.jsonl`
- `data/x_data/raw/fin46-85_5y.jsonl`
- `data/x_data/raw/fin86-140_5y.jsonl`
- `data/x_data/raw/fin141-178_5y.jsonl`

### 2.3 X processing steps

#### Step A (optional): raw split by type + KOL (no cleaning)

Script:
- `scripts/split_x_raw_by_kol_type.py`

Current output structure (type-first):
- `<output_dir>/processed/<type>/<type>_<kol>.jsonl`

Example:
```bash
python scripts/split_x_raw_by_kol_type.py \
  --input data/x_data/raw \
  --output-dir data/x_data \
  --types original,reply,quote
```

Output:
- `data/x_data/processed/original/original_<kol>.jsonl`
- `data/x_data/processed/reply/reply_<kol>.jsonl`
- `data/x_data/processed/quote/quote_<kol>.jsonl`
- `data/x_data/processed/manifest_raw_by_type.json`

#### Step B: export per-KOL JSONL in a date window (seed list mode)

Script:
- `scripts/process_x_data_topk.py`

Use seed list instead of top-k:
```bash
python scripts/process_x_data_topk.py \
  --input data/x_data/raw/fin1-45.jsonl \
  --output-dir data/x_data/processed_seed \
  --start 2022-01-01 \
  --end 2025-12-31 \
  --kol-list-file config/x_kol_seed_list.txt
```

Output:
- `data/x_data/processed_seed/<kol>.jsonl`
- `data/x_data/processed_seed/manifest_seed_2022_2025.json`

#### Step C: clean to text-centric schema for LLM labeling

Script:
- `scripts/clean_x_data.py`

Recommended (KOL + tweet_type layout):
```bash
python scripts/clean_x_data.py \
  --input data/x_data/processed_seed \
  --output-dir data/x_data/cleaned_by_type \
  --layout kol_type \
  --text-mode author
```

Output:
- `data/x_data/cleaned_by_type/<kol>/<tweet_type>.jsonl`
- `data/x_data/cleaned_by_type/manifest_cleaned.json`

Notes:
- `text_mode=author` means `text` keeps only KOL-authored text.
- `text_quoted` and `text_retweeted` are still preserved as context fields.
- For filtered datasets, do not assume `original`, `reply`, and `quote` share the same downstream schema.
- Current filtered implementation is only formalized for `original` via `scripts/filter_x_original_data.py`.
- See `data/x_data/filtered_data/README.md` for the current split-by-type policy.

#### Step D: aggregate filtered original tweets to US trading days

Script:
- `scripts/aggregate_x_original_to_trading_day.py`

Command:
```bash
python scripts/aggregate_x_original_to_trading_day.py \
  --input data/x_data/filtered_data/original \
  --output-dir data/x_data/daily/original
```

Output:
- `data/x_data/daily/original/original_<kol>.jsonl`
- `data/x_data/daily/original/manifest_daily_original.json`

Mapping rule:
- convert `created_at_utc` to `America/New_York`
- if a tweet falls on a non-trading day, map it to the next NYSE trading day
- if a tweet falls before `09:30 ET`, map it to the same trading day
- if a tweet falls during market hours (`09:30 <= t < 16:00 ET`), map it to the same trading day
- if a tweet falls at or after `16:00 ET`, map it to the next NYSE trading day

Aggregation unit:
- `(kol_username, trading_day)`

Aggregation behavior:
- concatenate same-day `text_main` values in chronological order into `combined_text`
- preserve `texts`, `tweet_ids`, `engagement_sum`, `engagement_max`, and de-duplicated `media`
- preserve tweet-to-media attribution via a `tweets` list, where each item keeps that tweet's own text, metadata, and media

### 2.4 X analysis helper

Ticker mention frequency for one KOL:
```bash
python scripts/x_kol_ticker_counts.py \
  --input data/x_data/raw/fin1-45.jsonl \
  --kol unusual_whales \
  --top 50 \
  --output-csv data/x_data/processed/unusual_whales_ticker_counts.csv
```

---

## 3) Two-route status summary

- YouTube route currently aligns with RL training pipeline (CSV -> split/clean -> embedding -> enrich/reward -> replay buffer).
- X route currently supports:
  - seed-list selection
  - raw/type split
  - text-centric cleaning
  - EDA/statistics
- X route is not yet fully wired into the same replay-buffer generation path as YouTube.

---

## 4) Recommended practice

1. Keep source-separated storage:
   - YouTube: `data/22-25_youtube`
   - X: `data/x_data/...`
2. Keep seed list file versioned:
   - `config/x_kol_seed_list.txt`
3. For paper reproducibility, always store generated manifests:
   - `manifest_raw_by_type.json`
   - `manifest_seed_2022_2025.json`
   - `manifest_cleaned.json`
