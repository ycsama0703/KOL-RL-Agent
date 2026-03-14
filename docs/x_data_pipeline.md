# X/Twitter Data Processing Pipeline (X-Data)

This document describes the **cleaning and processing steps** used to prepare the X/Twitter dataset for downstream modeling (e.g., ticker extraction, LLM labeling, and RL buffer construction later).

The guiding principles are:
- **Streaming-first** (the raw file can be very large; scripts process line-by-line).
- **Keep raw immutable** (raw stays raw; all outputs go to new folders).
- **Text-centric** (we drop image/media features for now and focus on tweet text).
- **Clear authorship separation** (KOL-authored text is separated from quoted/retweeted context).

---

## Data Locations (Conventions)

Recommended folder layout:
- Raw:
  - `data/x_data/raw/*.jsonl`
- Processed (organization / selection, still JSONL):
  - `data/x_data/processed/`
- Cleaned (text-centric schema for labeling):
  - `data/x_data/cleaned/` (flat layout) or
  - `data/x_data/cleaned_by_type/<kol>/<tweet_type>.jsonl` (by-type layout)

Optional extra outputs:
- Raw split-by-type (raw JSON payload preserved):
  - `data/x_data/raw_by_type/<kol>/<tweet_type>.jsonl`

Each script writes a manifest JSON to document:
- row counts
- date ranges
- input/output paths

---

## Input Schema (Raw JSONL)

Each line is a JSON object. Typical top-level fields include:
- `kol_username` (string)
- `tweet_type` (string, e.g. `tweet`, `quote`, `retweet`, `reply`)
- `created_at` (string, e.g. `"Wed Dec 31 21:47:37 +0000 2025"`)
- `tweet` (object) with nested fields such as:
  - `id`, `url` / `twitterUrl`, `text`, `lang`
  - `entities.symbols` (tickers)
  - optional `quoted_tweet`, `retweeted_tweet` (each may have its own `text`, `entities`, etc.)

Notes:
- Not all lines have every field.
- Some lines may be malformed JSON; scripts skip such lines.

---

## Step A (Optional): Split Raw by KOL + `tweet_type` (No Cleaning)

**Use this step when you want one file per (KOL, tweet_type) but keep the raw payload unchanged.**

Script:
- `scripts/split_x_raw_by_kol_type.py`

Command:
```bash
python scripts/split_x_raw_by_kol_type.py \
  --input data/x_data/raw \
  --output-dir data/x_data/raw_by_type
```

Outputs:
- `data/x_data/raw_by_type/<kol_username>/<tweet_type>.jsonl` (raw JSON objects, unchanged)
- `data/x_data/raw_by_type/manifest_raw_by_type.json` (counts + date ranges)

Performance notes:
- This script maintains an LRU pool of open file handles.
- You can reduce simultaneous open files with:
```bash
python scripts/split_x_raw_by_kol_type.py \
  --input data/x_data/raw \
  --output-dir data/x_data/raw_by_type \
  --max-open-files 64
```

---

## Step B: Select Top-K KOLs in a Date Window (Streaming, Two-Pass)

**Use this step to focus on the most active KOLs within a specific time window (e.g., 2022–2025).**

Script:
- `scripts/process_x_data_topk.py`

What it does:
1) Pass-1: Count tweets per `kol_username` inside `[start, end]`.
2) Choose Top-K by count.
3) Pass-2: Export all tweets for Top-K KOLs into per-KOL `.jsonl` files.

Command (example: Top-20 within 2022-01-01..2025-12-31):
```bash
python scripts/process_x_data_topk.py \
  --input data/x_data/raw \
  --output-dir data/x_data/processed \
  --start 2022-01-01 \
  --end 2025-12-31 \
  --top-k 20
```

Outputs:
- `data/x_data/processed/<kol_username>.jsonl`
- `data/x_data/processed/manifest_topk_2022_2025.json`

Notes:
- This is **selection + organization**, not “cleaning”. The exported JSONL lines are still close to raw.
- The date is parsed from `created_at` / `tweet.createdAt` using the Twitter format:
  - `"%a %b %d %H:%M:%S %z %Y"`

---

## Step C: Clean to a Text-Centric Schema (For LLM Labeling)

**Use this step to remove heavy non-text fields and create a stable, compact schema suitable for LLM-based ticker sentiment labeling.**

Script:
- `scripts/clean_x_data.py`

Core transformations:
- Parse and normalize timestamps into `created_at_utc` (ISO-8601 with `Z`).
- Separate authorship vs context:
  - `text_main` (KOL-authored)
  - `text_quoted` (quoted tweet text, context)
  - `text_retweeted` (retweeted tweet text, context)
- Extract tickers from:
  - `entities.symbols` (and nested quote/retweet entities)
  - cashtags found in text (e.g. `$AAPL`)
- Normalize tickers:
  - upper-case
  - `.` replaced by `-` (e.g., `BF.B` → `BF-B`)

### C1) Clean into “by KOL + by tweet_type” files (recommended)

This creates one file per (KOL, tweet_type), which is convenient if you later want different LLM prompts or labeling workflows by type.

Command (writes to a new directory):
```bash
python scripts/clean_x_data.py \
  --input data/x_data/processed \
  --output-dir data/x_data/cleaned_by_type \
  --layout kol_type \
  --text-mode author
```

Outputs:
- `data/x_data/cleaned_by_type/<kol_username>/<tweet_type>.jsonl`
- `data/x_data/cleaned_by_type/manifest_cleaned.json`

Important:
- This cleaning step is still a generic normalization step.
- If you build filtered modeling datasets, `original`, `reply`, and `quote` should be treated as separate pipelines.
- Current filtered implementation is only defined for original tweets:
  - script: `scripts/filter_x_original_data.py`
  - output: `data/x_data/filtered_data/original/*.jsonl`
  - manifest: `data/x_data/filtered_data/original/manifest_filtered_original.json`

### C3) Aggregate filtered original tweets to US trading days

This step is for daily-frequency modeling. It does not aggregate by ticker. It aggregates by:

- `kol_username`
- `trading_day`

Command:
```bash
python scripts/aggregate_x_original_to_trading_day.py \
  --input data/x_data/filtered_data/original \
  --output-dir data/x_data/daily/original
```

Outputs:
- `data/x_data/daily/original/original_<kol>.jsonl`
- `data/x_data/daily/original/manifest_daily_original.json`

Trading-day mapping rule:
- convert `created_at_utc` to `America/New_York`
- if the post is on a non-trading day, map it to the next NYSE trading day
- if the post is during pre-market (`t < 09:30 ET`), map it to the same trading day
- if the post is during market hours (`09:30 <= t < 16:00 ET`), map it to the same trading day
- if the post is at or after `16:00 ET`, map it to the next NYSE trading day

Aggregation rule:
- group by `(kol_username, trading_day)`
- sort tweets within the day by timestamp
- concatenate same-day `text_main` values into `combined_text`
- preserve the underlying `texts` list, `tweet_ids`, `engagement_sum`, `engagement_max`, and de-duplicated `media`
- preserve per-tweet attribution in `tweets`, including each tweet's own `text`, `tickers`, `engagement`, and `media`

Notes:
- this script uses an in-script NYSE holiday calendar for standard closures
- unscheduled special closures are not modeled

### C2) Clean into a flat directory (backward-compatible)

Command:
```bash
python scripts/clean_x_data.py \
  --input data/x_data/processed \
  --output-dir data/x_data/cleaned \
  --layout flat \
  --text-mode author
```

Outputs:
- `data/x_data/cleaned/*.jsonl`
- `data/x_data/cleaned/manifest_cleaned.json`

### Text modes (`--text-mode`)

This only controls the top-level `text` field; authorship fields remain available regardless.
- `author` (default): `text == text_main` (KOL-authored only). Recommended to avoid attribution confusion.
- `combined`: `text == text_main + quoted + retweeted` joined by blank lines.
- `labeled_combined`: same as combined, but with explicit labels:
  - `AUTHOR_TEXT:`
  - `QUOTED_TEXT (context, not author stance):`
  - `RETWEETED_TEXT (context, not author stance):`

### Disabling quote/retweet context text

If you want to drop quoted/retweeted texts entirely during cleaning:
```bash
python scripts/clean_x_data.py \
  --input data/x_data/processed \
  --output-dir data/x_data/cleaned_by_type \
  --layout kol_type \
  --text-mode author \
  --no-quote-text \
  --no-retweet-text
```

---

## Step D (Analysis Utility): Ticker Mention Counts for One KOL

**Use this step to analyze which tickers a given KOL mentions and how frequently.**

Script:
- `scripts/x_kol_ticker_counts.py`

Example:
```bash
python scripts/x_kol_ticker_counts.py \
  --input data/x_data/raw \
  --kol unusual_whales \
  --top 50 \
  --output-csv data/x_data/processed/unusual_whales_ticker_counts.csv
```

Output columns (CSV):
- `ticker`
- `mentions` (total mention occurrences)
- `tweets_mentioning` (number of distinct tweets that mention ticker)

---

## Cleaned Record Schema (Output)

Each cleaned line is a JSON object with a stable schema:
- `platform`: `"x"`
- `kol_username`: string
- `tweet_id`: string
- `tweet_type`: string (may be `"unknown"` if missing)
- `created_at_utc`: string or null (ISO-8601 with `Z`)
- `lang`: string or null
- `url`: string or null
- `text`: string (depends on `--text-mode`)
- `text_main`: string (KOL-authored)
- `text_quoted`: string or null (context)
- `text_retweeted`: string or null (context)
- `tickers`: list[string] (union of main+context tickers)
- `tickers_main`: list[string]
- `tickers_context`: list[string]

---

## Common Pitfalls / Notes

- **Authorship vs context**:
  - For quote/retweet cases, keep `text_main` as the only “author stance” input unless you intentionally design a different labeling policy.
- **Time normalization**:
  - `created_at_utc` is derived from `created_at` / `tweet.createdAt`. Missing/invalid timestamps become `null`.
- **Ticker normalization**:
  - Replacing `.` with `-` helps downstream compatibility (e.g., Yahoo Finance conventions often prefer `BRK-B` instead of `BRK.B`).
- **Huge raw size**:
  - Prefer the streaming scripts above; avoid loading the full file into memory.
