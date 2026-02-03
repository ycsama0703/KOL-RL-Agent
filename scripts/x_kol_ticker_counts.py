#!/usr/bin/env python3
"""Count ticker mentions for a given KOL in an X/Twitter jsonl dump.

- Streams the jsonl (safe for large files).
- Extracts tickers from:
  - tweet.entities.symbols[].text
  - tweet.quoted_tweet.entities.symbols[].text
  - tweet.retweeted_tweet.entities.symbols[].text
  - text fields via cashtag regex ($AAPL, $ES_F, $BRK.B ...)

Outputs:
- Prints summary + top-N to stdout.
- Optionally writes full counts to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


TWITTER_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"

# Cashtag: starts with $, then a letter, then letters/digits/_/./- up to 15 chars total.
CASHTAG_RE = re.compile(r"\\$([A-Za-z][A-Za-z0-9_.\\-]{0,14})")


def parse_utc_date(value: str, *, end_of_day: bool = False) -> datetime:
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        return dt.replace(hour=23, minute=59, second=59)
    return dt


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def get_created_at(obj: Dict[str, Any]) -> datetime | None:
    created = obj.get("created_at") or obj.get("tweet", {}).get("createdAt")
    if not created:
        return None
    try:
        return datetime.strptime(created, TWITTER_TIME_FMT)
    except Exception:
        return None


def in_window(dt: datetime | None, start: datetime | None, end: datetime | None) -> bool:
    if dt is None:
        return False
    if start is not None and dt < start:
        return False
    if end is not None and dt > end:
        return False
    return True


def sanitize_ticker(ticker: str) -> str:
    # keep futures/crypto tickers with underscores; normalize dot to dash for downstream consistency
    return ticker.strip().lstrip("$").upper().replace(".", "-")


def extract_symbols(obj: Dict[str, Any]) -> Iterable[str]:
    tweet = obj.get("tweet", {}) or {}
    for key in ("entities",):
        entities = tweet.get(key) or {}
        for item in entities.get("symbols") or []:
            text = item.get("text")
            if text:
                yield str(text)

    for nested in ("quoted_tweet", "retweeted_tweet"):
        nested_tweet = tweet.get(nested) or {}
        entities = nested_tweet.get("entities") or {}
        for item in entities.get("symbols") or []:
            text = item.get("text")
            if text:
                yield str(text)


def extract_texts(obj: Dict[str, Any]) -> Iterable[str]:
    tweet = obj.get("tweet", {}) or {}
    for key in ("text",):
        val = tweet.get(key)
        if isinstance(val, str) and val:
            yield val
    for nested in ("quoted_tweet", "retweeted_tweet"):
        nested_tweet = tweet.get(nested) or {}
        val = nested_tweet.get("text")
        if isinstance(val, str) and val:
            yield val


def extract_cashtags(text: str) -> Iterable[str]:
    for match in CASHTAG_RE.finditer(text):
        yield match.group(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Count ticker mentions for a KOL in X jsonl.")
    p.add_argument("--input", default="data/x_data/raw/fin1-45.jsonl")
    p.add_argument("--kol", required=True, help="KOL username (kol_username).")
    p.add_argument("--start", default=None, help="Optional inclusive start date (YYYY-MM-DD, UTC).")
    p.add_argument("--end", default=None, help="Optional inclusive end date (YYYY-MM-DD, UTC).")
    p.add_argument("--top", type=int, default=50, help="Top-N tickers to print.")
    p.add_argument("--output-csv", default=None, help="Optional path to write full ticker counts CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    start = parse_utc_date(args.start, end_of_day=False) if args.start else None
    end = parse_utc_date(args.end, end_of_day=True) if args.end else None

    mention_counts: Counter[str] = Counter()
    tweet_counts: Counter[str] = Counter()
    rows = 0
    rows_in_window = 0
    rows_for_kol = 0

    for obj in iter_jsonl(input_path):
        rows += 1
        kol = str(obj.get("kol_username") or obj.get("tweet", {}).get("author", {}).get("userName") or "")
        if kol != args.kol:
            continue
        rows_for_kol += 1

        dt = get_created_at(obj)
        if not in_window(dt, start, end):
            continue
        rows_in_window += 1

        tickers = []
        tickers.extend(extract_symbols(obj))
        for text in extract_texts(obj):
            tickers.extend(extract_cashtags(text))

        norm = [sanitize_ticker(t) for t in tickers if t]
        for t in norm:
            mention_counts[t] += 1

        for t in set(norm):
            tweet_counts[t] += 1

    print(f"input: {input_path}")
    print(f"kol: {args.kol}")
    if start or end:
        print(f"window_utc: {start.date() if start else 'NA'} -> {end.date() if end else 'NA'}")
    print(f"rows_total: {rows}")
    print(f"rows_kol_total: {rows_for_kol}")
    print(f"rows_kol_in_window: {rows_in_window}")
    print(f"unique_tickers: {len(mention_counts)}")
    print("")
    print(f"top_{args.top}: ticker\\tmentions\\ttweets_mentioning")
    for ticker, mentions in mention_counts.most_common(args.top):
        print(f"{ticker}\\t{mentions}\\t{tweet_counts[ticker]}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "mentions", "tweets_mentioning"])
            for ticker, mentions in mention_counts.most_common():
                w.writerow([ticker, mentions, tweet_counts[ticker]])
        print(f"\\nSaved CSV -> {out_path}")


if __name__ == "__main__":
    main()

