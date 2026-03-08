#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parent))

from clean_x_data import (
    build_text,
    extract_media_items,
    extract_tickers_context,
    extract_tickers_main,
    get_kol,
    iso_z,
    iter_jsonl,
    parse_created_at,
    pick_text_field,
    safe_int,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter X original-tweet data into a compact original-only schema.")
    p.add_argument("--input", default="data/x_data/processed_seed/original", help="Directory of original_<kol>.jsonl files.")
    p.add_argument("--output-dir", default="data/x_data/filtered_data/original", help="Output directory for filtered original files.")
    p.add_argument(
        "--text-mode",
        choices=("author", "combined", "labeled_combined"),
        default="author",
        help="How to populate the top-level text field.",
    )
    return p.parse_args()


def build_original_record(obj: Dict[str, Any], text_mode: str) -> Dict[str, Any]:
    tweet = obj.get("tweet") or {}
    created_utc = parse_created_at(obj)
    combined, quoted, retweeted = build_text(
        tweet,
        include_quote_text=True,
        include_retweet_text=True,
    )
    main_text = (tweet.get("text") or "").strip() if isinstance(tweet.get("text"), str) else ""
    context_parts = [p for p in [quoted, retweeted] if p]
    context_text = "\n\n".join(context_parts)
    tickers_main = extract_tickers_main(tweet, main_text)
    tickers_context = extract_tickers_context(tweet, context_text)
    tickers_all = sorted(set(tickers_main + tickers_context))

    return {
        "platform": "x",
        "schema": "filtered_original_v1",
        "kol_username": get_kol(obj),
        "tweet_id": str(tweet.get("id") or ""),
        "tweet_type": obj.get("tweet_type"),
        "created_at_utc": iso_z(created_utc),
        "fetched_at_utc": obj.get("fetched_at_utc"),
        "lang": tweet.get("lang"),
        "url": tweet.get("url") or tweet.get("twitterUrl"),
        "in_reply_to_username": tweet.get("inReplyToUsername"),
        "text": pick_text_field(text_mode, main=main_text, quoted=quoted, retweeted=retweeted),
        "text_main": main_text,
        "text_quoted": quoted,
        "text_retweeted": retweeted,
        "tickers": tickers_all,
        "tickers_main": tickers_main,
        "tickers_context": tickers_context,
        "engagement": {
            "like_count": safe_int(tweet.get("likeCount")),
            "reply_count": safe_int(tweet.get("replyCount")),
            "retweet_count": safe_int(tweet.get("retweetCount")),
            "quote_count": safe_int(tweet.get("quoteCount")),
            "view_count": safe_int(tweet.get("viewCount")),
            "bookmark_count": safe_int(tweet.get("bookmarkCount")),
        },
        "author": {
            "username": tweet.get("author", {}).get("userName"),
            "name": tweet.get("author", {}).get("name"),
            "followers": safe_int(tweet.get("author", {}).get("followers")),
        },
        "media": extract_media_items(obj, tweet),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(input_dir.glob("original_*.jsonl"))
    if not inputs:
        raise SystemExit(f"No original_*.jsonl files found under {input_dir}")

    manifest = {
        "task": "filter_x_original_data",
        "schema": "filtered_original_v1",
        "input": str(input_dir),
        "output_dir": str(output_dir),
        "text_mode": args.text_mode,
        "files": [],
    }

    for in_path in inputs:
        out_path = output_dir / in_path.name
        rows = 0
        date_min = None
        date_max = None
        with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
            for line in src:
                raw = line.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                rec = build_original_record(obj, args.text_mode)
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows += 1
                dt = rec["created_at_utc"]
                if dt is not None:
                    if date_min is None or dt < date_min:
                        date_min = dt
                    if date_max is None or dt > date_max:
                        date_max = dt
        manifest["files"].append(
            {
                "input": str(in_path),
                "output": str(out_path),
                "rows": rows,
                "date_min_utc": date_min,
                "date_max_utc": date_max,
            }
        )

    (output_dir / "manifest_filtered_original.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Filtered {len(inputs)} original file(s) -> {output_dir}")


if __name__ == "__main__":
    main()
