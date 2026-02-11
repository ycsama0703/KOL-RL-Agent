#!/usr/bin/env python3
"""Rank KOL influence from X jsonl dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank KOLs by followers and engagement metrics.")
    p.add_argument("--input-dir", default="data/22-25_x_data", help="Directory containing source jsonl files.")
    p.add_argument("--output", default="outputs/analysis/x_kol_influence_ranking.csv", help="Output CSV path.")
    p.add_argument("--top-k", type=int, default=30, help="Print top-k results to stdout.")
    p.add_argument("--w-followers", type=float, default=0.40, help="Weight for followers term.")
    p.add_argument("--w-views", type=float, default=0.25, help="Weight for views term.")
    p.add_argument("--w-likes", type=float, default=0.20, help="Weight for likes term.")
    p.add_argument("--w-retweets", type=float, default=0.10, help="Weight for retweets term.")
    p.add_argument("--w-replies", type=float, default=0.05, help="Weight for replies term.")
    return p.parse_args()


def safe_int(v) -> int:
    try:
        if v is None:
            return 0
        return int(v)
    except Exception:
        return 0


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No jsonl files found under {input_dir}")

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                kol = (obj.get("kol_username") or "UNKNOWN").strip() or "UNKNOWN"
                tweet = obj.get("tweet") or {}
                author = tweet.get("author") or {}

                s = stats[kol]
                s["tweets"] += 1
                s["retweets_sum"] += safe_int(tweet.get("retweetCount"))
                s["replies_sum"] += safe_int(tweet.get("replyCount"))
                s["likes_sum"] += safe_int(tweet.get("likeCount"))
                s["quotes_sum"] += safe_int(tweet.get("quoteCount"))
                s["views_sum"] += safe_int(tweet.get("viewCount"))
                s["bookmarks_sum"] += safe_int(tweet.get("bookmarkCount"))
                s["followers_max"] = max(s.get("followers_max", 0), safe_int(author.get("followers")))

    rows = []
    for kol, s in stats.items():
        tweets = max(1, s["tweets"])
        # Per-post averages
        likes_avg = s["likes_sum"] / tweets
        views_avg = s["views_sum"] / tweets
        retweets_avg = s["retweets_sum"] / tweets
        replies_avg = s["replies_sum"] / tweets

        # Log-scaled composite score
        score = (
            args.w_followers * math.log1p(s["followers_max"])
            + args.w_views * math.log1p(views_avg)
            + args.w_likes * math.log1p(likes_avg)
            + args.w_retweets * math.log1p(retweets_avg)
            + args.w_replies * math.log1p(replies_avg)
        )

        row = {
            "kol_username": kol,
            "tweets": s["tweets"],
            "followers_max": s["followers_max"],
            "views_sum": s["views_sum"],
            "likes_sum": s["likes_sum"],
            "retweets_sum": s["retweets_sum"],
            "replies_sum": s["replies_sum"],
            "quotes_sum": s["quotes_sum"],
            "bookmarks_sum": s["bookmarks_sum"],
            "views_avg": round(views_avg, 4),
            "likes_avg": round(likes_avg, 4),
            "retweets_avg": round(retweets_avg, 4),
            "replies_avg": round(replies_avg, 4),
            "influence_score": round(score, 6),
        }
        rows.append(row)

    rows.sort(key=lambda x: x["influence_score"], reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "kol_username",
        "influence_score",
        "tweets",
        "followers_max",
        "views_sum",
        "likes_sum",
        "retweets_sum",
        "replies_sum",
        "quotes_sum",
        "bookmarks_sum",
        "views_avg",
        "likes_avg",
        "retweets_avg",
        "replies_avg",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved ranking to {output}")
    for i, r in enumerate(rows[: max(1, args.top_k)], start=1):
        print(
            f"{i:>2}. {r['kol_username']:<24} score={r['influence_score']:.6f} "
            f"followers={r['followers_max']} tweets={r['tweets']}"
        )


if __name__ == "__main__":
    main()
