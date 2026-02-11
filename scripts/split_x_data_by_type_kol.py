#!/usr/bin/env python3
"""Split X jsonl data into type-first folders and KOL-specific jsonl files.

Output structure:
  <output_root>/<type>/<kol>_<type>.jsonl

Default target types are: original, retweet, quote.
Records with other types (e.g. reply) are skipped by default and counted.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, TextIO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split X jsonl data by type then by KOL.")
    p.add_argument(
        "--input-dir",
        default="data/22-25_x_data",
        help="Directory containing source jsonl files.",
    )
    p.add_argument(
        "--output-dir",
        default="data/22-25_x_data_by_type",
        help="Output root directory.",
    )
    p.add_argument(
        "--types",
        default="original,retweet,quote",
        help="Comma-separated target types.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max rows to process (0 means all).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=200000,
        help="Print progress every N rows.",
    )
    return p.parse_args()


def infer_type(record: dict) -> str:
    raw = (record.get("tweet_type") or "").strip().lower()
    if raw:
        return raw
    tweet = record.get("tweet") or {}
    if tweet.get("retweeted_tweet") is not None:
        return "retweet"
    if tweet.get("quoted_tweet") is not None:
        return "quote"
    if tweet.get("isReply"):
        return "reply"
    return "original"


def sanitize_name(name: str) -> str:
    name = (name or "UNKNOWN").strip()
    if name.startswith("@"):
        name = name[1:]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "UNKNOWN"


def iter_jsonl_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.jsonl")):
        if path.is_file():
            yield path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_types = [t.strip().lower() for t in args.types.split(",") if t.strip()]
    target_set = set(target_types)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if not target_types:
        raise ValueError("No target types provided.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for t in target_types:
        (output_dir / t).mkdir(parents=True, exist_ok=True)

    handles: Dict[Path, TextIO] = {}
    stats = Counter()

    def get_handle(path: Path) -> TextIO:
        fh = handles.get(path)
        if fh is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = path.open("a", encoding="utf-8")
            handles[path] = fh
        return fh

    processed = 0
    try:
        for src in iter_jsonl_files(input_dir):
            with src.open("r", encoding="utf-8") as f:
                for line in f:
                    if args.limit > 0 and processed >= args.limit:
                        break
                    if not line.strip():
                        continue
                    processed += 1
                    stats["total"] += 1

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        stats["json_error"] += 1
                        continue

                    typ = infer_type(obj)
                    stats[f"type_{typ}"] += 1

                    if typ not in target_set:
                        stats["skipped_non_target_type"] += 1
                        continue

                    kol = sanitize_name(obj.get("kol_username") or "UNKNOWN")
                    out_path = output_dir / typ / f"{kol}_{typ}.jsonl"
                    get_handle(out_path).write(line)
                    stats["written"] += 1

                    if args.progress_every > 0 and processed % args.progress_every == 0:
                        print(
                            f"[progress] processed={processed} written={stats['written']} "
                            f"skipped={stats['skipped_non_target_type']}"
                        )

            if args.limit > 0 and processed >= args.limit:
                break
    finally:
        for fh in handles.values():
            fh.close()

    print("Done.")
    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(f"processed={stats['total']}")
    print(f"written={stats['written']}")
    print(f"skipped_non_target_type={stats['skipped_non_target_type']}")
    for key, value in sorted(stats.items()):
        if key.startswith("type_"):
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
