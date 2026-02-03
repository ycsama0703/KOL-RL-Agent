#!/usr/bin/env python3
"""Filter X/Twitter jsonl by time window, pick top-K KOLs, and export per-KOL jsonl files.

Two-pass streaming:
1) Count tweets per KOL within [start, end]
2) Re-scan and write rows for top-K KOLs into output_dir/<kol>.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple


TWITTER_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_dir: Path
    start: datetime
    end: datetime
    top_k: int


def parse_utc_date(value: str, *, end_of_day: bool = False) -> datetime:
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        return dt.replace(hour=23, minute=59, second=59)
    return dt


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Export per-KOL jsonl for top-K KOLs in a date window.")
    p.add_argument("--input", default="data/x_data/raw/fin1-45.jsonl")
    p.add_argument("--output-dir", default="data/x_data/processed")
    p.add_argument("--start", default="2022-01-01", help="Inclusive start date (YYYY-MM-DD, UTC).")
    p.add_argument("--end", default="2025-12-31", help="Inclusive end date (YYYY-MM-DD, UTC).")
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    return Config(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        start=parse_utc_date(args.start, end_of_day=False),
        end=parse_utc_date(args.end, end_of_day=True),
        top_k=int(args.top_k),
    )


def iter_jsonl(path: Path) -> Iterable[Tuple[str, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            yield raw, obj


def get_kol(obj: Dict) -> str:
    return str(obj.get("kol_username") or obj.get("tweet", {}).get("author", {}).get("userName") or "__MISSING_KOL__")


def get_created_at(obj: Dict) -> datetime | None:
    created = obj.get("created_at") or obj.get("tweet", {}).get("createdAt")
    if not created:
        return None
    try:
        return datetime.strptime(created, TWITTER_TIME_FMT)
    except Exception:
        return None


def in_window(dt: datetime | None, start: datetime, end: datetime) -> bool:
    if dt is None:
        return False
    return start <= dt <= end


def safe_filename(name: str) -> str:
    # Twitter usernames are usually safe, but keep a minimal sanitizer.
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)


def main() -> None:
    cfg = parse_args()
    if not cfg.input_path.exists():
        raise SystemExit(f"Input not found: {cfg.input_path}")

    counts: Counter[str] = Counter()
    total_in_window = 0
    for _, obj in iter_jsonl(cfg.input_path):
        dt = get_created_at(obj)
        if not in_window(dt, cfg.start, cfg.end):
            continue
        kol = get_kol(obj)
        counts[kol] += 1
        total_in_window += 1

    if not counts:
        raise SystemExit("No rows found in the specified date window.")

    top = counts.most_common(cfg.top_k)
    top_kols = [k for k, _ in top]
    top_set = set(top_kols)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    try:
        for kol in top_kols:
            out_path = cfg.output_dir / f"{safe_filename(kol)}.jsonl"
            writers[kol] = out_path.open("w", encoding="utf-8")

        written = Counter()
        for raw, obj in iter_jsonl(cfg.input_path):
            dt = get_created_at(obj)
            if not in_window(dt, cfg.start, cfg.end):
                continue
            kol = get_kol(obj)
            if kol not in top_set:
                continue
            writers[kol].write(raw + "\n")
            written[kol] += 1

    finally:
        for fp in writers.values():
            fp.close()

    manifest = {
        "input": str(cfg.input_path),
        "output_dir": str(cfg.output_dir),
        "start_utc": cfg.start.isoformat(),
        "end_utc": cfg.end.isoformat(),
        "top_k": cfg.top_k,
        "rows_in_window": total_in_window,
        "top_kols": [{"kol": kol, "count": int(counts[kol]), "written": int(written[kol])} for kol in top_kols],
    }
    (cfg.output_dir / "manifest_topk_2022_2025.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Window rows: {total_in_window}")
    print(f"Unique KOLs in window: {len(counts)}")
    print(f"Exported top-{cfg.top_k} KOLs into: {cfg.output_dir}")
    print("Top-K:")
    for i, (kol, n) in enumerate(top, 1):
        print(f"{i:02d}. {kol}\\t{n}\\twritten={written[kol]}")


if __name__ == "__main__":
    main()

