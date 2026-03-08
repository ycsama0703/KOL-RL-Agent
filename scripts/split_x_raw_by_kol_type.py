#!/usr/bin/env python3
"""Split X/Twitter raw jsonl by canonical tweet type and KOL.

This is a pure re-organization step:
- Keeps each input JSON object as-is (no field dropping / rewriting)
- Routes each line to: <output_dir>/processed/<type>/<type>_<kol>.jsonl
- Default exported types: original/reply/quote
- Writes a manifest with per-file counts + date ranges (parsed from created_at)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, TextIO


TWITTER_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"


@dataclass
class Config:
    input_path: Path
    output_dir: Path
    max_open_files: int
    target_types: tuple[str, ...]


@dataclass
class FileStats:
    rows: int = 0
    min_dt: datetime | None = None
    max_dt: datetime | None = None


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Split raw X/Twitter jsonl by type/kol.")
    p.add_argument("--input", default="data/x_data/raw", help="Input .jsonl path or directory.")
    p.add_argument("--output-dir", default="data/x_data", help="Output root directory.")
    p.add_argument(
        "--types",
        default="original,reply,quote",
        help="Comma separated canonical types to export. e.g. original,reply,quote",
    )
    p.add_argument(
        "--max-open-files",
        type=int,
        default=128,
        help="Max concurrently open output files (LRU-closed when exceeded).",
    )
    args = p.parse_args()
    target_types = tuple(t.strip().lower() for t in args.types.split(",") if t.strip())
    if not target_types:
        raise SystemExit("No target types provided via --types.")
    return Config(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        max_open_files=int(args.max_open_files),
        target_types=target_types,
    )


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


def iter_jsonl_paths(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise SystemExit(f"Input file is not .jsonl: {input_path}")
        yield input_path
        return
    if not input_path.is_dir():
        raise SystemExit(f"Input path not found: {input_path}")
    files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix == ".jsonl"])
    if not files:
        raise SystemExit(f"No .jsonl files found under: {input_path}")
    for p in files:
        yield p


def safe_segment(name: str, *, fallback: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return fallback
    return re.sub(r"[^A-Za-z0-9_\\-]+", "_", raw)


def get_kol(obj: Dict[str, Any]) -> str:
    return str(obj.get("kol_username") or obj.get("tweet", {}).get("author", {}).get("userName") or "__MISSING_KOL__")


def get_canonical_tweet_type(obj: Dict[str, Any]) -> str:
    raw = str(obj.get("tweet_type") or "").strip().lower()
    tweet = obj.get("tweet", {}) or {}

    # Explicit mapping first.
    if raw in {"tweet", "original"}:
        return "original"
    if raw in {"reply", "comment"}:
        return "reply"
    if raw == "quote":
        return "quote"
    if raw == "retweet":
        return "retweet"

    # Fallback inference when tweet_type missing/other.
    if tweet.get("retweeted_tweet") is not None:
        return "retweet"
    if tweet.get("quoted_tweet") is not None:
        return "quote"
    if tweet.get("isReply"):
        return "reply"
    if raw:
        return raw
    return "original"


def parse_created_at_utc(obj: Dict[str, Any]) -> datetime | None:
    created = obj.get("created_at") or obj.get("tweet", {}).get("createdAt")
    if not created:
        return None
    try:
        return datetime.strptime(created, TWITTER_TIME_FMT).astimezone(timezone.utc)
    except Exception:
        return None


def iso_z(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


class WriterPool:
    def __init__(self, *, max_open_files: int) -> None:
        self.max_open_files = max_open_files
        self._writers: "OrderedDict[Path, TextIO]" = OrderedDict()
        self._initialized_paths: set[Path] = set()

    def get(self, path: Path) -> TextIO:
        w = self._writers.get(path)
        if w is not None:
            self._writers.move_to_end(path)
            return w

        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if path not in self._initialized_paths else "a"
        w = path.open(mode, encoding="utf-8")
        self._initialized_paths.add(path)
        self._writers[path] = w
        self._writers.move_to_end(path)

        while len(self._writers) > self.max_open_files:
            old_path, old_fp = self._writers.popitem(last=False)
            try:
                old_fp.close()
            except Exception:
                pass
        return w

    def close_all(self) -> None:
        for fp in self._writers.values():
            try:
                fp.close()
            except Exception:
                pass
        self._writers.clear()


def main() -> None:
    cfg = parse_args()
    if not cfg.input_path.exists():
        raise SystemExit(f"Input not found: {cfg.input_path}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    writers = WriterPool(max_open_files=cfg.max_open_files)
    stats: Dict[str, FileStats] = {}

    skipped_non_target = 0
    type_counter: Dict[str, int] = {}
    processed_rows = 0
    input_files = [str(p) for p in iter_jsonl_paths(cfg.input_path)]

    try:
        for src_path in input_files:
            for obj in iter_jsonl(Path(src_path)):
                processed_rows += 1
                kol = safe_segment(get_kol(obj), fallback="__MISSING_KOL__")
                ttype = get_canonical_tweet_type(obj)
                type_counter[ttype] = type_counter.get(ttype, 0) + 1

                if ttype not in cfg.target_types:
                    skipped_non_target += 1
                    continue

                out_path = cfg.output_dir / "processed" / ttype / f"{ttype}_{kol}.jsonl"

                dt = parse_created_at_utc(obj)
                key = str(out_path)
                st = stats.get(key)
                if st is None:
                    st = FileStats()
                    stats[key] = st
                st.rows += 1
                if dt is not None:
                    if st.min_dt is None or dt < st.min_dt:
                        st.min_dt = dt
                    if st.max_dt is None or dt > st.max_dt:
                        st.max_dt = dt

                fp = writers.get(out_path)
                fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        writers.close_all()

    manifest = {
        "input": str(cfg.input_path),
        "input_files": input_files,
        "output_dir": str(cfg.output_dir),
        "layout": "processed/type/type_kol.jsonl",
        "target_types": list(cfg.target_types),
        "processed_rows": processed_rows,
        "skipped_non_target_type": skipped_non_target,
        "type_counts_before_filter": dict(sorted(type_counter.items())),
        "files": [
            {"output": out, "rows": st.rows, "date_min_utc": iso_z(st.min_dt), "date_max_utc": iso_z(st.max_dt)}
            for out, st in sorted(stats.items())
        ],
    }
    manifest_path = cfg.output_dir / "processed" / "manifest_raw_by_type.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    print(f"Wrote raw split files -> {cfg.output_dir / 'processed'}")


if __name__ == "__main__":
    main()
