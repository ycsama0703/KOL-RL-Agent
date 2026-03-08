#!/usr/bin/env python3
"""Filter X/Twitter jsonl by time window, then export per-KOL jsonl files.

Selection modes:
1) Top-K mode (default):
   - Count tweets per KOL within [start, end]
   - Re-scan and write rows for top-K KOLs into output_dir/<kol>.jsonl
2) Seed-list mode:
   - Read KOL usernames from --kol-list-file
   - Re-scan and write rows for listed KOLs into output_dir/<kol>.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TWITTER_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_dir: Path
    start: datetime
    end: datetime
    top_k: int
    kol_list_file: Path | None


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
    p.add_argument(
        "--kol-list-file",
        default=None,
        help="Optional text file of KOL usernames (one per line). If set, overrides --top-k.",
    )
    args = p.parse_args()

    return Config(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        start=parse_utc_date(args.start, end_of_day=False),
        end=parse_utc_date(args.end, end_of_day=True),
        top_k=int(args.top_k),
        kol_list_file=Path(args.kol_list_file) if args.kol_list_file else None,
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


def iter_jsonl_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise SystemExit(f"Input file is not .jsonl: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix == ".jsonl"])
        if not files:
            raise SystemExit(f"No .jsonl files found under: {input_path}")
        return files
    raise SystemExit(f"Input path not found: {input_path}")


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


def read_kol_list(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"KOL list file not found: {path}")
    names: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            names.append(raw.lstrip("@"))
    # de-dup but preserve order
    seen = set()
    ordered: list[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        ordered.append(n)
    return ordered


def main() -> None:
    cfg = parse_args()
    if not cfg.input_path.exists():
        raise SystemExit(f"Input not found: {cfg.input_path}")
    input_files = iter_jsonl_paths(cfg.input_path)

    counts: Counter[str] = Counter()
    total_in_window = 0
    for src in input_files:
        for _, obj in iter_jsonl(src):
            dt = get_created_at(obj)
            if not in_window(dt, cfg.start, cfg.end):
                continue
            kol = get_kol(obj)
            counts[kol] += 1
            total_in_window += 1

    if not counts:
        raise SystemExit("No rows found in the specified date window.")

    if cfg.kol_list_file:
        selected_kols = read_kol_list(cfg.kol_list_file)
        selected_set = set(selected_kols)
        selection_mode = "seed_list"
    else:
        top = counts.most_common(cfg.top_k)
        selected_kols = [k for k, _ in top]
        selected_set = set(selected_kols)
        selection_mode = "top_k"

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    try:
        written = Counter()
        for src in input_files:
            for raw, obj in iter_jsonl(src):
                dt = get_created_at(obj)
                if not in_window(dt, cfg.start, cfg.end):
                    continue
                kol = get_kol(obj)
                if kol not in selected_set:
                    continue
                if kol not in writers:
                    out_path = cfg.output_dir / f"{safe_filename(kol)}.jsonl"
                    writers[kol] = out_path.open("w", encoding="utf-8")
                writers[kol].write(raw + "\n")
                written[kol] += 1

    finally:
        for fp in writers.values():
            fp.close()

    manifest = {
        "input": str(cfg.input_path),
        "input_files": [str(p) for p in input_files],
        "output_dir": str(cfg.output_dir),
        "start_utc": cfg.start.isoformat(),
        "end_utc": cfg.end.isoformat(),
        "selection_mode": selection_mode,
        "top_k": cfg.top_k,
        "kol_list_file": str(cfg.kol_list_file) if cfg.kol_list_file else None,
        "rows_in_window": total_in_window,
        "selected_kols": [{"kol": kol, "count": int(counts[kol]), "written": int(written[kol])} for kol in selected_kols],
    }
    manifest_name = "manifest_seed_2022_2025.json" if cfg.kol_list_file else "manifest_topk_2022_2025.json"
    (cfg.output_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Window rows: {total_in_window}")
    print(f"Unique KOLs in window: {len(counts)}")
    if cfg.kol_list_file:
        print(f"Exported seed list KOLs into: {cfg.output_dir}")
        for i, kol in enumerate(selected_kols, 1):
            print(f"{i:02d}. {kol}\\tcount={counts[kol]}\\twritten={written[kol]}")
    else:
        print(f"Exported top-{cfg.top_k} KOLs into: {cfg.output_dir}")
        print("Top-K:")
        for i, (kol, n) in enumerate(top, 1):
            print(f"{i:02d}. {kol}\\t{n}\\twritten={written[kol]}")


if __name__ == "__main__":
    main()
