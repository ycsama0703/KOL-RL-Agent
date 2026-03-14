"""Prepare YouTube + X datasets with a unified trading_day column.

Mapping rule (America/New_York):
1) pre_market  (t < 09:30)      -> same trading day
2) intraday    (09:30 <= t <16) -> same trading day
3) after_hours (t >= 16:00)     -> next trading day
4) non_trading day              -> next trading day
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from zoneinfo import ZoneInfo


UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Config:
    youtube_input: Path
    x_input: Path
    output_root: Path


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Prepare YouTube + X CSV datasets with unified trading_day.")
    p.add_argument("--youtube-input", default="data/22-25_youtube")
    p.add_argument("--x-input", default="data/x_data/youtube_like_22-25")
    p.add_argument("--output-root", default="data/ready_22-25")
    args = p.parse_args()
    return Config(
        youtube_input=Path(args.youtube_input),
        x_input=Path(args.x_input),
        output_root=Path(args.output_root),
    )


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(days=7 * (n - 1))


def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def observed_fixed_holiday(year: int, month: int, day: int) -> date:
    d = date(year, month, day)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set[date]:
    holidays = set()
    holidays.add(observed_fixed_holiday(year, 1, 1))
    holidays.add(nth_weekday_of_month(year, 1, 0, 3))  # MLK
    holidays.add(nth_weekday_of_month(year, 2, 0, 3))  # Presidents
    holidays.add(easter_sunday(year) - timedelta(days=2))  # Good Friday
    holidays.add(last_weekday_of_month(year, 5, 0))  # Memorial
    holidays.add(observed_fixed_holiday(year, 6, 19))  # Juneteenth
    holidays.add(observed_fixed_holiday(year, 7, 4))  # Independence
    holidays.add(nth_weekday_of_month(year, 9, 0, 1))  # Labor
    holidays.add(nth_weekday_of_month(year, 11, 3, 4))  # Thanksgiving
    holidays.add(observed_fixed_holiday(year, 12, 25))  # Christmas
    return holidays


def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    return d not in nyse_holidays(d.year)


def next_trading_day(d: date) -> date:
    cur = d
    while not is_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def map_to_trading_day(ts_utc: pd.Timestamp, market_open: time, market_close: time) -> date:
    ts_et = ts_utc.tz_convert(ET)
    et_day = ts_et.date()
    local_t = ts_et.timetz().replace(tzinfo=None)
    same_day_is_trading = is_trading_day(et_day)
    if not same_day_is_trading:
        return next_trading_day(et_day)
    if local_t >= market_close:
        return next_trading_day(et_day + timedelta(days=1))
    # pre-market and intraday map to same day
    return et_day


def collect_csvs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.csv") if p.is_file()])


def prepare_one_file(path: Path, out_dir: Path, market_open: time, market_close: time) -> Dict[str, object]:
    df = pd.read_csv(path)
    ts_col = "published_at" if "published_at" in df.columns else "publishedAt" if "publishedAt" in df.columns else None
    if ts_col is None:
        return {"file": path.name, "written": 0, "skipped": int(len(df)), "reason": "missing_timestamp_column"}

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    valid = ts.notna()
    skipped = int((~valid).sum())
    df = df.loc[valid].copy()
    ts = ts.loc[valid]

    if "published_at" not in df.columns:
        df["published_at"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "publishedAt" not in df.columns:
        df["publishedAt"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Canonical cross-source event id; keep backward-compatible input support.
    if "event_id" not in df.columns:
        if "video_id" in df.columns:
            df["event_id"] = df["video_id"]
        else:
            fallback_name = path.stem
            df["event_id"] = [f"{fallback_name}_{i}" for i in range(len(df))]

    df["trading_day"] = [map_to_trading_day(t, market_open, market_close).isoformat() for t in ts]
    df = df.sort_values(["trading_day"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    df.to_csv(out_path, index=False)

    return {
        "file": path.name,
        "written": int(len(df)),
        "skipped": skipped,
        "trading_day_min": str(df["trading_day"].min()) if len(df) else None,
        "trading_day_max": str(df["trading_day"].max()) if len(df) else None,
        "output": str(out_path),
    }


def prepare_folder(source_name: str, in_dir: Path, out_dir: Path, market_open: time, market_close: time) -> Dict[str, object]:
    files = collect_csvs(in_dir)
    results = []
    for f in files:
        results.append(prepare_one_file(f, out_dir, market_open, market_close))
    return {
        "source": source_name,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "file_count": len(files),
        "files": results,
    }


def main() -> None:
    cfg = parse_args()
    market_open = time(9, 30)
    market_close = time(16, 0)

    if not cfg.youtube_input.exists():
        raise SystemExit(f"YouTube input not found: {cfg.youtube_input}")
    if not cfg.x_input.exists():
        raise SystemExit(f"X input not found: {cfg.x_input}")

    out_youtube = cfg.output_root / "youtube"
    out_x = cfg.output_root / "x"

    summary = {
        "task": "prepare_multisource_with_trading_day",
        "config": {
            **asdict(cfg),
            "youtube_input": str(cfg.youtube_input),
            "x_input": str(cfg.x_input),
            "output_root": str(cfg.output_root),
            "market_tz": "America/New_York",
            "mapping_rule": {
                "pre_market_maps_to_same_day": True,
                "intraday_maps_to_same_day": True,
                "after_hours_maps_to_next_day": True,
                "non_trading_day_maps_to_next_day": True,
            },
        },
        "sources": [],
    }

    summary["sources"].append(prepare_folder("youtube", cfg.youtube_input, out_youtube, market_open, market_close))
    summary["sources"].append(prepare_folder("x", cfg.x_input, out_x, market_open, market_close))

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    manifest = cfg.output_root / "manifest_multisource_with_trading_day.json"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Prepared datasets in: {cfg.output_root}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
