#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_OPEN_ET = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)
TEXT_SEP = "\n\n[DAY_TWEET_SEP]\n\n"


@dataclass(frozen=True)
class Config:
    input_dir: Path
    output_dir: Path
    market_open_hour: int
    market_open_minute: int
    market_close_hour: int
    market_close_minute: int


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Aggregate filtered X original tweets into US trading-day documents.")
    p.add_argument("--input", default="data/x_data/filtered_data/original", help="Directory of original_<kol>.jsonl files.")
    p.add_argument("--output-dir", default="data/x_data/daily/original", help="Output directory for daily aggregated original files.")
    p.add_argument("--market-open-hour", type=int, default=9, help="Market open hour in US/Eastern.")
    p.add_argument("--market-open-minute", type=int, default=30, help="Market open minute in US/Eastern.")
    p.add_argument("--market-close-hour", type=int, default=16, help="Market close hour in US/Eastern.")
    p.add_argument("--market-close-minute", type=int, default=0, help="Market close minute in US/Eastern.")
    args = p.parse_args()
    return Config(
        input_dir=Path(args.input),
        output_dir=Path(args.output_dir),
        market_open_hour=int(args.market_open_hour),
        market_open_minute=int(args.market_open_minute),
        market_close_hour=int(args.market_close_hour),
        market_close_minute=int(args.market_close_minute),
    )


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def observed_fixed_holiday(year: int, month: int, day: int) -> date:
    d = date(year, month, day)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
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


def nyse_holidays(year: int) -> set[date]:
    holidays = {
        observed_fixed_holiday(year, 1, 1),
        nth_weekday(year, 1, 0, 3),   # MLK Day
        nth_weekday(year, 2, 0, 3),   # Presidents Day
        easter_sunday(year) - timedelta(days=2),  # Good Friday
        last_weekday(year, 5, 0),     # Memorial Day
        observed_fixed_holiday(year, 7, 4),
        nth_weekday(year, 9, 0, 1),   # Labor Day
        nth_weekday(year, 11, 3, 4),  # Thanksgiving
        observed_fixed_holiday(year, 12, 25),
    }
    if year >= 2022:
        holidays.add(observed_fixed_holiday(year, 6, 19))  # Juneteenth
    return holidays


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def next_trading_day(d: date) -> date:
    cur = d
    while not is_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def map_to_trading_day(
    ts_utc: datetime,
    market_open: time,
    market_close: time,
) -> tuple[date, date, bool, bool, bool]:
    ts_et = ts_utc.astimezone(ET)
    et_day = ts_et.date()
    local_time = ts_et.timetz().replace(tzinfo=None)
    same_day_is_trading = is_trading_day(et_day)
    was_non_trading = not same_day_is_trading
    intraday = same_day_is_trading and (market_open <= local_time < market_close)
    after_close = same_day_is_trading and local_time >= market_close

    if was_non_trading:
        trading_day = next_trading_day(et_day)
    elif after_close:
        trading_day = next_trading_day(et_day + timedelta(days=1))
    else:
        # pre-market and intraday both map to same trading day
        trading_day = et_day

    return trading_day, et_day, was_non_trading, intraday, after_close


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            yield json.loads(raw)


def combine_engagement_sum(records: List[Dict[str, Any]]) -> Dict[str, int]:
    keys = ["like_count", "reply_count", "retweet_count", "quote_count", "view_count", "bookmark_count"]
    out: Dict[str, int] = {}
    for key in keys:
        out[key] = int(sum((r.get("engagement") or {}).get(key) or 0 for r in records))
    return out


def combine_engagement_max(records: List[Dict[str, Any]]) -> Dict[str, int]:
    keys = ["like_count", "reply_count", "retweet_count", "quote_count", "view_count", "bookmark_count"]
    out: Dict[str, int] = {}
    for key in keys:
        out[key] = int(max([((r.get("engagement") or {}).get(key) or 0) for r in records] or [0]))
    return out


def aggregate_file(in_path: Path, out_path: Path, market_open: time, market_close: time) -> Dict[str, Any]:
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for rec in iter_jsonl(in_path):
        ts_utc = parse_utc(rec.get("created_at_utc"))
        if ts_utc is None:
            skipped += 1
            continue
        trading_day, et_day, was_non_trading, intraday, after_close = map_to_trading_day(
            ts_utc,
            market_open,
            market_close,
        )
        rec["__created_at_utc_dt"] = ts_utc
        rec["__created_at_et"] = ts_utc.astimezone(ET)
        rec["__calendar_day_et"] = et_day.isoformat()
        rec["__trading_day"] = trading_day.isoformat()
        rec["__was_non_trading_day"] = was_non_trading
        rec["__intraday"] = intraday
        rec["__after_close"] = after_close
        key = (str(rec.get("kol_username") or "__MISSING_KOL__"), trading_day.isoformat())
        groups[key].append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for (_, trading_day), items in sorted(groups.items(), key=lambda x: x[0]):
            items.sort(key=lambda r: r["__created_at_utc_dt"])
            texts = [r.get("text_main") or r.get("text") or "" for r in items]
            texts = [t for t in texts if isinstance(t, str) and t.strip()]
            tickers = sorted({t for r in items for t in (r.get("tickers") or []) if t})
            tweet_bundles = []
            media = []
            seen_media = set()
            for r in items:
                tweet_media = []
                tweet_media_seen = set()
                for m in (r.get("media") or []):
                    if not isinstance(m, dict):
                        continue
                    key = (m.get("kind"), m.get("url"))
                    if key not in tweet_media_seen:
                        tweet_media_seen.add(key)
                        tweet_media.append(m)
                    if key in seen_media:
                        continue
                    seen_media.add(key)
                    media.append(m)
                intraday_mapped = bool(r["__intraday"] and r["__trading_day"] != r["__calendar_day_et"])
                tweet_bundles.append(
                    {
                        "tweet_id": r.get("tweet_id"),
                        "created_at_utc": r.get("created_at_utc"),
                        "created_at_et": r["__created_at_et"].isoformat(),
                        "calendar_day_et": r["__calendar_day_et"],
                        "mapped_from_non_trading_day": r["__was_non_trading_day"],
                        # Kept for backward compatibility; under current mapping this should be false.
                        "mapped_from_intraday": intraday_mapped,
                        "is_intraday": r["__intraday"],
                        "mapped_after_close": r["__after_close"],
                        "text": r.get("text_main") or r.get("text"),
                        "tickers": r.get("tickers") or [],
                        "engagement": r.get("engagement") or {},
                        "media": tweet_media,
                    }
                )
            daily = {
                "platform": "x",
                "schema": "daily_original_document_v1",
                "kol_username": items[0].get("kol_username"),
                "tweet_type": "original",
                "trading_day": trading_day,
                "calendar_days_et": sorted({r["__calendar_day_et"] for r in items}),
                "tweet_count": len(items),
                "tweet_ids": [r.get("tweet_id") for r in items if r.get("tweet_id")],
                "created_at_utc_min": items[0]["__created_at_utc_dt"].isoformat().replace("+00:00", "Z"),
                "created_at_utc_max": items[-1]["__created_at_utc_dt"].isoformat().replace("+00:00", "Z"),
                "created_at_et_min": items[0]["__created_at_et"].isoformat(),
                "created_at_et_max": items[-1]["__created_at_et"].isoformat(),
                "mapped_from_non_trading_day_count": sum(1 for r in items if r["__was_non_trading_day"]),
                # Kept for backward compatibility; under current mapping this should be zero.
                "mapped_from_intraday_count": sum(
                    1 for r in items if (r["__intraday"] and r["__trading_day"] != r["__calendar_day_et"])
                ),
                "intraday_count": sum(1 for r in items if r["__intraday"]),
                "mapped_after_close_count": sum(1 for r in items if r["__after_close"]),
                "combined_text": TEXT_SEP.join(texts),
                "texts": texts,
                "tweets": tweet_bundles,
                "tickers": tickers,
                "engagement_sum": combine_engagement_sum(items),
                "engagement_max": combine_engagement_max(items),
                "has_media": any((r.get("media") or []) for r in items),
                "media_count": len(media),
                "media": media,
            }
            f.write(json.dumps(daily, ensure_ascii=False) + "\n")
            rows += 1

    return {
        "input": str(in_path),
        "output": str(out_path),
        "daily_rows": rows,
        "source_rows": int(sum(len(v) for v in groups.values())),
        "skipped_rows_missing_timestamp": skipped,
    }


def main() -> None:
    cfg = parse_args()
    if not cfg.input_dir.exists():
        raise SystemExit(f"Input not found: {cfg.input_dir}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    market_open = time(cfg.market_open_hour, cfg.market_open_minute)
    market_close = time(cfg.market_close_hour, cfg.market_close_minute)

    inputs = sorted(cfg.input_dir.glob("original_*.jsonl"))
    if not inputs:
        raise SystemExit(f"No original_*.jsonl files found under {cfg.input_dir}")

    manifest = {
        "task": "aggregate_x_original_to_trading_day",
        "schema": "daily_original_document_v1",
        "input": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "market_timezone": "America/New_York",
        "market_open_time_et": market_open.strftime("%H:%M"),
        "market_close_time_et": market_close.strftime("%H:%M"),
        "mapping_rule": {
            "pre_market_maps_to_same_trading_day": True,
            "intraday_maps_to_same_trading_day": True,
            "after_close_maps_to_next_trading_day": True,
            "non_trading_day_maps_to_next_trading_day": True,
            "holiday_calendar": "NYSE standard holiday set implemented in-script",
            "special_unscheduled_closures": "not modeled",
        },
        "files": [],
    }

    for in_path in inputs:
        out_path = cfg.output_dir / in_path.name
        manifest["files"].append(aggregate_file(in_path, out_path, market_open, market_close))

    (cfg.output_dir / "manifest_daily_original.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Aggregated {len(inputs)} original file(s) -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
