"""Convert X daily_results JSONL into YouTube-like tabular CSVs.

This script bridges:
  data/x_data/daily_results/<KOL>.jsonl
to:
  data/processed/<KOL>.csv-like schema used by the existing RL pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Config:
    input_dir: Path
    output_dir: Path
    start: str | None
    end: str | None
    dedup_mode: str
    text_mode: str
    output_schema: str


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Convert X daily_results jsonl files into YouTube-like CSV format.")
    p.add_argument("--input-dir", default="data/x_data/daily_results")
    p.add_argument("--output-dir", default="data/processed/x_bridge")
    p.add_argument("--start", default=None, help="Inclusive start trading day (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="Inclusive end trading day (YYYY-MM-DD).")
    p.add_argument(
        "--dedup-mode",
        default="last",
        choices=["none", "last", "max_conf", "conf_weighted"],
        help="How to resolve duplicate (trading_day, ticker) rows within one KOL.",
    )
    p.add_argument(
        "--text-mode",
        default="statement",
        choices=["statement", "statement_reasoning"],
        help="Text payload used for `text`/`excerpt` columns.",
    )
    p.add_argument(
        "--output-schema",
        default="processed",
        choices=["processed", "youtube_cleaned"],
        help="Output column schema style.",
    )
    args = p.parse_args()
    return Config(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        start=args.start,
        end=args.end,
        dedup_mode=args.dedup_mode,
        text_mode=args.text_mode,
        output_schema=args.output_schema,
    )


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def in_window(day: str, start: str | None, end: str | None) -> bool:
    if start and day < start:
        return False
    if end and day > end:
        return False
    return True


def normalize_symbol(symbol: str | None) -> str:
    if not symbol:
        return ""
    out = str(symbol).strip().upper()
    if out.startswith("$"):
        out = out[1:]
    return out.replace(".", "-")


def build_text(summary: Dict, mode: str) -> str:
    statement = str(summary.get("statement") or "").strip()
    if mode == "statement":
        return statement
    reasoning = str(summary.get("reasoning") or "").strip()
    if reasoning:
        return f"{statement}\n\nReasoning: {reasoning}" if statement else f"Reasoning: {reasoning}"
    return statement


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def collapse_group(rows: List[Dict], dedup_mode: str) -> Dict:
    if dedup_mode == "none" or len(rows) == 1:
        return rows[-1]

    if dedup_mode == "last":
        return max(rows, key=lambda r: as_float(r.get("_row_order"), 0.0))

    if dedup_mode == "max_conf":
        return max(rows, key=lambda r: (as_float(r.get("confidence"), 0.0), as_float(r.get("_row_order"), 0.0)))

    # conf_weighted
    confs = [max(as_float(r.get("confidence"), 0.0), 0.0) for r in rows]
    sents = [as_float(r.get("sentiment"), 0.0) for r in rows]
    total_w = sum(confs)
    if total_w > 0:
        weighted_sent = sum(c * s for c, s in zip(confs, sents)) / total_w
    else:
        weighted_sent = sum(sents) / max(len(sents), 1)

    best = max(rows, key=lambda r: (as_float(r.get("confidence"), 0.0), as_float(r.get("_row_order"), 0.0)))
    merged = dict(best)
    merged["sentiment"] = float(weighted_sent)
    merged["confidence"] = float(max(confs) if confs else 0.0)
    return merged


def convert_file(path: Path, cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    rows: List[Dict] = []
    row_order = 0
    skipped_empty_symbol = 0
    skipped_invalid_day = 0

    for doc in iter_jsonl(path):
        kol = str(doc.get("kol") or path.stem)
        day = str(doc.get("trading_day") or "").strip()
        if not day:
            skipped_invalid_day += 1
            continue
        if not in_window(day, cfg.start, cfg.end):
            continue

        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            skipped_invalid_day += 1
            continue

        event_id = f"x_{kol}_{day}"
        published_at = f"{day}T00:00:00Z"
        title = f"X Daily Summary - {kol} - {day}"

        summaries = (doc.get("data") or {}).get("summaries") or []
        for s in summaries:
            ticker = normalize_symbol(s.get("symbol"))
            if not ticker:
                skipped_empty_symbol += 1
                continue
            text = build_text(s, cfg.text_mode)
            rows.append(
                {
                    "source_file": path.name,
                    "platform": "x",
                    "event_id": event_id,
                    "channel_name": kol,
                    "published_at": published_at,
                    "publishedAt": published_at,
                    "title": title,
                    "text": text,
                    "excerpt": text,
                    "company": ticker,
                    "ticker": ticker,
                    "confidence": as_float(s.get("confidence"), 0.0),
                    "sentiment": as_float(s.get("sentiment"), 0.0),
                    "_trading_day": day,
                    "_row_order": row_order,
                }
            )
            row_order += 1

    if not rows:
        return pd.DataFrame(), {
            "file": path.name,
            "rows_before_dedup": 0,
            "rows_after_dedup": 0,
            "skipped_empty_symbol": skipped_empty_symbol,
            "skipped_invalid_day": skipped_invalid_day,
        }

    df = pd.DataFrame(rows)

    if cfg.dedup_mode != "none":
        grouped = []
        for _, g in df.groupby(["_trading_day", "ticker"], sort=True):
            grouped.append(collapse_group(g.to_dict("records"), cfg.dedup_mode))
        df = pd.DataFrame(grouped)

    df = df.sort_values(["published_at", "ticker", "_row_order"]).reset_index(drop=True)

    stats = {
        "file": path.name,
        "rows_before_dedup": len(rows),
        "rows_after_dedup": len(df),
        "skipped_empty_symbol": skipped_empty_symbol,
        "skipped_invalid_day": skipped_invalid_day,
    }
    return df, stats


def keep_columns(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    if schema == "youtube_cleaned":
        cols = ["channel_name", "event_id", "publishedAt", "title", "company", "excerpt", "confidence", "sentiment"]
    else:
        cols = [
            "source_file",
            "platform",
            "event_id",
            "channel_name",
            "published_at",
            "title",
            "text",
            "company",
            "ticker",
            "confidence",
            "sentiment",
        ]
    return df[cols]


def main() -> None:
    cfg = parse_args()
    if not cfg.input_dir.exists():
        raise SystemExit(f"Input directory not found: {cfg.input_dir}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(cfg.input_dir.glob("*.jsonl"))
    if not inputs:
        raise SystemExit(f"No jsonl files found in: {cfg.input_dir}")

    manifest = {
        "task": "convert_x_daily_results_to_youtube_like",
        "config": {
            **asdict(cfg),
            "input_dir": str(cfg.input_dir),
            "output_dir": str(cfg.output_dir),
        },
        "files": [],
    }

    for path in inputs:
        df, stats = convert_file(path, cfg)
        if df.empty:
            print(f"Skip {path.name} (no valid rows)")
            manifest["files"].append(stats)
            continue

        if cfg.output_schema == "youtube_cleaned":
            out_name = f"{path.stem}_companies_cleaned.csv"
        else:
            out_name = path.with_suffix(".csv").name
        out_path = cfg.output_dir / out_name
        keep_columns(df, cfg.output_schema).to_csv(out_path, index=False)
        stats["output"] = str(out_path)
        print(f"{path.name}: {stats['rows_after_dedup']} rows -> {out_path}")
        manifest["files"].append(stats)

    manifest_path = cfg.output_dir / "manifest_x_to_youtube_like.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
