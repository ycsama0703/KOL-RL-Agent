"""Step-02: unify schema for multi-source (YouTube + X) datasets.

Input (default):
  data/multisource_ready_22-25/01_raw_aligned/{youtube,x}/*.csv

Output:
  data/multisource_ready_22-25/02_unified_schema/{youtube,x}/*.csv

Unified columns:
  source_file, platform, event_id, channel_name, published_at, title, text,
  company, confidence, sentiment, trading_day, ticker
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class Config:
    input_root: Path
    output_root: Path


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Prepare step-02 unified schema for YouTube + X.")
    p.add_argument("--input-root", default="data/multisource_ready_22-25/01_raw_aligned")
    p.add_argument("--output-root", default="data/multisource_ready_22-25/02_unified_schema")
    args = p.parse_args()
    return Config(input_root=Path(args.input_root), output_root=Path(args.output_root))


def normalize_ticker(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).strip().upper()
    if not s or s in {"NAN", "NONE", "NULL"}:
        return None
    if s.startswith("$"):
        s = s[1:]
    return s.replace(".", "-")


def unify_one_file(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "published_at" in df.columns:
        published = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    elif "publishedAt" in df.columns:
        published = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
    else:
        published = pd.Series([pd.NaT] * len(df))

    if "text" in df.columns:
        text = df["text"].fillna("").astype(str)
    elif "excerpt" in df.columns:
        text = df["excerpt"].fillna("").astype(str)
    else:
        text = pd.Series([""] * len(df))

    if "title" in df.columns:
        title = df["title"].fillna("").astype(str)
    else:
        title = pd.Series([""] * len(df))

    if "event_id" in df.columns:
        event_id = df["event_id"].fillna("").astype(str)
    elif "video_id" in df.columns:
        event_id = df["video_id"].fillna("").astype(str)
    else:
        event_id = pd.Series([""] * len(df))

    if "channel_name" in df.columns:
        channel = df["channel_name"].fillna("").astype(str)
    else:
        channel = pd.Series([path.stem] * len(df))

    company = df["company"].fillna("").astype(str) if "company" in df.columns else pd.Series([""] * len(df))
    confidence = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0) if "confidence" in df.columns else pd.Series([0.0] * len(df))
    sentiment = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0.0) if "sentiment" in df.columns else pd.Series([0.0] * len(df))

    if "trading_day" in df.columns:
        trading_day = df["trading_day"].astype(str)
    else:
        trading_day = pd.Series([""] * len(df))

    if "ticker" in df.columns:
        ticker = df["ticker"].map(normalize_ticker)
    elif source == "x":
        ticker = company.map(normalize_ticker)
    else:
        ticker = pd.Series([None] * len(df))

    out = pd.DataFrame(
        {
            "source_file": path.name,
            "platform": source,
            "event_id": event_id,
            "channel_name": channel,
            "published_at": published.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "title": title,
            "text": text,
            "company": company,
            "confidence": confidence.astype(float),
            "sentiment": sentiment.astype(float),
            "trading_day": trading_day,
            "ticker": ticker,
        }
    )

    out = out.dropna(subset=["published_at"]).reset_index(drop=True)
    out = out.sort_values(["published_at", "event_id"]).reset_index(drop=True)
    return out


def process_source(source: str, in_dir: Path, out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in in_dir.glob("*.csv") if p.is_file()])
    stats: List[Dict[str, object]] = []

    for path in files:
        before = len(pd.read_csv(path))
        out_df = unify_one_file(path, source)
        out_path = out_dir / path.name
        out_df.to_csv(out_path, index=False)
        stats.append(
            {
                "file": path.name,
                "rows_before": int(before),
                "rows_after": int(len(out_df)),
                "output": str(out_path),
            }
        )
        print(f"{source}/{path.name}: {len(out_df)} rows -> {out_path}")

    return {
        "source": source,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "file_count": len(files),
        "files": stats,
    }


def main() -> None:
    cfg = parse_args()
    if not cfg.input_root.exists():
        raise SystemExit(f"Input root not found: {cfg.input_root}")

    manifest = {
        "task": "prepare_step02_unified_schema",
        "config": {
            **asdict(cfg),
            "input_root": str(cfg.input_root),
            "output_root": str(cfg.output_root),
        },
        "sources": [],
    }

    for source in ["youtube", "x"]:
        in_dir = cfg.input_root / source
        if not in_dir.exists():
            print(f"Skip source={source}: missing {in_dir}")
            continue
        out_dir = cfg.output_root / source
        manifest["sources"].append(process_source(source, in_dir, out_dir))

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.output_root / "manifest_02_unified_schema.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
