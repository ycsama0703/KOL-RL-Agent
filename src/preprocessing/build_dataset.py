"""Build a cleaned dataset that pairs raw KOL text with sentiment/confidence labels."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

MIN_TEXT_LENGTH = 30
USELESS_PATTERNS = [
    re.compile(r"^\s*\[[^\]]+\]\s*$"),  # e.g. [Music]
    re.compile(r"^\s*https?://", re.IGNORECASE),
]


@dataclass
class Record:
    source_file: str
    platform: str
    video_id: str
    channel_name: str
    published_at: str
    title: str
    text: str
    company: str
    confidence: float
    sentiment: float


def standardize_columns(columns: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for col in columns:
        key = col.strip().lower()
        mapping[key] = col
    return mapping


def pick_text(row: pd.Series, column_map: Dict[str, str]) -> str:
    for candidate in ("transcript", "summary", "description"):
        col = column_map.get(candidate)
        if col is None:
            continue
        value = row.get(col)
        if isinstance(value, str):
            normalized = re.sub(r"\s+", " ", value.strip())
            if normalized:
                return normalized
    return ""


def looks_useless(text: str) -> bool:
    if len(text) < MIN_TEXT_LENGTH:
        return True
    for pattern in USELESS_PATTERNS:
        if pattern.search(text):
            return True
    return False


def detect_platform(path: Path) -> str:
    name = path.name.lower()
    if "youtube" in name:
        return "youtube"
    if "tiktok" in name:
        return "tiktok"
    return "unknown"


def process_file(path: Path) -> List[Record]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - diagnostic print
        print(f"[WARN] Failed to read {path.name}: {exc}")
        return []

    column_map = standardize_columns(df.columns)
    required = ["confidence", "sentiment"]
    for field in required:
        if field not in column_map:
            return []

    platform = detect_platform(path)
    records: List[Record] = []
    for _, row in df.iterrows():
        try:
            confidence = float(row[column_map["confidence"]])
            sentiment = float(row[column_map["sentiment"]])
        except Exception:
            continue
        text = pick_text(row, column_map)
        if not text or looks_useless(text):
            continue

        video_id = str(row.get(column_map.get("video_id", ""), "")).strip()
        channel_name = str(row.get(column_map.get("channel_name", ""), "")).strip()
        published_at = str(row.get(column_map.get("publishedat", ""), "")).strip()
        title = str(row.get(column_map.get("title", ""), "")).strip()
        company = str(row.get(column_map.get("company", ""), "")).strip()

        records.append(
            Record(
                source_file=path.name,
                platform=platform,
                video_id=video_id,
                channel_name=channel_name,
                published_at=published_at,
                title=title,
                text=text,
                company=company,
                confidence=confidence,
                sentiment=sentiment,
            )
        )
    return records


def build_dataset(input_dir: Path) -> pd.DataFrame:
    all_records: List[Record] = []
    for csv_file in sorted(input_dir.glob("*.csv")):
        all_records.extend(process_file(csv_file))
    if not all_records:
        return pd.DataFrame()
    df = pd.DataFrame([record.__dict__ for record in all_records])
    df = df.drop_duplicates(subset=["video_id", "company", "text"])
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    df = df.reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cleaned dataset with sentiment labels.")
    parser.add_argument("--input", default="data/input", help="Directory containing raw CSV files")
    parser.add_argument(
        "--output",
        default="data/processed/kol_text_with_sentiment.csv",
        help="Path to store the processed dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(input_dir)
    if dataset.empty:
        print("No valid records found. Please check input files.")
        return
    dataset.to_csv(output_path, index=False)
    print(f"Saved {len(dataset)} rows to {output_path}")


if __name__ == "__main__":
    main()
