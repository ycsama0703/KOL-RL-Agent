"""Clean processed channel datasets (text + company normalization)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.preprocessing.text_cleaner import TextCleaner


DEFAULT_STOP_COMPANIES = [
    "marketbeat",
    "everything money",
    "invest with henry",
    "market beat",
    "marketbeat.com",
]


def normalize_company(name: str) -> str:
    if not isinstance(name, str):
        return ""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"\.com\b", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9&\-\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def should_drop_company(name: str, stoplist: Iterable[str]) -> bool:
    if not name:
        return True
    for stop in stoplist:
        if stop and name == stop:
            return True
    return False


def clean_file(
    path: Path,
    input_root: Path,
    output_root: Path,
    cleaner: TextCleaner,
    min_length: int,
    stoplist: List[str],
) -> None:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        print(f"[WARN] {path} missing `text` column; skipping.")
        return
    df["text"] = df["text"].astype(str).map(cleaner.clean)
    df = df[df["text"].str.len() >= min_length]

    if "company" in df.columns:
        df["company"] = df["company"].astype(str)
        df["company"] = df["company"].map(normalize_company)
        df = df[~df["company"].apply(lambda name: should_drop_company(name, stoplist))]

    subset_cols = [col for col in ["video_id", "company", "text"] if col in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)
    df = df.reset_index(drop=True)

    try:
        relative = path.relative_to(input_root)
    except ValueError:
        relative = path.name
    output_path = output_root / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned {path} -> {output_path} ({len(df)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean processed CSV datasets.")
    parser.add_argument(
        "--input",
        default="data/processed/splits",
        help="Directory containing channel CSV files or subdirectories.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/clean",
        help="Directory to store cleaned CSVs while preserving structure.",
    )
    parser.add_argument("--min_length", type=int, default=50, help="Minimum text length.")
    parser.add_argument(
        "--stop_companies",
        default=",".join(DEFAULT_STOP_COMPANIES),
        help="Comma separated list of company names to drop after normalization.",
    )
    parser.add_argument(
        "--keep_case",
        action="store_true",
        help="Do not lowercase text (default lowercase).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    stoplist = [item.strip().lower() for item in args.stop_companies.split(",") if item.strip()]
    cleaner = TextCleaner(lowercase=not args.keep_case)

    files: List[Path] = []
    if input_path.is_dir():
        for csv in input_path.rglob("*.csv"):
            files.append(csv)
    elif input_path.suffix == ".csv":
        files.append(input_path)
    else:
        print(f"[WARN] Unsupported input {input_path}")
        return

    for csv_file in files:
        clean_file(csv_file, input_path, output_path, cleaner, args.min_length, stoplist)


if __name__ == "__main__":
    main()
