"""Add baseline raw_score derived from sentiment/confidence to reward datasets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment reward CSVs with baseline raw_score.")
    parser.add_argument(
        "--input",
        default="data/processed/reward",
        help="Directory containing reward CSV files (or single CSV).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory. If omitted, files are updated in place.",
    )
    return parser.parse_args()


def collect_csv_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    if path.suffix == ".csv":
        return [path]
    raise ValueError(f"Unsupported path {path}")


def baseline_raw_score(sentiment: float, confidence: float) -> float:
    value = 2.0 * sentiment * confidence
    return math.tanh(value)


def process_file(csv_path: Path, output_root: Path | None) -> None:
    df = pd.read_csv(csv_path)
    if "sentiment" not in df.columns or "confidence" not in df.columns:
        print(f"[WARN] {csv_path} missing sentiment/confidence; skipping.")
        return

    df["baseline_raw_score"] = df["sentiment"].fillna(0.0) * df["confidence"].fillna(0.0) * 2.0
    df["baseline_raw_score"] = df["baseline_raw_score"].apply(math.tanh)

    if output_root:
        try:
            relative = csv_path.relative_to(Path("data/processed/reward"))
        except ValueError:
            relative = csv_path.name
        out_path = output_root / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = csv_path

    df.to_csv(out_path, index=False)
    sample = df[["sentiment", "confidence", "baseline_raw_score"]].head(3).to_dict("records")
    print(f"{csv_path.name}: saved {len(df)} rows -> {out_path} (sample {sample})")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output) if args.output else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)

    for csv_file in collect_csv_files(input_path):
        process_file(csv_file, output_root)


if __name__ == "__main__":
    main()
