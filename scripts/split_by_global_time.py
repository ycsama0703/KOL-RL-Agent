"""Split datasets into train/test using a global time cutoff across all channels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split channel datasets into train/test by a global time cutoff.",
    )
    parser.add_argument(
        "--input_dir",
        default="data/processed/22-24",
        help="Directory containing channel CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/splits/22-24",
        help="Where to store channel split CSVs.",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma separated list of channel file stems (no .csv). Default: all CSVs in input_dir.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of data to assign to train by global time quantile (rest to test).",
    )
    return parser.parse_args()


def select_files(input_dir: Path, channels: List[str] | None) -> List[Path]:
    if channels:
        return [input_dir / f"{name}.csv" for name in channels]
    return sorted(input_dir.glob("*.csv"))


def parse_timestamps(series: pd.Series) -> pd.Series:
    # Use UTC to avoid mixed tz issues, then drop timezone for comparisons.
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def compute_global_cutoff(files: Iterable[Path], train_ratio: float) -> pd.Timestamp:
    timestamps: List[pd.Series] = []
    for path in files:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["published_at"])
        except ValueError:
            continue
        ts = parse_timestamps(df["published_at"]).dropna()
        if not ts.empty:
            timestamps.append(ts)
    if not timestamps:
        raise ValueError("No valid published_at timestamps found in input files.")
    all_ts = pd.concat(timestamps, ignore_index=True)
    return all_ts.quantile(train_ratio)


def split_file(path: Path, output_dir: Path, cutoff: pd.Timestamp) -> None:
    df = pd.read_csv(path)
    if "published_at" not in df.columns:
        print(f"[WARN] {path.name} missing published_at; skipping.")
        return

    ts = parse_timestamps(df["published_at"])
    valid = ts.notna()
    train_mask = valid & (ts <= cutoff)
    test_mask = valid & (ts > cutoff)

    if (~valid).any():
        print(f"[WARN] {path.name} has {int((~valid).sum())} rows with invalid published_at; dropping.")

    channel_dir = output_dir / path.stem
    channel_dir.mkdir(parents=True, exist_ok=True)

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()
    train_df.to_csv(channel_dir / "train.csv", index=False)
    test_df.to_csv(channel_dir / "test.csv", index=False)

    print(
        f"{path.stem}: train={len(train_df)} test={len(test_df)} total={len(train_df) + len(test_df)}"
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    channels = None
    if args.channels:
        channels = [item.strip() for item in args.channels.split(",") if item.strip()]

    files = select_files(input_dir, channels)
    if not files:
        raise ValueError(f"No CSV files found in {input_dir}")

    cutoff = compute_global_cutoff(files, args.train_ratio)
    print(f"Global cutoff (train_ratio={args.train_ratio}): {cutoff}")

    for path in files:
        if not path.exists():
            print(f"[WARN] {path} not found; skipping.")
            continue
        split_file(path, output_dir, cutoff)


if __name__ == "__main__":
    main()
