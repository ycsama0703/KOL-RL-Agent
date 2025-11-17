"""Split channel datasets into train/val/test partitions by video chronology."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


DEFAULT_CHANNELS = ["Everything_Money", "Invest_with_Henry", "MarketBeat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create chronological splits per channel.")
    parser.add_argument(
        "--input_dir",
        default="data/processed",
        help="Directory containing <channel>.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/splits",
        help="Where to store channel split CSVs.",
    )
    parser.add_argument(
        "--channels",
        default=",".join(DEFAULT_CHANNELS),
        help="Comma separated list of channel file prefixes (no .csv).",
    )
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    return parser.parse_args()


def compute_counts(total: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    train_ratio, val_ratio, test_ratio = ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train = int(total * train_ratio)
    val = int(total * val_ratio)
    test = total - train - val

    # ensure each split has at least one sample when possible
    def borrow_from_larger() -> None:
        nonlocal train, val, test
        while test <= 0 and (train > 1 or val > 1):
            if train >= val and train > 1:
                train -= 1
            else:
                val -= 1
            test = total - train - val

    if train == 0 and total >= 1:
        train = 1
    if val == 0 and total - train >= 2:
        val = 1
    borrow_from_larger()

    if test <= 0 and total > 2:
        test = 1
        if val > 1:
            val -= 1
        elif train > 1:
            train -= 1

    return train, val, total - train - val


def assign_splits(video_order: Sequence[str], counts: Tuple[int, int, int]) -> Dict[str, str]:
    train_count, val_count, _ = counts
    mapping: Dict[str, str] = {}
    train_ids = set(video_order[:train_count])
    val_ids = set(video_order[train_count : train_count + val_count])
    test_ids = set(video_order[train_count + val_count :])
    for vid in train_ids:
        mapping[vid] = "train"
    for vid in val_ids:
        mapping[vid] = "val"
    for vid in test_ids:
        mapping[vid] = "test"
    return mapping


def split_channel(path: Path, output_dir: Path, ratios: Tuple[float, float, float]) -> None:
    df = pd.read_csv(path)
    if "published_at" not in df.columns or "video_id" not in df.columns:
        print(f"[WARN] {path.name} missing required columns; skipping.")
        return

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    grouped = (
        df.groupby("video_id")["published_at"].min().dropna().sort_values().reset_index()
    )
    total_videos = len(grouped)
    if total_videos == 0:
        print(f"[WARN] {path.name} has no valid videos; skipping.")
        return

    counts = compute_counts(total_videos, ratios)
    video_order = grouped["video_id"].tolist()
    mapping = assign_splits(video_order, counts)
    df["split"] = df["video_id"].map(mapping)

    channel_dir = output_dir / path.stem
    channel_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name].drop(columns=["split"])
        summary[split_name] = split_df["video_id"].nunique()
        split_df.to_csv(channel_dir / f"{split_name}.csv", index=False)

    print(
        f"{path.stem}: total_videos={total_videos} "
        f"train={summary['train']} val={summary['val']} test={summary['test']}"
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    channels = [item.strip() for item in args.channels.split(",") if item.strip()]
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    for channel in channels:
        path = input_dir / f"{channel}.csv"
        if not path.exists():
            print(f"[WARN] {path} not found; skipping.")
            continue
        split_channel(path, output_dir, ratios)


if __name__ == "__main__":
    main()
