"""Split processed dataset into per-channel CSV files for the top N KOLs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into per-channel CSV files.")
    parser.add_argument(
        "--input",
        default="data/processed/kol_text_with_sentiment.csv",
        help="Path to the processed dataset.",
    )
    parser.add_argument("--top", type=int, default=5, help="Number of channels to keep.")
    parser.add_argument(
        "--output_dir",
        default="data/processed/top_channels",
        help="Directory where channel-specific CSVs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_channels = df["channel_name"].value_counts().head(args.top).index.tolist()

    for channel in top_channels:
        safe_name = (
            channel.replace(" ", "_")
            .replace("/", "_")
            .replace(",", "_")
            .replace("'", "")
            .replace('"', "")
        )
        channel_df = df[df["channel_name"] == channel]
        out_path = output_dir / f"{safe_name}.csv"
        channel_df.to_csv(out_path, index=False)
        print(f"Saved {len(channel_df)} rows for {channel} -> {out_path}")


if __name__ == "__main__":
    main()
