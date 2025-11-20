"""Summarize portfolio compositions from positions_test.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize portfolio composition per date.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to positions_test.csv (or similar positions file).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path (CSV). Defaults to <input_dir>/portfolio_summary.csv.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top holdings to include in summary column.",
    )
    return parser.parse_args()


def summarize(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    summaries: List[Dict[str, object]] = []
    for date, group in df.groupby("date"):
        weights = group[["ticker", "weight"]].set_index("ticker")["weight"]
        allocation = group.set_index("ticker")["allocation"]
        total_long = weights[weights > 0].sum()
        total_short = weights[weights < 0].sum()
        composition = {ticker: float(weight) for ticker, weight in weights.items()}
        top_holdings = (
            ", ".join(f"{ticker}:{weight:.3f}" for ticker, weight in weights.abs().sort_values(ascending=False).head(top_k).items())
            if not weights.empty
            else ""
        )
        summaries.append(
            {
                "date": date.isoformat(),
                "holdings_count": len(weights),
                "long_count": int((weights > 0).sum()),
                "short_count": int((weights < 0).sum()),
                "total_long_weight": float(total_long),
                "total_short_weight": float(total_short),
                "net_weight": float(weights.sum()),
                "total_long_allocation": float(allocation[allocation > 0].sum()) if not allocation.empty else 0.0,
                "total_short_allocation": float(allocation[allocation < 0].sum()) if not allocation.empty else 0.0,
                "composition_json": json.dumps(composition, ensure_ascii=False),
                "top_holdings": top_holdings,
            }
        )
    return pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    required_cols = {"date", "ticker", "weight", "allocation"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in positions file: {missing}")

    summary = summarize(df, top_k=args.top_k)
    output_path = Path(args.output) if args.output else input_path.parent / "portfolio_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved portfolio summary to {output_path} ({len(summary)} rows).")


if __name__ == "__main__":
    main()
