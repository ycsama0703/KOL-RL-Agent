"""Analyze KOL signal coverage: pos/neg/neutral balance, singleton sentiment, gaps for tickers with >=2 mentions.

Usage:
  python scripts/analyze_signal_coverage.py \
    --reward-csv data/processed/reward/Everything_Money/train.csv \
    --output-dir outputs/signal_coverage \
    --pos-threshold 0.0 --neg-threshold 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze positive/negative/neutral signals and silence gaps.")
    p.add_argument("--reward-csv", required=True, help="Path to reward CSV (train/val/test).")
    p.add_argument("--output-dir", default="outputs/signal_coverage", help="Directory to save reports.")
    p.add_argument("--pos-threshold", type=float, default=0.0, help="sentiment > threshold -> positive.")
    p.add_argument("--neg-threshold", type=float, default=0.0, help="sentiment < threshold -> negative.")
    return p.parse_args()


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # parse dates
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"])
    required = {"ticker", "sentiment", "published_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    # ensure numeric sentiment
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["sentiment"])
    return df


def classify_sentiment(df: pd.DataFrame, pos_thr: float, neg_thr: float) -> pd.DataFrame:
    df = df.copy()
    df["pos"] = df["sentiment"] > pos_thr
    df["neg"] = df["sentiment"] < neg_thr
    df["neu"] = (~df["pos"]) & (~df["neg"])
    return df


def compute_balance(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("ticker")
        .agg(
            total=("ticker", "size"),
            pos=("pos", "sum"),
            neg=("neg", "sum"),
            neu=("neu", "sum"),
        )
        .reset_index()
    )
    summary["pos_pct"] = summary["pos"] / summary["total"]
    summary["neg_pct"] = summary["neg"] / summary["total"]
    summary["neu_pct"] = summary["neu"] / summary["total"]
    return summary


def analyze_singletons(df: pd.DataFrame) -> pd.DataFrame:
    single_tickers = df["ticker"].value_counts()
    singles = single_tickers[single_tickers == 1].index
    sub = df[df["ticker"].isin(singles)]
    sentiment_counts = {
        "pos": int(sub["pos"].sum()),
        "neg": int(sub["neg"].sum()),
        "neu": int(sub["neu"].sum()),
        "total": len(sub),
    }
    return pd.DataFrame([sentiment_counts])


def compute_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Only for tickers with >=2 mentions; compute gaps and tail silence (all/pos/neg)."""
    rows: List[Dict[str, object]] = []
    end_date = df["published_at"].max()
    for ticker, g in df.sort_values("published_at").groupby("ticker"):
        if len(g) < 2:
            continue
        dates = g["published_at"].sort_values().tolist()
        gaps = pd.Series(pd.Series(dates).diff().dt.days.dropna())
        mean_gap = gaps.mean()
        median_gap = gaps.median()
        max_gap = gaps.max()
        tail_gap = (end_date - dates[-1]).days

        g_pos = g[g["pos"]]
        g_neg = g[g["neg"]]
        tail_pos = (end_date - g_pos["published_at"].max()).days if len(g_pos) else np.nan
        tail_neg = (end_date - g_neg["published_at"].max()).days if len(g_neg) else np.nan

        rows.append(
            {
                "ticker": ticker,
                "n_signals": len(dates),
                "mean_gap_days": mean_gap,
                "median_gap_days": median_gap,
                "max_gap_days": max_gap,
                "tail_silence_days": tail_gap,
                "tail_silence_pos_days": tail_pos,
                "tail_silence_neg_days": tail_neg,
                "first_date": dates[0],
                "last_date": dates[-1],
            }
        )
    return pd.DataFrame(rows)


def compute_flips(df: pd.DataFrame) -> pd.DataFrame:
    """For tickers with >=2 mentions, count sign flips (pos->neg or neg->pos)."""
    rows: List[Dict[str, object]] = []
    end_date = df["published_at"].max()
    for ticker, g in df.sort_values("published_at").groupby("ticker"):
        if len(g) < 2:
            continue
        g = g.sort_values("published_at")
        signs = g["sentiment"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).tolist()
        dates = g["published_at"].tolist()
        flips = 0
        first_flip_gap = None
        last_sign = signs[0]
        last_date = dates[0]
        for s, d in zip(signs[1:], dates[1:]):
            if s == 0 or last_sign == 0:
                # ignore neutral in flip counting; update last_sign if current is non-zero
                if s != 0:
                    last_sign = s
                    last_date = d
                continue
            if s != last_sign:
                flips += 1
                if first_flip_gap is None:
                    first_flip_gap = (d - last_date).days
                last_sign = s
                last_date = d
        rows.append(
            {
                "ticker": ticker,
                "n_signals": len(g),
                "flips": flips,
                "has_flip": flips > 0,
                "first_flip_gap_days": first_flip_gap,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.reward_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(csv_path)
    df = classify_sentiment(df, args.pos_threshold, args.neg_threshold)

    balance = compute_balance(df)
    singles_stats = analyze_singletons(df)
    gaps = compute_gaps(df)
    flips = compute_flips(df)

    # Global stats
    total = len(df)
    pos = int(df["pos"].sum())
    neg = int(df["neg"].sum())
    neu = int(df["neu"].sum())
    print(f"File: {csv_path}")
    print(f"Total signals: {total}, pos: {pos} ({pos/total:.2%}), neg: {neg} ({neg/total:.2%}), neu: {neu} ({neu/total:.2%})")

    top_pos = balance.sort_values("pos", ascending=False).head(10)
    top_neg = balance.sort_values("neg", ascending=False).head(10)

    print("\nTop tickers by positive mentions:")
    print(top_pos[["ticker", "pos", "neg", "neu", "total", "pos_pct", "neg_pct", "neu_pct"]].to_string(index=False))
    print("\nTop tickers by negative mentions:")
    print(top_neg[["ticker", "pos", "neg", "neu", "total", "pos_pct", "neg_pct", "neu_pct"]].to_string(index=False))

    # Singleton sentiment
    print("\nSingleton (mentioned once) sentiment counts:")
    print(singles_stats.to_string(index=False))

    # Silence summaries (for tickers with >=2 mentions)
    if not gaps.empty:
        print("\nSilence (tail gap) summary [days] for tickers with >=2 mentions:")
        print(gaps["tail_silence_days"].describe(percentiles=[0.5, 0.75, 0.9, 0.95]))
        if gaps["tail_silence_pos_days"].notna().any():
            print("\nSilence since last positive signal [days]:")
            print(gaps["tail_silence_pos_days"].dropna().describe(percentiles=[0.5, 0.75, 0.9, 0.95]))
        if gaps["tail_silence_neg_days"].notna().any():
            print("\nSilence since last negative signal [days]:")
            print(gaps["tail_silence_neg_days"].dropna().describe(percentiles=[0.5, 0.75, 0.9, 0.95]))
    else:
        print("\nNo tickers with >=2 mentions; gap analysis skipped.")

    # Flip stats
    if not flips.empty:
        flip_rate = flips["has_flip"].mean()
        print(f"\nFlip rate (pos<->neg) among tickers with >=2 mentions: {flip_rate:.2%}")
        if flips["first_flip_gap_days"].notna().any():
            print("First flip gap [days] stats:")
            print(flips["first_flip_gap_days"].dropna().describe(percentiles=[0.5, 0.75, 0.9, 0.95]))
    else:
        print("\nNo flip stats (tickers with >=2 mentions only).")

    # Save reports
    balance.to_csv(out_dir / f"{csv_path.stem}_signal_balance.csv", index=False)
    gaps.to_csv(out_dir / f"{csv_path.stem}_signal_gaps.csv", index=False)
    singles_stats.to_csv(out_dir / f"{csv_path.stem}_singletons.csv", index=False)
    print(f"\nSaved balance   -> {out_dir / (csv_path.stem + '_signal_balance.csv')}")
    flips.to_csv(out_dir / f"{csv_path.stem}_signal_flips.csv", index=False)
    print(f"Saved gaps      -> {out_dir / (csv_path.stem + '_signal_gaps.csv')} (tickers with >=2 mentions)")
    print(f"Saved flips     -> {out_dir / (csv_path.stem + '_signal_flips.csv')} (tickers with >=2 mentions)")
    print(f"Saved singletons-> {out_dir / (csv_path.stem + '_singletons.csv')}")


if __name__ == "__main__":
    main()
