"""Generate next-day return rewards for enriched KOL datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute next-day return rewards for KOL data.")
    parser.add_argument(
        "--input",
        default="data/processed/enriched",
        help="Directory containing enriched CSV files (or a single CSV).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/reward",
        help="Directory to store reward-augmented CSV files.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional list of tickers to restrict downloads (default: infer from data).",
    )
    parser.add_argument(
        "--period",
        default="max",
        help="Historical period to download via yfinance (default: max).",
    )
    return parser.parse_args()


def collect_csv_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    if path.suffix == ".csv":
        return [path]
    raise ValueError(f"Unsupported input path: {path}")


def normalize_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    if value.tzinfo is not None:
        value = value.tz_convert("UTC").tz_localize(None)
    return value


def fetch_prices(
    ticker: str,
    period: str,
    cache: Dict[str, pd.Series],
) -> pd.Series:
    if ticker in cache:
        return cache[ticker]
    data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if data.empty:
        series = pd.Series(dtype=float)
        cache[ticker] = series
        return series
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            cache[ticker] = pd.Series(dtype=float)
            return cache[ticker]
        prices = close.iloc[:, 0].copy()
    else:
        prices = close.copy()
    prices.index = prices.index.tz_localize(None)
    cache[ticker] = prices
    return prices


def find_reward(
    signal_time: pd.Timestamp,
    prices: pd.Series,
) -> Tuple[float, Optional[pd.Timestamp], bool]:
    """Return (reward_1d, next_date, done)."""
    if prices.empty:
        return 0.0, None, True
    signal_time = signal_time.floor("D")
    idx = prices.index.searchsorted(signal_time, side="left")
    if idx >= len(prices):
        return 0.0, None, True

    close_idx = idx
    while close_idx < len(prices) and pd.isna(prices.iloc[close_idx]):
        close_idx += 1
    if close_idx >= len(prices):
        return 0.0, None, True
    close_t = prices.iloc[close_idx]

    next_idx = close_idx + 1
    while next_idx < len(prices) and pd.isna(prices.iloc[next_idx]):
        next_idx += 1
    if next_idx >= len(prices):
        return 0.0, None, True

    next_close = prices.iloc[next_idx]
    next_date = prices.index[next_idx]
    if pd.isna(next_close) or pd.isna(close_t) or close_t == 0:
        return 0.0, next_date, False

    reward = float(next_close / close_t - 1.0)
    return reward, next_date, False


def process_file(
    csv_path: Path,
    output_dir: Path,
    period: str,
    global_price_cache: Dict[str, pd.Series],
) -> None:
    df = pd.read_csv(csv_path)
    if "ticker" not in df.columns or "published_at" not in df.columns:
        print(f"[WARN] {csv_path} missing ticker or published_at; skipping.")
        return

    df["published_at"] = pd.to_datetime(df["published_at"])
    df["ticker"] = df["ticker"].astype(str)

    reward_col = []
    next_dates = []
    done_flags = []

    for row in df.itertuples(index=False):
        ticker = row.ticker
        signal_time = normalize_timestamp(row.published_at)
        prices = fetch_prices(ticker, period=period, cache=global_price_cache)
        reward, next_date, done = find_reward(signal_time, prices)
        reward_col.append(reward)
        next_dates.append(next_date.isoformat() if isinstance(next_date, pd.Timestamp) else None)
        done_flags.append(bool(done))

    df["reward_1d"] = reward_col
    df["next_date"] = next_dates
    df["done"] = done_flags

    try:
        relative = csv_path.relative_to(Path("data/processed/enriched"))
    except ValueError:
        relative = csv_path.name
    output_path = output_dir / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"{csv_path.name}: saved {len(df)} rows -> {output_path} "
        f"(sample: {df[['ticker','published_at','reward_1d','next_date','done']].head(3).to_dict('records')})"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_cache: Dict[str, pd.DataFrame] = {}
    csv_files = collect_csv_files(input_path)
    if not csv_files:
        print(f"No CSV files found under {input_path}")
        return

    for csv_path in csv_files:
        process_file(csv_path, output_dir, period=args.period, global_price_cache=price_cache)


if __name__ == "__main__":
    main()
