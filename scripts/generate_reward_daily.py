"""Generate 1-trading-day reward labels for enriched KOL datasets.

This script is an alternative to `scripts/generate_reward.py`:
- `generate_reward.py`: event-to-next-event span reward
- `generate_reward_daily.py`: current trading day close -> next trading day close
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute next-trading-day return rewards for KOL data.")
    parser.add_argument(
        "--input",
        default="data/processed/enriched",
        help="Directory containing enriched CSV files (or a single CSV).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/reward_daily",
        help="Directory to store reward-augmented CSV files.",
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


def sanitize_ticker(ticker: str) -> str:
    return ticker.strip().replace(".", "-").upper()


def fetch_prices(
    ticker: str,
    period: str,
    cache: Dict[str, pd.Series],
) -> pd.Series:
    yf_ticker = sanitize_ticker(ticker)
    if yf_ticker in cache:
        return cache[yf_ticker]
    data = yf.download(yf_ticker, period=period, progress=False, auto_adjust=False)
    if data.empty:
        series = pd.Series(dtype=float)
        cache[yf_ticker] = series
        return series
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            cache[yf_ticker] = pd.Series(dtype=float)
            return cache[yf_ticker]
        prices = close.iloc[:, 0].copy()
    else:
        prices = close.copy()
    prices.index = prices.index.tz_localize(None)
    cache[yf_ticker] = prices
    return prices


def _next_valid_index(prices: pd.Series, start_idx: int) -> Optional[int]:
    idx = start_idx
    while idx < len(prices) and pd.isna(prices.iloc[idx]):
        idx += 1
    if idx >= len(prices):
        return None
    return idx


def find_daily_reward(
    trading_day: pd.Timestamp,
    prices: pd.Series,
) -> Tuple[float, Optional[pd.Timestamp], bool]:
    """Return (reward_1d, next_trading_day, done) from close[t] to close[t+1]."""
    if prices.empty:
        return 0.0, None, True

    day = trading_day.floor("D")
    idx_start = prices.index.searchsorted(day, side="left")
    idx_start = _next_valid_index(prices, idx_start)
    if idx_start is None:
        return 0.0, None, True

    idx_end = _next_valid_index(prices, idx_start + 1)
    if idx_end is None:
        return 0.0, None, True

    p0 = prices.iloc[idx_start]
    p1 = prices.iloc[idx_end]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return 0.0, prices.index[idx_end], False

    reward = float(p1 / p0 - 1.0)
    return reward, prices.index[idx_end], False


def process_file(
    csv_path: Path,
    input_root: Path,
    output_dir: Path,
    period: str,
    global_price_cache: Dict[str, pd.Series],
) -> None:
    df = pd.read_csv(csv_path)
    if "ticker" not in df.columns:
        print(f"[WARN] {csv_path} missing ticker; skipping.")
        return

    if "trading_day" in df.columns:
        trading_day = pd.to_datetime(df["trading_day"], errors="coerce")
    elif "published_at" in df.columns:
        published = pd.to_datetime(df["published_at"], errors="coerce")
        published = published.apply(normalize_timestamp)
        trading_day = published.dt.floor("D")
    else:
        print(f"[WARN] {csv_path} missing trading_day/published_at; skipping.")
        return

    df["ticker"] = df["ticker"].astype(str)
    df["_trading_day"] = trading_day

    rewards: list[float] = []
    next_dates: list[Optional[str]] = []
    done_flags: list[bool] = []

    for ticker, td in zip(df["ticker"].astype(str).tolist(), df["_trading_day"].tolist()):
        if pd.isna(td):
            rewards.append(0.0)
            next_dates.append(None)
            done_flags.append(True)
            continue

        prices = fetch_prices(ticker, period=period, cache=global_price_cache)
        reward, next_day, done = find_daily_reward(pd.Timestamp(td), prices)
        rewards.append(reward)
        next_dates.append(next_day.isoformat() if isinstance(next_day, pd.Timestamp) else None)
        done_flags.append(bool(done))

    df = df.drop(columns=["_trading_day"])
    df["reward_1d"] = rewards
    df["next_date"] = next_dates
    df["done"] = done_flags

    try:
        relative = csv_path.relative_to(input_root)
    except ValueError:
        relative = Path(csv_path.name)
    output_path = output_dir / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"{csv_path.name}: saved {len(df)} rows -> {output_path} "
        f"(sample: {df[['ticker','reward_1d','next_date','done']].head(3).to_dict('records')})"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_cache: Dict[str, pd.Series] = {}
    csv_files = collect_csv_files(input_path)
    if not csv_files:
        print(f"No CSV files found under {input_path}")
        return

    for csv_path in csv_files:
        process_file(
            csv_path,
            input_path,
            output_dir,
            period=args.period,
            global_price_cache=price_cache,
        )


if __name__ == "__main__":
    main()
