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


def find_span_reward(
    signal_time: pd.Timestamp,
    next_signal_time: Optional[pd.Timestamp],
    prices: pd.Series,
) -> Tuple[float, Optional[pd.Timestamp], bool]:
    """Return (reward_span, end_trading_date, done) for窗口=当前视频→下一次视频.

    - 若 next_signal_time 为 None，视为 episode 终点，reward=0，done=True；
    - 否则：
      - start_price: 信号日当天（或之后最近一个有价格的交易日）的收盘价；
      - end_price:   下一次视频日期当天（或之后最近一个有价格的交易日）的收盘价；
      - reward_span = end_price / start_price - 1。
    """
    if prices.empty:
        return 0.0, None, True
    if next_signal_time is None:
        # 最后一条视频，没有“下一个视频”，episode 结束。
        return 0.0, None, True

    signal_time = signal_time.floor("D")
    next_signal_time = next_signal_time.floor("D")

    # 找起点收盘价
    idx_start = prices.index.searchsorted(signal_time, side="left")
    if idx_start >= len(prices):
        return 0.0, None, True
    while idx_start < len(prices) and pd.isna(prices.iloc[idx_start]):
        idx_start += 1
    if idx_start >= len(prices):
        return 0.0, None, True
    start_price = prices.iloc[idx_start]

    # 找终点收盘价（对应下一次视频日期）
    idx_end = prices.index.searchsorted(next_signal_time, side="left")
    if idx_end >= len(prices):
        return 0.0, None, True
    while idx_end < len(prices) and pd.isna(prices.iloc[idx_end]):
        idx_end += 1
    if idx_end >= len(prices):
        return 0.0, None, True
    end_price = prices.iloc[idx_end]
    end_date = prices.index[idx_end]

    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return 0.0, end_date, False

    reward = float(end_price / start_price - 1.0)
    return reward, end_date, False


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

    # 统一将时间戳转为无时区（tz-naive），避免与 yfinance 数据索引不兼容
    df["published_at"] = pd.to_datetime(df["published_at"]).apply(normalize_timestamp)
    df["ticker"] = df["ticker"].astype(str)
    # 按自然日对齐事件日期，用于定义“下一次视频”的窗口（同样为 tz-naive）
    df["event_date"] = df["published_at"].dt.floor("D")
    unique_dates = sorted(df["event_date"].unique())
    next_date_map = {d: unique_dates[i + 1] for i, d in enumerate(unique_dates[:-1])}

    reward_col = []
    next_dates = []
    done_flags = []

    for row in df.itertuples(index=False):
        ticker = row.ticker
        # 此时 published_at / event_date 均为 tz-naive
        signal_time = row.published_at
        event_date = row.event_date
        next_event_date = next_date_map.get(event_date)
        prices = fetch_prices(ticker, period=period, cache=global_price_cache)
        reward, end_trading_date, done = find_span_reward(signal_time, next_event_date, prices)
        reward_col.append(reward)
        next_dates.append(end_trading_date.isoformat() if isinstance(end_trading_date, pd.Timestamp) else None)
        done_flags.append(bool(done))

    # 名称仍沿用 reward_1d，但语义已变为“当前视频→下一次视频”的窗口收益。
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
