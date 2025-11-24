"""Augment cleaned datasets with ModernBERT embeddings and 5-day price history."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from pandas.tseries.offsets import BDay

# add repo root for local imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.market.company_mapper import CompanyTickerMapper


LOGGER = logging.getLogger("augment")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create enriched datasets with embeddings + price history.")
    parser.add_argument("--input", default="data/processed/cleaned", help="Root directory containing <KOL>/<split>.csv")
    parser.add_argument("--embeddings", default="data/processed/embeddings", help="Root directory containing <KOL>/<split>.pt")
    parser.add_argument("--output", default="data/processed/enriched", help="Destination root directory for augmented CSVs.")
    parser.add_argument(
        "--ticker-reference",
        default="data/input/top_500_companies_list.xlsx",
        help="Excel file containing at least [Symbol, Company Name] columns.",
    )
    parser.add_argument("--channels", default=None, help="Comma separated list of channel folders to process (default all).")
    parser.add_argument("--price-days", type=int, default=5, help="Number of business days of close prices to append.")
    parser.add_argument("--chunk-size", type=int, default=20, help="Number of tickers per Yahoo Finance download chunk.")
    return parser.parse_args()


def chunk_list(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def sanitize_ticker(ticker: str) -> str:
    return ticker.replace(".", "-")


def download_single_ticker(ticker: str, start, end) -> pd.Series | None:
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Failed download for %s: %s", ticker, exc)
        return None
    if data.empty or "Close" not in data.columns:
        return None
    return data["Close"].dropna()


def download_close_panel(tickers: List[str], start, end, chunk_size: int) -> Dict[str, pd.Series]:
    closes: Dict[str, pd.Series] = {}
    for chunk in chunk_list(tickers, chunk_size):
        try:
            data = yf.download(
                " ".join(chunk),
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to download chunk %s: %s", chunk, exc)
            data = pd.DataFrame()

        if data.empty:
            for ticker in chunk:
                series = download_single_ticker(ticker, start, end)
                if series is not None:
                    closes[ticker] = series
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for ticker in chunk:
                column = (ticker, "Close")
                if column in data.columns:
                    closes[ticker] = data[column].dropna()
                else:
                    series = download_single_ticker(ticker, start, end)
                    if series is not None:
                        closes[ticker] = series
        else:
            if len(chunk) == 1 and "Close" in data.columns:
                closes[chunk[0]] = data["Close"].dropna()
            else:
                # fallback per ticker
                for ticker in chunk:
                    series = download_single_ticker(ticker, start, end)
                    if series is not None:
                        closes[ticker] = series
    return closes


def append_price_windows(
    df: pd.DataFrame,
    closes: Dict[str, pd.Series],
    price_cols: List[str],
) -> pd.DataFrame:
    filled = []
    for idx, row in df.iterrows():
        ticker = row["yf_ticker"]
        series = closes.get(ticker)
        if series is None or series.empty:
            continue
        publish_ts = row["published_at"]
        if publish_ts.tzinfo is not None:
            publish_ts = publish_ts.tz_convert(None)
        cutoff = pd.Timestamp(publish_ts.date())
        history = series.loc[:cutoff].tail(len(price_cols))
        if len(history) < len(price_cols):
            continue
        enriched = row.to_dict()
        for label, price in zip(price_cols, history.tolist()):
            enriched[label] = float(price)
        enriched.pop("yf_ticker", None)
        filled.append(enriched)
    if not filled:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(filled)


def process_file(
    csv_path: Path,
    emb_path: Path,
    output_path: Path,
    mapper: CompanyTickerMapper,
    price_days: int,
    chunk_size: int,
) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        LOGGER.info("Skipping %s (empty)", csv_path)
        return
    if not emb_path.exists():
        LOGGER.warning("Missing embedding file %s; skipping %s", emb_path, csv_path)
        return
    payload = torch.load(emb_path, map_location="cpu")
    embeddings = payload["embeddings"]
    if len(embeddings) != len(df):
        LOGGER.warning("Row mismatch between %s and %s; skipping.", csv_path, emb_path)
        return

    emb_array = embeddings.numpy() if hasattr(embeddings, "numpy") else np.array(embeddings)
    emb_cols = [f"embedding_{idx}" for idx in range(emb_array.shape[1])]
    emb_df = pd.DataFrame(emb_array, columns=emb_cols)

    df = df.reset_index(drop=True)
    df = pd.concat([df, emb_df], axis=1)

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["ticker"] = df["company"].astype(str).apply(mapper.lookup)
    df = df.dropna(subset=["published_at", "ticker"]).reset_index(drop=True)
    if df.empty:
        LOGGER.info("No usable rows after ticker filtering for %s", csv_path)
        return
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["yf_ticker"] = df["ticker"].apply(sanitize_ticker)

    price_cols = [f"close_t-{offset}" for offset in range(price_days - 1, -1, -1)]
    for col in price_cols:
        df[col] = pd.NA

    tickers = sorted(df["yf_ticker"].unique())
    min_ts = df["published_at"].min()
    max_ts = df["published_at"].max()
    start = (min_ts - BDay(price_days * 2)).date()
    end = (max_ts + BDay(2)).date()
    close_panel = download_close_panel(tickers, start, end, chunk_size)
    if not close_panel:
        LOGGER.warning("No price data fetched for %s; skipping.", csv_path)
        return

    enriched_df = append_price_windows(df, close_panel, price_cols)
    if enriched_df.empty:
        LOGGER.warning("No rows retained after price merge for %s", csv_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows -> %s", len(enriched_df), output_path)


def main() -> None:
    configure_logging()
    args = parse_args()

    input_root = Path(args.input)
    emb_root = Path(args.embeddings)
    output_root = Path(args.output)

    overrides = {
        "s&p 500": None,
        "sp 500": None,
        "german dax index": None,
        "dax": None,
        "nikkei": None,
        "hang seng": None,
        "dow jones": None,
        "nasdaq": None,
        "walgreens": None,
        "walgreens boots alliance": None,
        "berkshire hathaway": "BRK-B",
        "brk.b": "BRK-B",
        "paramount": None,
        "paramount global": None,
        "coca cola": "KO",
        "under armour": "UAA",
        "mcdonald s": "MCD",
        "home depot": "HD",
        "lowe s": "LOW",
        "target": "TGT",
        "tesla": "TSLA",
        "apple": "AAPL",
        "nvidia": "NVDA",
        "nike": "NKE",
    }
    mapper = CompanyTickerMapper(Path(args.ticker_reference), manual_overrides=overrides)

    channels = None
    if args.channels:
        channels = [item.strip() for item in args.channels.split(",") if item.strip()]

    for channel_dir in sorted(input_root.iterdir()):
        if not channel_dir.is_dir():
            continue
        channel_name = channel_dir.name
        if channels and channel_name not in channels:
            continue
        LOGGER.info("Processing channel %s", channel_name)
        for split_file in sorted(channel_dir.glob("*.csv")):
            rel = split_file.relative_to(input_root)
            emb_path = emb_root / rel.with_suffix(".pt")
            output_path = output_root / rel
            process_file(
                csv_path=split_file,
                emb_path=emb_path,
                output_path=output_path,
                mapper=mapper,
                price_days=args.price_days,
                chunk_size=args.chunk_size,
            )


if __name__ == "__main__":
    main()
