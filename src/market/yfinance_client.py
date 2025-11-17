"""Utility wrapper for loading market data via yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yfinance as yf


@dataclass
class MarketFeatureConfig:
    """Controls how price history is transformed into RL features."""

    price_column: str = "Adj Close"
    volume_column: str = "Volume"
    return_horizon: int = 1  # daily return
    volatility_window: int = 5  # rolling std of returns
    turnover_window: int = 5  # rolling average volume for normalization


class YFinanceMarketData:
    """Thin client that downloads OHLCV data and computes RL-friendly features."""

    def __init__(
        self,
        config: MarketFeatureConfig | None = None,
        auto_adjust: bool = True,
        interval: str = "1d",
    ) -> None:
        self.config = config or MarketFeatureConfig()
        self.auto_adjust = auto_adjust
        self.interval = interval

    def fetch_history(
        self,
        tickers: Iterable[str],
        start: str | datetime | date,
        end: str | datetime | date,
    ) -> pd.DataFrame:
        """Download OHLCV data and return a tidy DataFrame indexed by (date, ticker)."""
        tickers = sorted({ticker.strip().upper() for ticker in tickers if ticker})
        if not tickers:
            raise ValueError("At least one ticker symbol must be provided.")

        joined = " ".join(tickers)
        raw = yf.download(
            tickers=joined,
            start=start,
            end=end,
            interval=self.interval,
            group_by="ticker",
            auto_adjust=self.auto_adjust,
            progress=False,
            threads=True,
        )

        if raw.empty:
            return pd.DataFrame(columns=["Date", "ticker"]).set_index(["Date", "ticker"])

        if not isinstance(raw.columns, pd.MultiIndex):
            raw["ticker"] = tickers[0]
            tidy = raw.reset_index().set_index(["Date", "ticker"])
            return tidy.sort_index()

        tidy_frames: List[pd.DataFrame] = []
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            df = raw[ticker].copy()
            df["ticker"] = ticker
            tidy_frames.append(df.reset_index().set_index(["Date", "ticker"]))
        if not tidy_frames:
            return pd.DataFrame(columns=["Date", "ticker"]).set_index(["Date", "ticker"])
        tidy = pd.concat(tidy_frames).sort_index()
        return tidy

    def build_features(self, history: pd.DataFrame) -> pd.DataFrame:
        """Compute returns/volatility/turnover features from raw history."""
        if history.empty:
            return history

        cfg = self.config
        grouped = history.groupby(level="ticker")

        price = history[cfg.price_column]
        returns = grouped[cfg.price_column].pct_change(cfg.return_horizon)

        volatility = (
            returns.groupby(level="ticker")
            .rolling(cfg.volatility_window)
            .std()
            .droplevel(0)
        )

        volume = history[cfg.volume_column]
        volume_ma = (
            grouped[cfg.volume_column]
            .rolling(cfg.turnover_window)
            .mean()
            .droplevel(0)
        )
        turnover = (volume / volume_ma) - 1.0

        features = pd.DataFrame(
            {
                "returns": returns,
                "volatility": volatility,
                "turnover": turnover,
            }
        )
        features = features.dropna()
        return features

    def get_feature_lookup(
        self,
        tickers: Iterable[str],
        start: str | datetime | date,
        end: str | datetime | date,
    ) -> Dict[Tuple[date, str], Dict[str, float]]:
        """Return a dict keyed by (date, ticker) → feature dict."""
        history = self.fetch_history(tickers=tickers, start=start, end=end)
        features = self.build_features(history)
        lookup: Dict[Tuple[date, str], Dict[str, float]] = {}
        for (timestamp, ticker), row in features.iterrows():
            current_date = timestamp.date() if isinstance(timestamp, datetime) else timestamp
            lookup[(current_date, ticker)] = {
                "returns": float(row["returns"]),
                "volatility": float(row["volatility"]),
                "turnover": float(row["turnover"]),
            }
        return lookup
