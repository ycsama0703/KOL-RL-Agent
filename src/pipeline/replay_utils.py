"""Shared helpers for building replay buffers and analysis inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.portfolio.layer import PortfolioLayer
from src.state.ticker_embedding import TickerEmbedding


def load_ticker_embedder(weights_path: Path, vocab_path: Path, embedding_dim: int = 32) -> TickerEmbedding:
    """Load a TickerEmbedding with consistent default dimension."""

    return TickerEmbedding.load(weights_path, vocab_path, embedding_dim=embedding_dim)


def annotate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct baseline positions to obtain last_position and baseline_weight.

    Expects columns: ticker, published_at, baseline_raw_score.
    """

    df = df.sort_values("published_at").reset_index(drop=True)
    portfolio = PortfolioLayer()
    prev_weights: Dict[str, float] = {}
    last_positions = np.zeros(len(df), dtype=np.float32)
    baseline_weights = np.zeros(len(df), dtype=np.float32)

    grouped = df.groupby("published_at", sort=True)
    for _, group in grouped:
        raw_dict = {row["ticker"]: row["baseline_raw_score"] for _, row in group.iterrows()}
        weights = portfolio.allocate(raw_dict)
        for idx, row in group.iterrows():
            ticker = row["ticker"]
            last_positions[idx] = prev_weights.get(ticker, 0.0)
            baseline_weights[idx] = float(weights.get(ticker, {"weight": 0.0})["weight"])
            prev_weights[ticker] = baseline_weights[idx]

    df["last_position"] = last_positions
    df["baseline_weight"] = baseline_weights
    return df


def compute_portfolio_rewards(
    df: pd.DataFrame,
    cost_per_turnover: float = 0.0005,
) -> pd.Series:
    """Compute portfolio-level reward (组合日收益 - 换手成本).

    - 组合日收益: r_t = Σ_i baseline_weight_{t,i} * reward_1d_{t,i}
    - 换手成本:   cost_t = cost_per_turnover * Σ_i |w_{t,i} - w_{t-1,i}|
      其中 w 为 baseline_weight；若某 ticker 当日或前一日不存在，则权重视为 0。
    返回一个与 df 等长的 Series，每条样本对应其所属日期的组合 reward。
    """

    if "baseline_weight" not in df.columns or "last_position" not in df.columns:
        raise ValueError("compute_portfolio_rewards expects 'baseline_weight' and 'last_position' columns.")

    df = df.sort_values("published_at").reset_index(drop=True)
    rewards = df["reward_1d"].astype(float).values
    dates = df["published_at"]
    weights = df["baseline_weight"].astype(float).values

    portfolio = PortfolioLayer()
    group_indices = df.groupby("published_at", sort=True).indices

    portfolio_rewards = np.zeros(len(df), dtype=np.float32)
    prev_weights_by_ticker: Dict[str, float] = {}

    for date, indices in group_indices.items():
        idx_list = list(indices)
        group = df.loc[idx_list]
        # 当日组合收益：baseline_weight * reward_1d
        w_today = group["baseline_weight"].astype(float)
        r_today = group["reward_1d"].astype(float)
        r_port = float((w_today * r_today).sum())

        # 换手率：上一日 baseline 权重 vs 今日 baseline 权重
        curr_weights_by_ticker: Dict[str, float] = {
            str(ticker): float(w) for ticker, w in zip(group["ticker"], w_today)
        }
        all_tickers = set(prev_weights_by_ticker) | set(curr_weights_by_ticker)
        turnover = 0.0
        for ticker in all_tickers:
            w_prev = prev_weights_by_ticker.get(ticker, 0.0)
            w_curr = curr_weights_by_ticker.get(ticker, 0.0)
            turnover += abs(w_curr - w_prev)

        cost_t = cost_per_turnover * turnover
        r_eff = r_port - cost_t

        portfolio_rewards[idx_list] = r_eff
        prev_weights_by_ticker = curr_weights_by_ticker

    return pd.Series(portfolio_rewards, index=df.index, name="portfolio_reward")


def build_states(df: pd.DataFrame, ticker_embedder: TickerEmbedding) -> np.ndarray:
    """Construct state vectors consistent with training time definition.

    state = [ModernBERT embedding || ticker embedding || sentiment || confidence || last_position]
    """

    embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
    text_emb = df[embedding_cols].values.astype(np.float32)
    ticker_vectors = np.stack(
        [ticker_embedder.encode_single(str(ticker)) for ticker in df["ticker"].astype(str)],
        dtype=np.float32,
    )
    feature_cols = ["sentiment", "confidence", "last_position"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    extra_features = df[feature_cols].fillna(0.0).values.astype(np.float32)
    states = np.concatenate([text_emb, ticker_vectors, extra_features], axis=1)
    return states
