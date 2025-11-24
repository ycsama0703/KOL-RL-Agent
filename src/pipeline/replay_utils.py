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
    """Reconstruct baseline positions and add carry rows for tickers not mentioned today.

    Outputs last_position, baseline_weight, silence_days. For tickers held yesterday but
    not mentioned today, we add a synthetic row with baseline_weight=0, baseline_raw_score=0,
    sentiment/confidence=0, reward_1d=0, embeddings/text set to 0/"" so that the decay branch
    has training/inference samples.
    """

    df = df.sort_values("published_at").reset_index(drop=True)
    portfolio = PortfolioLayer()

    embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
    base_defaults = {col: 0 for col in df.columns}
    if "text" in base_defaults:
        base_defaults["text"] = ""
    if "video_id" in base_defaults:
        base_defaults["video_id"] = ""
    if "company" in base_defaults:
        base_defaults["company"] = ""

    prev_weights: Dict[str, float] = {}
    last_dates: Dict[str, pd.Timestamp] = {}
    rows: list[dict] = []

    for date, group in df.groupby("published_at", sort=True):
        raw_dict = {
            row["ticker"]: float(row["baseline_raw_score"]) * float(np.sign(row.get("sentiment", 0.0)))
            for _, row in group.iterrows()
        }
        weights = portfolio.allocate(raw_dict, prev_weights=prev_weights)

        # real signal rows
        for _, row in group.iterrows():
            ticker = row["ticker"]
            cur_date = row["published_at"]
            prev_date = last_dates.get(ticker)
            silence = float((cur_date - prev_date).days) if prev_date is not None else 0.0
            last_dates[ticker] = cur_date
            enriched = row.to_dict()
            enriched["last_position"] = prev_weights.get(ticker, 0.0)
            enriched["baseline_weight"] = float(weights.get(ticker, {"weight": 0.0})["weight"])
            enriched["silence_days"] = silence
            enriched["has_signal"] = 1
            rows.append(enriched)

        # carry rows for tickers held yesterday but not mentioned today
        carry = [t for t in prev_weights.keys() if t not in raw_dict]
        for ticker in carry:
            cur_date = date
            prev_date = last_dates.get(ticker, cur_date)
            silence = float((cur_date - prev_date).days)
            last_dates[ticker] = cur_date
            enriched = base_defaults.copy()
            enriched["ticker"] = ticker
            enriched["published_at"] = cur_date
            enriched["sentiment"] = 0.0
            enriched["confidence"] = 0.0
            enriched["baseline_raw_score"] = 0.0
            enriched["reward_1d"] = 0.0
            enriched["done"] = False
            for col in embedding_cols:
                enriched[col] = 0.0
            enriched["last_position"] = prev_weights.get(ticker, 0.0)
            enriched["baseline_weight"] = 0.0
            enriched["silence_days"] = silence
            enriched["has_signal"] = 0
            rows.append(enriched)

        prev_weights = {t: info["weight"] for t, info in weights.items()}

    return pd.DataFrame(rows)


def compute_portfolio_rewards(df: pd.DataFrame) -> pd.Series:
    """Compute portfolio-level reward (组合日收益，无换手成本惩罚).

    组合日收益: r_t = Σ_i weight_{t,i} * reward_1d_{t,i}
    weight 可正可负（多/空），奖励自然体现做空盈利。
    返回一个与 df 等长的 Series，每条样本对应其所属日期的组合 reward。
    """

    if "baseline_weight" not in df.columns or "last_position" not in df.columns:
        raise ValueError("compute_portfolio_rewards expects 'baseline_weight' and 'last_position' columns.")

    df = df.sort_values("published_at").reset_index(drop=True)
    group_indices = df.groupby("published_at", sort=True).indices
    portfolio_rewards = np.zeros(len(df), dtype=np.float32)

    for _, indices in group_indices.items():
        idx_list = list(indices)
        group = df.loc[idx_list]
        w_today = group["baseline_weight"].astype(float)
        r_today = group["reward_1d"].astype(float)
        r_port = float((w_today * r_today).sum())
        portfolio_rewards[idx_list] = r_port

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
    feature_cols = ["sentiment", "confidence", "last_position", "silence_days"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    extra_features = df[feature_cols].fillna(0.0).values.astype(np.float32)
    states = np.concatenate([text_emb, ticker_vectors, extra_features], axis=1)
    return states
