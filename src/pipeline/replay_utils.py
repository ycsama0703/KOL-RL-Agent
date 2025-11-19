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

