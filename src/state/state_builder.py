"""Constructs RL state vectors from market, KOL, and portfolio inputs."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .ticker_embedding import TickerEmbedding


class StateBuilder:
    """Concatenate heterogeneous features to one flat vector."""

    def __init__(self, ticker_embedder: Optional[TickerEmbedding] = None) -> None:
        self.ticker_embedder = ticker_embedder

    def build(
        self,
        market_features: Dict[str, float],
        kol_features: Sequence[float],
        last_position: float,
        ticker: Optional[str] = None,
    ) -> List[float]:
        state: List[float] = list(kol_features)
        if self.ticker_embedder and ticker:
            state.extend(self.ticker_embedder.encode_single(ticker))
        state.extend(market_features.values())
        state.append(last_position)
        return state
