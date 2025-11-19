"""Portfolio layer to convert raw scores into dollar allocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PortfolioConfig:
    capital: float = 10_000.0
    epsilon: float = 1e-6


class PortfolioLayer:
    """Normalizes raw scores into portfolio weights and allocations."""

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()

    def allocate(self, raw_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        if not raw_scores:
            return {}

        tickers: List[str] = list(raw_scores.keys())
        scores = np.array([raw_scores[ticker] for ticker in tickers], dtype=np.float64)
        # Long-only: ignore负分数，只按正分数归一化；若全非正，则等权多头。
        scores = np.maximum(scores, 0.0)
        pos_sum = float(np.sum(scores))

        if pos_sum < self.config.epsilon:
            # fallback: equal weight long-only distribution
            weights = np.ones_like(scores, dtype=np.float64) / len(scores)
        else:
            weights = scores / pos_sum

        allocations = weights * self.config.capital
        result: Dict[str, Dict[str, float]] = {}
        for ticker, weight, allocation in zip(tickers, weights.tolist(), allocations.tolist()):
            result[ticker] = {"weight": float(weight), "allocation": float(allocation)}
        return result
