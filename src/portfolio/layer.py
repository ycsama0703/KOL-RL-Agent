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

    def allocate(
        self,
        raw_scores: Dict[str, float],
        prev_weights: Dict[str, float] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """Allocate capital by combining前一日仓位 (prev_weights) 与当日信号 raw_scores."""

        prev_weights = {
            ticker: max(float(weight), 0.0)
            for ticker, weight in (prev_weights or {}).items()
            if weight is not None
        }
        subset_scores = {
            ticker: max(float(score), 0.0)
            for ticker, score in raw_scores.items()
            if score is not None
        }
        result_weights: Dict[str, float] = {}

        if not prev_weights and not subset_scores:
            return {}

        keep_weights = {
            ticker: weight for ticker, weight in prev_weights.items() if ticker not in subset_scores
        }
        keep_sum = sum(keep_weights.values())
        keep_sum = max(keep_sum, 0.0)

        result_weights.update(keep_weights)
        remaining = max(1.0 - keep_sum, 0.0)

        if subset_scores:
            subset_sum = sum(subset_scores.values())
            if subset_sum < self.config.epsilon:
                if remaining > self.config.epsilon:
                    equal = remaining / len(subset_scores)
                    for ticker in subset_scores:
                        result_weights[ticker] = equal
                elif not result_weights:
                    equal = 1.0 / len(subset_scores)
                    for ticker in subset_scores:
                        result_weights[ticker] = equal
            else:
                if remaining < self.config.epsilon:
                    factor = 1.0 / subset_sum
                    for ticker, score in subset_scores.items():
                        result_weights[ticker] = score * factor
                else:
                    factor = remaining / subset_sum
                    for ticker, score in subset_scores.items():
                        result_weights[ticker] = score * factor
        elif keep_sum > self.config.epsilon:
            for ticker in list(result_weights.keys()):
                result_weights[ticker] = result_weights[ticker] / keep_sum

        total = sum(result_weights.values())
        if total > self.config.epsilon:
            for ticker in list(result_weights.keys()):
                result_weights[ticker] = result_weights[ticker] / total
        else:
            result_weights = {}

        allocations = {ticker: weight * self.config.capital for ticker, weight in result_weights.items()}
        return {
            ticker: {"weight": float(weight), "allocation": float(allocations[ticker])}
            for ticker, weight in result_weights.items()
        }
