"""Portfolio layer to convert raw scores into dollar allocations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List

import numpy as np


@dataclass
class PortfolioConfig:
    capital: float = 10_000.0
    epsilon: float = 1e-6
    max_long: float = 0.2  # Per-ticker long cap. Non-positive disables the cap.
    max_short: float = 0.2  # Per-ticker short cap in absolute value. Non-positive disables the cap.
    hold_decay: float = 1.0  # Decay applied to carried positions not refreshed by a new signal.


class PortfolioLayer:
    """Normalizes raw scores into portfolio weights and allocations."""

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()
        # Allow an environment override for quick sensitivity sweeps.
        env_max = os.getenv("PORTFOLIO_MAX_WEIGHT")
        if env_max:
            try:
                val = float(env_max)
                if val > 0:
                    self.config.max_long = val
                    self.config.max_short = val
            except ValueError:
                pass

    def allocate(
        self,
        raw_scores: Dict[str, float],
        prev_weights: Dict[str, float] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """Allocate capital by combining carried positions and current-day raw scores."""

        # Preserve signs so that long and short positions remain valid.
        prev_weights = {
            ticker: float(weight)
            for ticker, weight in (prev_weights or {}).items()
            if weight is not None
        }
        subset_scores = {
            ticker: float(score)
            for ticker, score in raw_scores.items()
            if score is not None
        }
        result_weights: Dict[str, float] = {}

        if not prev_weights and not subset_scores:
            return {}

        # Carry forward positions for tickers not refreshed by a current-day signal.
        keep_weights = {
            ticker: weight * self.config.hold_decay
            for ticker, weight in prev_weights.items()
            if ticker not in subset_scores
        }

        # Merge carried positions and current-day candidate weights.
        candidate = {**keep_weights, **subset_scores}
        total_abs = sum(abs(w) for w in candidate.values())
        if total_abs < self.config.epsilon:
            return {}

        # Normalize by total absolute exposure.
        result_weights = {t: w / total_abs for t, w in candidate.items()}

        # Apply long/short caps and renormalize by absolute exposure.
        if result_weights:
            for ticker in list(result_weights.keys()):
                w = result_weights[ticker]
                if w > 0 and self.config.max_long > 0:
                    w = min(w, self.config.max_long)
                if w < 0 and self.config.max_short > 0:
                    w = max(w, -self.config.max_short)
                result_weights[ticker] = w

            capped_abs = sum(abs(w) for w in result_weights.values())
            if capped_abs > self.config.epsilon:
                for ticker in list(result_weights.keys()):
                    result_weights[ticker] = result_weights[ticker] / capped_abs
            else:
                return {}

        allocations = {ticker: weight * self.config.capital for ticker, weight in result_weights.items()}
        return {
            ticker: {"weight": float(weight), "allocation": float(allocations[ticker])}
            for ticker, weight in result_weights.items()
        }
