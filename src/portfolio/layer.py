"""Portfolio layer that maintains cumulative positions per KOL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PortfolioState:
    """Tracks the current portfolio allocation for a single KOL."""

    capital: float = 10_000.0
    positions: Dict[str, float] = field(default_factory=dict)  # weight per ticker (sums to 1)

    def as_dollars(self) -> Dict[str, float]:
        """Return allocations expressed in dollars."""
        return {symbol: weight * self.capital for symbol, weight in self.positions.items()}


class PortfolioLayer:
    """Accumulates dynamic stock pools and converts raw scores to weights."""

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        use_signed_weights: bool = False,
        carry_unmentioned: bool = True,
        min_abs_score: float = 1e-8,
    ) -> None:
        self.initial_capital = initial_capital
        self.use_signed_weights = use_signed_weights
        self.carry_unmentioned = carry_unmentioned
        self.min_abs_score = min_abs_score

    def allocate(self, raw_scores: Dict[str, float], state: PortfolioState | None = None) -> PortfolioState:
        """Merge cumulative stock pool with latest raw scores and return new positions."""
        current_state = state or PortfolioState(capital=self.initial_capital)
        combined_scores: Dict[str, float] = {}

        if self.carry_unmentioned:
            for symbol, weight in current_state.positions.items():
                if symbol not in raw_scores:
                    # Treat previous weights as pseudo scores so they keep their share unless overwritten.
                    combined_scores[symbol] = weight

        for symbol, score in raw_scores.items():
            combined_scores[symbol] = score

        weights = self._scores_to_weights(combined_scores)
        return PortfolioState(capital=current_state.capital, positions=weights)

    def _scores_to_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        filtered = {symbol: score for symbol, score in scores.items() if abs(score) >= self.min_abs_score}
        if not filtered:
            return {}
        if self.use_signed_weights:
            total = sum(abs(score) for score in filtered.values())
            if total == 0.0:
                return {}
            return {symbol: score / total for symbol, score in filtered.items()}

        total = sum(abs(score) for score in filtered.values())
        if total == 0.0:
            return {}
        weights = {
            symbol: abs(score) / total
            for symbol, score in filtered.items()
        }
        return {symbol: weight for symbol, weight in weights.items() if weight > 0.0}
