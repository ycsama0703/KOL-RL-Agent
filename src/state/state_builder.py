"""Constructs RL state vectors from market, KOL, and portfolio inputs."""

from __future__ import annotations

from typing import Dict, List


class StateBuilder:
    """Concatenate heterogeneous features to one flat vector."""

    def build(self, market_features: Dict[str, float], kol_features: List[float], last_position: float) -> List[float]:
        state = list(kol_features)
        state.extend(market_features.values())
        state.append(last_position)
        return state
