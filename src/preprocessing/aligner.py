"""Utilities for aligning KOL text chunks with market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class AlignedSample:
    timestamp: datetime
    kol_features: List[float]
    market_features: Dict[str, float]


class MarketAligner:
    """Aligns semantic features with current market state."""

    def align(self, kol_features: List[float], market_state: Dict[str, float]) -> AlignedSample:
        return AlignedSample(
            timestamp=datetime.utcnow(),
            kol_features=kol_features,
            market_features=market_state,
        )
