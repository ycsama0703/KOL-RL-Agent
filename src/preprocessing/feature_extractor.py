"""Extracts handcrafted sentiment or intensity signals from text chunks."""

from __future__ import annotations

from typing import Dict


class FeatureExtractor:
    """Placeholder extractor that returns dummy statistical features."""

    def extract(self, text: str) -> Dict[str, float]:
        length = len(text)
        return {
            "length": float(length),
            "exclamation_ratio": text.count("!") / max(length, 1),
        }
