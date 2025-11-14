"""Encodes KOL text into dense vectors using placeholder logic."""

from __future__ import annotations

from typing import List


class KOLTextEncoder:
    """Simple averaging encoder until a transformer backend is integrated."""

    def encode(self, chunks: List[str]) -> List[float]:
        if not chunks:
            return []
        avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
        return [avg_length]
