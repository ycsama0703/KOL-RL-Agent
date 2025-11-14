"""Split cleaned text into manageable chunks for embedding."""

from __future__ import annotations

from typing import List


class TextChunker:
    """Naive chunker that splits text by period or configurable length."""

    def __init__(self, max_words: int = 128) -> None:
        self.max_words = max_words

    def chunk(self, text: str) -> List[str]:
        words = text.split()
        return [
            " ".join(words[i : i + self.max_words])
            for i in range(0, len(words), self.max_words)
        ]
