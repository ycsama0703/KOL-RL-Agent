"""Utility for cleaning and normalizing KOL text inputs."""

from __future__ import annotations

import re


class TextCleaner:
    """Performs basic text normalization before downstream processing."""

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def clean(self, text: str) -> str:
        """Apply simple regex rules to normalize whitespace/punctuation."""
        normalized = text.strip()
        normalized = re.sub(r"\s+", " ", normalized)
        if self.lowercase:
            normalized = normalized.lower()
        return normalized
