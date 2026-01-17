"""Utilities for mapping free-form company names to US ticker symbols."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


STOP_WORDS = {
    "inc",
    "inc.",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "co.",
    "limited",
    "ltd",
    "plc",
    "the",
    "group",
    "holdings",
}


@dataclass
class CompanyTickerMapper:
    """Loads a reference table (e.g. top 500 excel) and exposes fuzzy lookup."""

    reference_path: Path
    manual_overrides: Dict[str, Optional[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference_path.suffix.lower() == ".csv":
            df = pd.read_csv(self.reference_path)
        else:
            df = pd.read_excel(self.reference_path)

        col_map = {col.strip().upper(): col for col in df.columns}
        if {"SYMBOL", "COMPANY NAME"}.issubset(col_map):
            symbol_col = col_map["SYMBOL"]
            name_col = col_map["COMPANY NAME"]
        elif {"TICKER", "COMNAM"}.issubset(col_map):
            symbol_col = col_map["TICKER"]
            name_col = col_map["COMNAM"]
        else:
            raise ValueError(
                "Reference file missing required columns. "
                "Expected (Symbol, Company Name) or (TICKER, COMNAM)."
            )

        self._mapping: Dict[str, str] = {}
        for _, row in df.iterrows():
            symbol = str(row[symbol_col]).strip().upper()
            name = self._normalize(str(row[name_col]))
            if not name:
                continue
            if not symbol or symbol == "NAN":
                continue
            self._mapping[name] = symbol

        for key, value in self.manual_overrides.items():
            norm_key = self._normalize(key)
            if value is None:
                self._mapping.pop(norm_key, None)
            elif norm_key:
                self._mapping[norm_key] = value.strip().upper()

        self._keys = list(self._mapping.keys())

    def lookup(self, name: str | None) -> Optional[str]:
        if not name or not isinstance(name, str):
            return None
        normalized = self._normalize(name)
        if not normalized:
            return None
        if normalized in self._mapping:
            return self._mapping[normalized]

        matches = difflib.get_close_matches(normalized, self._keys, n=1, cutoff=0.82)
        if matches:
            return self._mapping[matches[0]]
        return None

    @staticmethod
    def _normalize(name: str) -> str:
        lowered = name.lower()
        lowered = lowered.replace("&", " and ")
        cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
        tokens = [token for token in cleaned.split() if token and token not in STOP_WORDS]
        return " ".join(tokens)
