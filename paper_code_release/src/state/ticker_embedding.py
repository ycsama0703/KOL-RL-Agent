"""Ticker embedding utilities using a learned nn.Embedding layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn


@dataclass
class TickerVocabulary:
    """Utility class that maps ticker symbols to integer ids."""

    token_to_id: Dict[str, int]

    PAD_TOKEN: str = "<pad>"
    UNK_TOKEN: str = "<unk>"

    def __init__(self, tokens: Iterable[str] | None = None) -> None:
        base = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        if tokens:
            for token in tokens:
                normalized = self._normalize(token)
                if normalized and normalized not in base:
                    base[normalized] = len(base)
        self.token_to_id = base

    @staticmethod
    def _normalize(ticker: str) -> str:
        return ticker.strip().upper()

    def __len__(self) -> int:
        return len(self.token_to_id)

    def add(self, ticker: str) -> int:
        normalized = self._normalize(ticker)
        if not normalized:
            return self.token_to_id[self.UNK_TOKEN]
        if normalized not in self.token_to_id:
            self.token_to_id[normalized] = len(self.token_to_id)
        return self.token_to_id[normalized]

    def lookup(self, ticker: str) -> int:
        normalized = self._normalize(ticker)
        return self.token_to_id.get(normalized, self.token_to_id[self.UNK_TOKEN])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TickerVocabulary":
        token_to_id = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls()
        vocab.token_to_id = token_to_id
        return vocab


class TickerEmbedding(nn.Module):
    """Learned embedding layer for ticker ids."""

    def __init__(self, vocab: TickerVocabulary, embedding_dim: int = 32) -> None:
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim, padding_idx=0)

    def forward(self, tickers: List[str]) -> torch.Tensor:
        ids = torch.tensor([self.vocab.lookup(ticker) for ticker in tickers], dtype=torch.long, device=self.embedding.weight.device)
        return self.embedding(ids)

    def encode_single(self, ticker: str) -> List[float]:
        with torch.no_grad():
            vector = self.forward([ticker])[0]
        return vector.cpu().tolist()

    def save(self, model_path: str | Path, vocab_path: str | Path) -> None:
        torch.save(self.state_dict(), model_path)
        self.vocab.save(vocab_path)

    @classmethod
    def load(cls, model_path: str | Path, vocab_path: str | Path, embedding_dim: int = 32) -> "TickerEmbedding":
        vocab = TickerVocabulary.load(vocab_path)
        model = cls(vocab, embedding_dim=embedding_dim)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        return model
