"""Inference entrypoint for converting KOL text to trading targets."""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from ..embedding.encoder import KOLTextEncoder
from ..preprocessing.text_cleaner import TextCleaner
from ..preprocessing.chunker import TextChunker
from ..state.state_builder import StateBuilder
from ..rl.actor_critic import Actor


class RLKolAgent:
    """Minimal agent wiring preprocessing, encoding, and actor inference."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.cleaner = TextCleaner()
        self.chunker = TextChunker()
        self.encoder = KOLTextEncoder()
        self.state_builder = StateBuilder()
        self.actor = Actor()
        self.last_position = 0.0

    def predict(self, kol_text: str, market_state: Dict[str, float]) -> Dict[str, float | str]:
        cleaned = self.cleaner.clean(kol_text)
        chunks = self.chunker.chunk(cleaned)
        kol_features = self.encoder.encode(chunks)
        state = self.state_builder.build(market_state, kol_features, self.last_position)
        actor_output = self.actor.act(state)
        self.last_position = actor_output.target_position
        return {
            "target_position": actor_output.target_position,
            "confidence": actor_output.confidence,
            "timestamp": datetime.utcnow().isoformat(),
        }
