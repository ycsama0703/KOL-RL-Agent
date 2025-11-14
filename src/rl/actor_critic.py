"""Placeholder Actor-Critic network definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ActorOutput:
    target_position: float
    confidence: float


class Actor:
    """Returns deterministic positions given a state vector."""

    def act(self, state: List[float]) -> ActorOutput:
        avg = sum(state) / max(len(state), 1)
        return ActorOutput(target_position=max(min(avg, 1.0), -1.0), confidence=0.5)


class Critic:
    """Estimates value of a state-action pair."""

    def evaluate(self, state: List[float], action: float) -> float:
        return float(sum(state) + action)
