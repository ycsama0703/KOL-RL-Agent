"""Simple replay buffer for offline RL experiments."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Tuple

Transition = Tuple[list[float], float, float, list[float], bool]


class ReplayBuffer:
    """Stores state transitions for offline learning."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.storage: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        batch = list(self.storage)[:batch_size]
        return batch

    def __len__(self) -> int:  # pragma: no cover
        return len(self.storage)
