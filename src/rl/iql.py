"""Implicit Q-Learning trainer placeholder."""

from __future__ import annotations

from typing import List

from .actor_critic import Actor, Critic
from .buffer import ReplayBuffer


class IQLTrainer:
    def __init__(self, buffer: ReplayBuffer) -> None:
        self.buffer = buffer
        self.actor = Actor()
        self.critic = Critic()

    def train_step(self) -> None:
        if not self.buffer.storage:
            return
        batch = self.buffer.sample(batch_size=1)
        state, action, reward, next_state, done = batch[0]
        _ = self.actor.act(state)
        _ = self.critic.evaluate(state, action)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("iql_checkpoint_placeholder")
