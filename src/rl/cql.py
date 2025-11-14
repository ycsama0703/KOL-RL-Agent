"""Conservative Q-Learning trainer placeholder."""

from __future__ import annotations

from .actor_critic import Actor, Critic
from .buffer import ReplayBuffer


class CQLTrainer:
    def __init__(self, buffer: ReplayBuffer) -> None:
        self.buffer = buffer
        self.actor = Actor()
        self.critic = Critic()

    def train_epoch(self) -> None:
        for _ in range(len(self.buffer)):
            batch = self.buffer.sample(batch_size=1)
            if not batch:
                break
            state, action, reward, next_state, done = batch[0]
            _ = self.actor.act(state)
            _ = self.critic.evaluate(state, action)

    def export_policy(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("cql_policy_placeholder")
