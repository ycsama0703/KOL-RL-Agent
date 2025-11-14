"""High level training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .buffer import ReplayBuffer
from .cql import CQLTrainer
from .iql import IQLTrainer


@dataclass
class TrainerConfig:
    algorithm: Literal["iql", "cql"] = "iql"
    checkpoint_path: str = "models/checkpoints/default/policy.pt"


class RLTrainer:
    def __init__(self, config: TrainerConfig, buffer: ReplayBuffer) -> None:
        self.config = config
        self.buffer = buffer
        self.trainer = self._build_trainer()

    def _build_trainer(self):
        if self.config.algorithm == "cql":
            return CQLTrainer(self.buffer)
        return IQLTrainer(self.buffer)

    def run(self) -> None:
        if isinstance(self.trainer, CQLTrainer):
            self.trainer.train_epoch()
            self.trainer.export_policy(self.config.checkpoint_path)
        else:
            self.trainer.train_step()
            self.trainer.save(self.config.checkpoint_path)
