"""Helpers for loading replay buffers into PyTorch datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator

import torch
from torch.utils.data import DataLoader, Dataset


class ReplayDataset(Dataset):
    def __init__(self, buffer_path: str | Path) -> None:
        data = torch.load(buffer_path)
        self.states = data["states"].float()
        self.actions = data["actions"].float()
        self.baseline_actions = data.get("baseline_actions", self.actions).float()
        # 如果存在组合层 reward（portfolio_rewards），优先使用；否则退回单票 reward_1d
        rewards_tensor = data.get("portfolio_rewards", data["rewards"])
        self.rewards = rewards_tensor.float()
        self.next_states = data["next_states"].float()
        self.dones = data["dones"].bool()

    def __len__(self) -> int:  # type: ignore[override]
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "state": self.states[idx],
            "action": self.actions[idx],  # 行为策略（基线签名权重）
            "baseline_action": self.baseline_actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.next_states[idx],
            "done": self.dones[idx],
        }


def create_dataloader(
    buffer_path: str | Path,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    dataset = ReplayDataset(buffer_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )


def load_buffer(buffer_path: str | Path) -> Dict[str, torch.Tensor]:
    data = torch.load(buffer_path)
    return data
