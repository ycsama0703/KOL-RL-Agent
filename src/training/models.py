"""Neural network modules for actor, critic, and value networks."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class MLP(nn.Module):
    """Generic multilayer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        output_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorNetwork(nn.Module):
    """Deterministic actor with two heads: signal scaling and no-signal decay."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.backbone = MLP(input_dim=state_dim, hidden_dims=(512, 512, 256), output_dim=256, output_activation=nn.ReLU())
        self.head_signal = nn.Sequential(nn.Linear(256, 1), nn.Tanh())    # for has_signal: same-direction scaling
        self.head_decay = nn.Sequential(nn.Linear(256, 1), nn.Tanh())     # for no-signal: decay factor

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(state)
        return {
            "delta_signal": self.head_signal(h),  # [-1,1]
            "delta_decay": self.head_decay(h),    # [-1,1]
        }


class CriticNetwork(nn.Module):
    """Q-network taking state-action pairs."""

    def __init__(self, state_dim: int, action_dim: int = 1) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=state_dim + action_dim,
            hidden_dims=(512, 512, 256),
            output_dim=1,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ValueNetwork(nn.Module):
    """State value estimator."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=state_dim,
            hidden_dims=(512, 512, 256),
            output_dim=1,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
