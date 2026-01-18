"""Parallel training script for vanilla IQL (single-head actor)."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.data import ReplayDataset
from src.training.models import MLP, CriticNetwork, ValueNetwork
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class VanillaActorNetwork(nn.Module):
    """Standard single-head actor producing absolute actions."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.backbone = MLP(
            input_dim=state_dim,
            hidden_dims=(512, 512, 256),
            output_dim=256,
            output_activation=nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(256, 1), nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(state))


@dataclass
class TrainingConfig:
    kol: str = "Everything_Money"
    replay_dir: str = "data/replay_buffer"
    checkpoints_dir: str = "models/checkpoints"
    output_dir: str = "outputs"

    # BC
    bc_epochs: int = 10
    bc_batch_size: int = 256
    bc_lr: float = 3e-4

    # IQL
    iql_steps: int = 200_000
    iql_batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    gamma: float = 0.99
    expectile: float = 0.7
    temperature_beta: float = 3.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainingConfig:
    p = argparse.ArgumentParser(description="Train vanilla IQL agent (single-head actor).")
    p.add_argument("--kol", default="Everything_Money")
    p.add_argument("--replay-dir", default="data/replay_buffer")
    p.add_argument("--checkpoints-dir", default="models/checkpoints")
    p.add_argument("--output-dir", default="outputs")

    p.add_argument("--bc-epochs", type=int, default=10)
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--bc-lr", type=float, default=3e-4)

    p.add_argument("--iql-steps", type=int, default=200_000)
    p.add_argument("--iql-batch-size", type=int, default=256)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--temperature-beta", type=float, default=3.0)

    args = p.parse_args()
    return TrainingConfig(
        kol=args.kol,
        replay_dir=args.replay_dir,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_lr=args.bc_lr,
        iql_steps=args.iql_steps,
        iql_batch_size=args.iql_batch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        gamma=args.gamma,
        expectile=args.expectile,
        temperature_beta=args.temperature_beta,
    )


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()


def behavior_cloning(
    actor: VanillaActorNetwork,
    dataloader: DataLoader,
    cfg: TrainingConfig,
    device: torch.device,
) -> float:
    mse = nn.MSELoss()
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.bc_lr)
    actor.train()

    total = 0.0
    steps = 0
    for ep in range(cfg.bc_epochs):
        ep_losses = []
        for batch in tqdm(dataloader, desc=f"BC {ep+1}/{cfg.bc_epochs}", leave=False):
            state = batch["state"].to(device)
            behavior_action = batch["action"].to(device)
            pred = actor(state)
            loss = mse(pred, behavior_action)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
            total += loss.item()
            steps += 1
        LOGGER.info("BC epoch %d/%d - loss=%.6f", ep + 1, cfg.bc_epochs, float(np.mean(ep_losses)))
    return total / max(steps, 1)


def iql_training(
    actor: VanillaActorNetwork,
    critic: CriticNetwork,
    value_net: ValueNetwork,
    dataloader: DataLoader,
    cfg: TrainingConfig,
    device: torch.device,
) -> None:
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    mse = nn.MSELoss()
    it = cycle(dataloader)

    actor.train()
    critic.train()
    value_net.train()

    for step in tqdm(range(1, cfg.iql_steps + 1), desc="IQL Training"):
        batch = next(it)
        state = batch["state"].to(device)
        next_state = batch["next_state"].to(device)
        behavior_action = batch["action"].to(device)
        reward = batch["reward"].to(device)
        done = batch["done"].to(device).float()

        with torch.no_grad():
            next_v = value_net(next_state).squeeze(-1)
            target_q = reward + cfg.gamma * (1.0 - done) * next_v

        q_pred = critic(state, behavior_action).squeeze(-1)
        critic_loss = mse(q_pred, target_q)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        with torch.no_grad():
            q_sa = critic(state, behavior_action).squeeze(-1)
        v_pred = value_net(state).squeeze(-1)
        value_loss = expectile_loss(q_sa - v_pred, cfg.expectile)
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        with torch.no_grad():
            q_b = critic(state, behavior_action).squeeze(-1)
            v = value_net(state).squeeze(-1)
            adv_b = q_b - v
            weights = torch.clamp(torch.exp(cfg.temperature_beta * adv_b), max=100.0)

        policy_action = actor(state)
        loss_fit = (weights * (policy_action - behavior_action).pow(2).squeeze(-1)).mean()

        actor_opt.zero_grad()
        loss_fit.backward()
        actor_opt.step()

        if step % 1000 == 0:
            LOGGER.info(
                "IQL step %d/%d - critic=%.6f value=%.6f actor=%.6f",
                step,
                cfg.iql_steps,
                critic_loss.item(),
                value_loss.item(),
                loss_fit.item(),
            )


def save_checkpoints(
    actor: VanillaActorNetwork,
    critic: CriticNetwork,
    value_net: ValueNetwork,
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), checkpoint_dir / "actor.pt")
    torch.save(critic.state_dict(), checkpoint_dir / "critic.pt")
    torch.save(value_net.state_dict(), checkpoint_dir / "value.pt")
    torch.save({"actor_state_dict": actor.state_dict()}, checkpoint_dir / "policy.pt")
    LOGGER.info("Saved checkpoints to %s", checkpoint_dir)


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device)

    train_path = Path(cfg.replay_dir) / cfg.kol / "train.pt"
    val_path = Path(cfg.replay_dir) / cfg.kol / "val.pt"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.kol}_{timestamp}"
    run_dir = Path(cfg.output_dir) / run_name
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / Path(cfg.checkpoints_dir).name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "training.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(fh)

    LOGGER.info("Starting vanilla IQL run %s", run_name)
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Checkpoints will be saved under %s", checkpoint_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {train_path}")

    train_dataset = ReplayDataset(train_path)
    state_dim = int(train_dataset.states.shape[1])
    LOGGER.info("Loaded replay buffer for %s with %d samples, state_dim=%d", cfg.kol, len(train_dataset), state_dim)

    bc_loader = DataLoader(train_dataset, batch_size=cfg.bc_batch_size, shuffle=True, drop_last=True, pin_memory=True)
    iql_loader = DataLoader(train_dataset, batch_size=cfg.iql_batch_size, shuffle=True, drop_last=True, pin_memory=True)

    actor = VanillaActorNetwork(state_dim).to(device)
    critic = CriticNetwork(state_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)

    if cfg.bc_epochs > 0:
        bc_loss = behavior_cloning(actor, bc_loader, cfg, device)
        LOGGER.info("Behavior cloning finished. Avg loss=%.6f", bc_loss)
    else:
        bc_loss = 0.0

    if cfg.iql_steps > 0:
        iql_training(actor, critic, value_net, iql_loader, cfg, device)

    save_checkpoints(actor, critic, value_net, checkpoint_dir)

    summary = {
        "run_name": run_name,
        "timestamp": timestamp,
        "kol": cfg.kol,
        "train_samples": len(train_dataset),
        "bc_loss": bc_loss,
        "config": asdict(cfg),
    }
    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    LOGGER.info("Saved run summary to %s", summary_path)


if __name__ == "__main__":
    main()
