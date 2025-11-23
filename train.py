"""Train script for BC → IQL with ModernBERT + ticker embedding states."""

from __future__ import annotations

import argparse
import math
import json
import logging
from dataclasses import asdict, dataclass
from itertools import cycle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.portfolio.layer import PortfolioLayer
from src.state.ticker_embedding import TickerEmbedding
from src.training.data import ReplayDataset, load_buffer
from src.training.models import ActorNetwork, CriticNetwork, ValueNetwork
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass
class TrainingConfig:
    kol: str = "Everything_Money"
    replay_dir: str = "data/replay_buffer"
    ticker_vocab: str = "models/embedding/ticker_vocab.json"
    ticker_embedding: str = "models/embedding/ticker_embedding.pt"
    checkpoints_dir: str = "models/checkpoints"
    output_dir: str = "outputs"
    bc_epochs: int = 10
    bc_batch_size: int = 256
    bc_lr: float = 3e-4
    iql_steps: int = 200_000
    iql_batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    gamma: float = 0.99
    expectile: float = 0.7
    temperature_beta: float = 3.0
    fidelity_lambda: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train KOL agent with BC + IQL.")
    parser.add_argument("--kol", default="Everything_Money", help="KOL name (directory under replay buffer).")
    parser.add_argument("--replay-dir", default="data/replay_buffer", help="Replay buffer root directory.")
    parser.add_argument("--ticker-vocab", default="models/embedding/ticker_vocab.json", help="Ticker vocab path.")
    parser.add_argument("--ticker-embedding", default="models/embedding/ticker_embedding.pt", help="Ticker embedding weights.")
    parser.add_argument("--checkpoints-dir", default="models/checkpoints", help="Directory to store checkpoints.")
    parser.add_argument("--output-dir", default="outputs", help="Root directory to store training outputs.")
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--iql-steps", type=int, default=200_000)
    parser.add_argument("--iql-batch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature-beta", type=float, default=3.0)
    parser.add_argument("--fidelity-lambda", type=float, default=0.1, help="Weight for fidelity reward shaping.")
    args = parser.parse_args()
    return TrainingConfig(
        kol=args.kol,
        replay_dir=args.replay_dir,
        ticker_vocab=args.ticker_vocab,
        ticker_embedding=args.ticker_embedding,
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
        fidelity_lambda=args.fidelity_lambda,
    )


def behavior_cloning(
    actor: ActorNetwork,
    dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(actor.parameters(), lr=config.bc_lr)
    actor.train()
    total_loss = 0.0
    steps = 0
    for epoch in range(config.bc_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"BC Epoch {epoch+1}/{config.bc_epochs}", leave=False):
            states = batch["state"].to(device)
            baseline_actions = batch["action"].to(device)  # 基线签名权重
            delta = actor(states)  # 学习偏移量
            preds = baseline_actions + delta  # 最终动作 = 基线 + 残差

            # 约束残差，BC 训练希望贴合基线（残差→0）
            loss = criterion(preds, baseline_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        LOGGER.info("BC Epoch %d/%d - loss=%.6f", epoch + 1, config.bc_epochs, epoch_loss / max(len(dataloader), 1))
        total_loss += epoch_loss
    return total_loss / max(steps, 1)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()


def iql_training(
    actor: ActorNetwork,
    critic: CriticNetwork,
    value_net: ValueNetwork,
    dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    actor_opt = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=config.value_lr)

    mse_loss = nn.MSELoss()
    iterator = cycle(dataloader)
    actor.train()
    critic.train()
    value_net.train()

    for step in tqdm(range(1, config.iql_steps + 1), desc="IQL Training"):
        batch = next(iterator)
        states = batch["state"].to(device)
        baseline_actions = batch["action"].to(device)  # 行为数据 = 基线动作（签名权重）
        rewards = batch["reward"].to(device)
        next_states = batch["next_state"].to(device)
        dones = batch["done"].to(device).float()

        # Policy: 基线 + 残差
        delta = actor(states)
        policy_actions = baseline_actions + delta

        fidelity_penalty = (policy_actions.detach() - baseline_actions).pow(2).squeeze(-1)
        reward_aug = rewards - config.fidelity_lambda * fidelity_penalty

        with torch.no_grad():
            next_values = value_net(next_states)
            target_q = reward_aug + config.gamma * (1 - dones) * next_values

        # Q 学习在行为数据上（稳定）
        critic_pred = critic(states, baseline_actions)
        critic_loss = mse_loss(critic_pred, target_q)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        with torch.no_grad():
            q_sa = critic(states, baseline_actions)
        value_pred = value_net(states)
        value_loss = expectile_loss(q_sa - value_pred, config.expectile)
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        q_pi = critic(states, policy_actions)
        with torch.no_grad():
            v = value_net(states)
            advantages = q_pi - v
            weights = torch.clamp(torch.exp(config.temperature_beta * advantages), max=100.0)
        actor_loss = (weights * (policy_actions - baseline_actions).pow(2)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        if step % 1000 == 0:
            LOGGER.info(
                "IQL step %d/%d - critic=%.6f value=%.6f actor=%.6f",
                step,
                config.iql_steps,
                critic_loss.item(),
                value_loss.item(),
                actor_loss.item(),
            )


def compute_metrics(daily_returns: np.ndarray) -> Dict[str, float]:
    cumulative_return = float(np.prod(1 + daily_returns) - 1)
    sharpe = 0.0
    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-8:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252))
    equity = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / (peak + 1e-8)
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    return {
        "cumulative_return": cumulative_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def evaluate(actor: ActorNetwork, buffer_path: Path, device: torch.device) -> Dict[str, float]:
    actor.eval()
    buffer = load_buffer(buffer_path)
    states = buffer["states"]
    rewards = buffer["rewards"].numpy()
    baseline_actions = buffer["actions"].numpy()  # 签名基线权重
    dates = buffer["meta"]["published_at"]
    tickers = buffer["meta"]["ticker"]

    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, states.size(0), 1024):
            batch = states[start : start + 1024].to(device)
            preds.append(actor(batch).squeeze(-1).cpu())
    delta = torch.cat(preds).numpy()
    raw_scores = baseline_actions.squeeze(-1) + delta  # 基线 + 残差

    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": tickers,
            "reward": rewards,
            "raw_score": raw_scores,
        }
    )
    portfolio = PortfolioLayer()
    daily_returns: list[float] = []
    for date, group in df.groupby("date"):
        raw_dict = {row["ticker"]: row["raw_score"] for _, row in group.iterrows()}
        weights = portfolio.allocate(raw_dict)
        day_return = 0.0
        for _, row in group.iterrows():
            ticker = row["ticker"]
            reward = row["reward"]
            weight = weights.get(ticker, {"weight": 0.0})["weight"]
            day_return += weight * reward
        daily_returns.append(day_return)

    if not daily_returns:
        return {"cumulative_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    return compute_metrics(np.array(daily_returns))


def save_checkpoints(
    actor: ActorNetwork,
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
    config = parse_args()
    device = torch.device(config.device)
    train_path = Path(config.replay_dir) / config.kol / "train.pt"
    val_path = Path(config.replay_dir) / config.kol / "val.pt"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.kol}_{timestamp}"
    run_dir = Path(config.output_dir) / run_name
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / Path(config.checkpoints_dir).name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training.log"
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.info("Starting training run %s", run_name)
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Checkpoints will be saved under %s", checkpoint_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {train_path}")

    train_dataset = ReplayDataset(train_path)
    state_dim = train_dataset.states.shape[1]
    LOGGER.info("Loaded replay buffer for %s with %d samples, state_dim=%d", config.kol, len(train_dataset), state_dim)

    bc_loader = DataLoader(
        train_dataset,
        batch_size=config.bc_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    iql_loader = DataLoader(
        train_dataset,
        batch_size=config.iql_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    actor = ActorNetwork(state_dim).to(device)
    critic = CriticNetwork(state_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)

    bc_loss = behavior_cloning(actor, bc_loader, config, device)
    LOGGER.info("Behavior cloning finished. Avg loss=%.6f", bc_loss)

    iql_training(actor, critic, value_net, iql_loader, config, device)

    metrics = {}
    if val_path.exists():
        metrics = evaluate(actor, val_path, device)
        LOGGER.info(
            "Validation metrics for %s: cumulative_return=%.4f, sharpe=%.4f, max_drawdown=%.4f",
            config.kol,
            metrics["cumulative_return"],
            metrics["sharpe"],
            metrics["max_drawdown"],
        )
    else:
        LOGGER.warning("Validation buffer %s not found; skipping evaluation.", val_path)

    save_checkpoints(actor, critic, value_net, checkpoint_dir)

    summary = {
        "run_name": run_name,
        "timestamp": timestamp,
        "kol": config.kol,
        "train_samples": len(train_dataset),
        "bc_loss": bc_loss,
        "metrics": metrics,
        "config": asdict(config),
    }
    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    LOGGER.info("Saved run summary to %s", summary_path)


if __name__ == "__main__":
    main()
