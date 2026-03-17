"""Train generic RL benchmarks with the same run structure as train.py.

Supported methods:
- bc: vanilla single-head behavior cloning only
- iql: vanilla single-head IQL without BC warm start
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio.layer import PortfolioLayer
from src.training.data import ReplayDataset, load_buffer
from src.training.models import CriticNetwork, MLP, ValueNetwork
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
    method: str = "bc"
    kol: str = "youtube/Adam_Khoo"
    replay_dir: str = "data/multisource_ready_22-25/08_replay_buffer"
    checkpoints_dir: str = "models/checkpoints"
    output_dir: str = "outputs/benchmarks/generic_rl/bc"

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

    log_interval: int = 200
    write_iql_csv: bool = True
    progress_bar: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainingConfig:
    p = argparse.ArgumentParser(description="Train generic RL benchmark with current run structure.")
    p.add_argument("--method", choices=["bc", "iql"], required=True)
    p.add_argument("--kol", required=True, help="Replay subdir, e.g. youtube/Adam_Khoo")
    p.add_argument("--replay-dir", default="data/multisource_ready_22-25/08_replay_buffer")
    p.add_argument("--checkpoints-dir", default="models/checkpoints")
    p.add_argument("--output-dir", required=True)

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

    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--write-iql-csv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)

    args = p.parse_args()
    return TrainingConfig(
        method=args.method,
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
        log_interval=args.log_interval,
        write_iql_csv=args.write_iql_csv,
        progress_bar=args.progress_bar,
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
    return {"cumulative_return": cumulative_return, "sharpe": sharpe, "max_drawdown": max_drawdown}


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
        iterator = dataloader
        if cfg.progress_bar:
            iterator = tqdm(dataloader, desc=f"BC {ep + 1}/{cfg.bc_epochs}", leave=False)
        for batch in iterator:
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
    iql_metrics_csv_path: Path | None = None,
) -> None:
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    mse = nn.MSELoss()
    it = cycle(dataloader)

    csv_fp = None
    csv_writer = None
    if cfg.write_iql_csv and iql_metrics_csv_path is not None:
        iql_metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_fp = iql_metrics_csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(
            [
                "step",
                "critic_loss",
                "value_loss",
                "actor_loss",
                "reward_mean",
                "done_ratio",
                "adv_mean",
                "adv_std",
                "weight_mean",
                "weight_max",
                "policy_abs_mean",
                "behavior_abs_mean",
            ]
        )
        csv_fp.flush()

    rolling = {
        "critic_loss": 0.0,
        "value_loss": 0.0,
        "actor_loss": 0.0,
        "reward_mean": 0.0,
        "done_ratio": 0.0,
        "adv_mean": 0.0,
        "adv_std": 0.0,
        "weight_mean": 0.0,
        "weight_max": 0.0,
        "policy_abs_mean": 0.0,
        "behavior_abs_mean": 0.0,
    }
    rolling_n = 0

    steps_iter = range(1, cfg.iql_steps + 1)
    if cfg.progress_bar:
        steps_iter = tqdm(steps_iter, desc="IQL Training")

    actor.train()
    critic.train()
    value_net.train()

    for step in steps_iter:
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
        actor_loss = (weights * (policy_action - behavior_action).pow(2).squeeze(-1)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        rolling["critic_loss"] += float(critic_loss.item())
        rolling["value_loss"] += float(value_loss.item())
        rolling["actor_loss"] += float(actor_loss.item())
        rolling["reward_mean"] += float(reward.mean().item())
        rolling["done_ratio"] += float(done.mean().item())
        rolling["adv_mean"] += float(adv_b.mean().item())
        rolling["adv_std"] += float(adv_b.std(unbiased=False).item())
        rolling["weight_mean"] += float(weights.mean().item())
        rolling["weight_max"] += float(weights.max().item())
        rolling["policy_abs_mean"] += float(policy_action.abs().mean().item())
        rolling["behavior_abs_mean"] += float(behavior_action.abs().mean().item())
        rolling_n += 1

        if step % max(cfg.log_interval, 1) == 0 or step == cfg.iql_steps:
            avg = {k: v / max(rolling_n, 1) for k, v in rolling.items()}
            LOGGER.info(
                "IQL step %d/%d | critic=%.6f value=%.6f actor=%.6f reward=%.6f done=%.4f adv=%.6f(+/-%.6f) w=%.6f(max=%.4f) | |a|=%.6f |beh|=%.6f",
                step,
                cfg.iql_steps,
                avg["critic_loss"],
                avg["value_loss"],
                avg["actor_loss"],
                avg["reward_mean"],
                avg["done_ratio"],
                avg["adv_mean"],
                avg["adv_std"],
                avg["weight_mean"],
                avg["weight_max"],
                avg["policy_abs_mean"],
                avg["behavior_abs_mean"],
            )
            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        step,
                        avg["critic_loss"],
                        avg["value_loss"],
                        avg["actor_loss"],
                        avg["reward_mean"],
                        avg["done_ratio"],
                        avg["adv_mean"],
                        avg["adv_std"],
                        avg["weight_mean"],
                        avg["weight_max"],
                        avg["policy_abs_mean"],
                        avg["behavior_abs_mean"],
                    ]
                )
                csv_fp.flush()

            for key in rolling:
                rolling[key] = 0.0
            rolling_n = 0

    if csv_fp is not None:
        csv_fp.close()


def evaluate(actor: VanillaActorNetwork, buffer_path: Path, device: torch.device) -> Dict[str, float]:
    actor.eval()
    buffer = load_buffer(buffer_path)

    states = buffer["states"].float()
    rewards = buffer["rewards"].float().cpu().numpy()
    dates = buffer["meta"]["published_at"]
    tickers = buffer["meta"]["ticker"]

    preds = []
    with torch.no_grad():
        for start in range(0, states.size(0), 1024):
            s = states[start : start + 1024].to(device)
            preds.append(actor(s).squeeze(-1).cpu().numpy())
    raw_scores = np.concatenate(preds, axis=0)

    df = pd.DataFrame({"date": dates, "ticker": tickers, "reward": rewards, "raw_score": raw_scores})
    portfolio = PortfolioLayer()

    daily_returns = []
    for date, group in df.groupby("date"):
        raw_dict = {row["ticker"]: row["raw_score"] for _, row in group.iterrows()}
        weights = portfolio.allocate(raw_dict)

        day_ret = 0.0
        for _, row in group.iterrows():
            w = weights.get(row["ticker"], {"weight": 0.0})["weight"]
            day_ret += w * float(row["reward"])
        daily_returns.append(day_ret)

    if not daily_returns:
        return {"cumulative_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    return compute_metrics(np.array(daily_returns))


def save_checkpoints(actor: VanillaActorNetwork, critic: CriticNetwork, value_net: ValueNetwork, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), checkpoint_dir / "actor.pt")
    torch.save(critic.state_dict(), checkpoint_dir / "critic.pt")
    torch.save(value_net.state_dict(), checkpoint_dir / "value.pt")
    torch.save({"actor_state_dict": actor.state_dict()}, checkpoint_dir / "policy.pt")
    LOGGER.info("Saved checkpoints to %s", checkpoint_dir)


def split_group_name(group: str) -> Tuple[str | None, str]:
    if "/" in group:
        source, kol = group.split("/", 1)
        return source, kol
    return None, group


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device)

    train_path = Path(cfg.replay_dir) / cfg.kol / "train.pt"
    val_path = Path(cfg.replay_dir) / cfg.kol / "val.pt"

    source, kol_name = split_group_name(cfg.kol)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{kol_name}_{timestamp}"
    output_root = Path(cfg.output_dir)
    run_dir = output_root / source / run_name if source else output_root / run_name
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / Path(cfg.checkpoints_dir).name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "training.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(fh)

    LOGGER.info("Starting generic benchmark run %s", run_name)
    LOGGER.info("Method: %s", cfg.method)
    LOGGER.info("Group: %s", cfg.kol)
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Checkpoints will be saved under %s", checkpoint_dir)
    LOGGER.info("Training config: %s", json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True))
    if torch.cuda.is_available() and device.type == "cuda":
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(device))
    LOGGER.info("Device: %s", device)

    if not train_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {train_path}")

    train_dataset = ReplayDataset(train_path)
    state_dim = int(train_dataset.states.shape[1])
    LOGGER.info("Loaded replay buffer for %s with %d samples, state_dim=%d", cfg.kol, len(train_dataset), state_dim)
    LOGGER.info(
        "Dataset stats | baseline_abs_mean=%.6f behavior_abs_mean=%.6f",
        float(train_dataset.baseline_actions.abs().mean().item()),
        float(train_dataset.actions.abs().mean().item()),
    )

    bc_loader = DataLoader(train_dataset, batch_size=cfg.bc_batch_size, shuffle=True, drop_last=True, pin_memory=True)
    iql_loader = DataLoader(train_dataset, batch_size=cfg.iql_batch_size, shuffle=True, drop_last=True, pin_memory=True)

    actor = VanillaActorNetwork(state_dim).to(device)
    critic = CriticNetwork(state_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)

    bc_loss = 0.0
    if cfg.method == "bc":
        bc_loss = behavior_cloning(actor, bc_loader, cfg, device)
        LOGGER.info("Behavior cloning finished. Avg loss=%.6f", bc_loss)
    elif cfg.method == "iql":
        iql_metrics_csv_path = log_dir / "iql_metrics.csv"
        if cfg.write_iql_csv:
            LOGGER.info("IQL metrics CSV: %s", iql_metrics_csv_path)
        iql_training(
            actor,
            critic,
            value_net,
            iql_loader,
            cfg,
            device,
            iql_metrics_csv_path=iql_metrics_csv_path,
        )
    else:
        raise ValueError(f"Unsupported method: {cfg.method}")

    metrics = {}
    if val_path.exists():
        metrics = evaluate(actor, val_path, device)
        LOGGER.info(
            "Validation metrics for %s: cumulative_return=%.4f, sharpe=%.4f, max_drawdown=%.4f",
            cfg.kol,
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
        "method": cfg.method,
        "group": cfg.kol,
        "source": source,
        "kol": kol_name,
        "train_samples": len(train_dataset),
        "bc_loss": bc_loss,
        "metrics": metrics,
        "config": asdict(cfg),
    }
    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    LOGGER.info("Saved run summary to %s", summary_path)


if __name__ == "__main__":
    main()
