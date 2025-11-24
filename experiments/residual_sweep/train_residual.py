"""Residual policy training: action = baseline * (1 + residual_scale * tanh(delta))."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

# add repo root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.training.data import ReplayDataset, load_buffer
from src.training.models import ActorNetwork, CriticNetwork, ValueNetwork
from src.portfolio.layer import PortfolioConfig, PortfolioLayer
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TrainingConfig:
    kol: str = "Everything_Money"
    replay_dir: str = "data/replay_buffer_residual"
    ticker_vocab: str = "models/embedding/ticker_vocab.json"
    ticker_embedding: str = "models/embedding/ticker_embedding.pt"
    checkpoints_dir: str = "models/checkpoints"
    output_dir: str = "outputs"
    bc_epochs: int = 1
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
    fidelity_lambda: float = 0.3
    residual_scale: float = 0.2  # 控制有信号分支的同向缩放幅度
    decay_scale: float = 0.5  # 控制无信号分支的衰减幅度（decay = sigmoid(scale * delta_decay)）
    max_weight: float = 0.2
    hold_decay: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainingConfig:
    p = argparse.ArgumentParser(description="Train residual policy on top of baseline.")
    p.add_argument("--kol", default="Everything_Money")
    p.add_argument("--replay-dir", default="data/replay_buffer_residual")
    p.add_argument("--ticker-vocab", default="models/embedding/ticker_vocab.json")
    p.add_argument("--ticker-embedding", default="models/embedding/ticker_embedding.pt")
    p.add_argument("--checkpoints-dir", default="models/checkpoints")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--bc-epochs", type=int, default=1)
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
    p.add_argument("--fidelity-lambda", type=float, default=0.3)
    p.add_argument("--residual-scale", type=float, default=0.2)
    p.add_argument("--decay-scale", type=float, default=0.5)
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--hold-decay", type=float, default=1.0)
    p.add_argument("--device", default=None, help="Force device (cuda/cpu). If None, auto-detect.")
    args = p.parse_args()
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
        residual_scale=args.residual_scale,
        decay_scale=args.decay_scale,
        max_weight=args.max_weight,
        hold_decay=args.hold_decay,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()


def behavior_cloning(actor, loader, cfg: TrainingConfig, device):
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.bc_lr)
    actor.train()
    total = 0.0
    steps = 0
    for epoch in range(cfg.bc_epochs):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"BC {epoch+1}/{cfg.bc_epochs}", leave=False):
            states = batch["state"].to(device)
            baseline_actions = batch["action"].to(device)
            out = actor(states)
            delta_sig = out["delta_signal"]
            delta_dec = out["delta_decay"]
            has_signal = (baseline_actions.abs() > 1e-6).float()
            # signal branch: same-direction scaling
            policy_sig = baseline_actions * (1 + cfg.residual_scale * delta_sig)
            # no-signal branch: decay on last_position (included as last feature)
            last_pos = states[:, -2].unsqueeze(-1)  # last_position is penultimate feature; silence_days is last
            decay = torch.sigmoid(cfg.decay_scale * delta_dec)
            policy_nosig = last_pos * decay
            policy = has_signal * policy_sig + (1 - has_signal) * policy_nosig
            # BC: only enforce fidelity on has_signal branch
            loss = criterion(policy * has_signal, baseline_actions * has_signal)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            steps += 1
        LOGGER.info("BC epoch %d/%d loss=%.6f", epoch + 1, cfg.bc_epochs, epoch_loss / max(len(loader), 1))
        total += epoch_loss
    return total / max(steps, 1)


def iql_training(actor, critic, value_net, loader, cfg: TrainingConfig, device):
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)
    mse = nn.MSELoss()

    it = iter(torch.utils.data.DataLoader(loader.dataset, batch_size=cfg.iql_batch_size, shuffle=True, drop_last=True, pin_memory=True))
    for step in tqdm(range(1, cfg.iql_steps + 1), desc="IQL"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(torch.utils.data.DataLoader(loader.dataset, batch_size=cfg.iql_batch_size, shuffle=True, drop_last=True, pin_memory=True))
            batch = next(it)

        states = batch["state"].to(device)
        baseline_actions = batch["action"].to(device)
        rewards = batch["reward"].to(device)
        next_states = batch["next_state"].to(device)
        dones = batch["done"].to(device).float()

        out = actor(states)
        delta_sig = out["delta_signal"]
        delta_dec = out["delta_decay"]
        has_signal = (baseline_actions.abs() > 1e-6).float()
        last_pos = states[:, -2].unsqueeze(-1)  # last_position
        decay = torch.sigmoid(cfg.decay_scale * delta_dec)
        policy_sig = baseline_actions * (1 + cfg.residual_scale * delta_sig)
        policy_nosig = last_pos * decay
        policy_actions = has_signal * policy_sig + (1 - has_signal) * policy_nosig

        fidelity_penalty = (policy_actions.detach() - baseline_actions).pow(2).squeeze(-1) * has_signal.squeeze(-1)
        reward_aug = (rewards - cfg.fidelity_lambda * fidelity_penalty).unsqueeze(-1)

        with torch.no_grad():
            next_v = value_net(next_states)
            target_q = reward_aug + cfg.gamma * (1 - dones.unsqueeze(-1)) * next_v

        critic_pred = critic(states, baseline_actions)
        critic_loss = mse(critic_pred, target_q)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        with torch.no_grad():
            q_sa = critic(states, baseline_actions)
        v_pred = value_net(states)
        v_loss = expectile_loss(q_sa - v_pred, cfg.expectile)
        value_opt.zero_grad()
        v_loss.backward()
        value_opt.step()

        q_pi = critic(states, policy_actions)
        with torch.no_grad():
            adv = q_pi - v_pred
            weights = torch.clamp(torch.exp(cfg.temperature_beta * adv), max=100.0)
        # Actor: maximize advantage-weighted Q; fidelity only on signal branch
        reg = cfg.fidelity_lambda * (policy_actions - baseline_actions).pow(2) * has_signal
        actor_loss = -(weights * q_pi).mean() + reg.mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        if step % 1000 == 0:
            LOGGER.info("IQL %d/%d critic=%.6f value=%.6f actor=%.6f", step, cfg.iql_steps, critic_loss.item(), v_loss.item(), actor_loss.item())


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


def evaluate(actor: ActorNetwork, buffer_path: Path, cfg: TrainingConfig, device: torch.device):
    actor.eval()
    buf = load_buffer(buffer_path)
    states = buf["states"]
    rewards = buf["rewards"].numpy()
    baseline_actions = buf["actions"].numpy()
    dates = buf["meta"]["published_at"]
    tickers = buf["meta"]["ticker"]

    delta_sig_all = []
    delta_dec_all = []
    with torch.no_grad():
        for start in range(0, states.size(0), 1024):
            batch = states[start : start + 1024].to(device)
            out = actor(batch)
            delta_sig_all.append(out["delta_signal"].squeeze(-1).cpu())
            delta_dec_all.append(out["delta_decay"].squeeze(-1).cpu())
    delta_sig = torch.cat(delta_sig_all).numpy()
    delta_dec = torch.cat(delta_dec_all).numpy()

    has_signal = (baseline_actions.squeeze(-1) != 0).astype(float)
    last_pos = states[:, -2].numpy()
    decay = 1 / (1 + np.exp(-cfg.decay_scale * delta_dec))
    policy_sig = baseline_actions.squeeze(-1) * (1 + cfg.residual_scale * delta_sig)
    policy_nosig = last_pos * decay
    raw_scores = has_signal * policy_sig + (1 - has_signal) * policy_nosig

    df = torch.tensor(rewards)  # dummy to avoid unused
    import pandas as pd

    dfp = pd.DataFrame({"date": dates, "ticker": tickers, "reward": rewards, "raw_score": raw_scores})
    portfolio = PortfolioLayer(PortfolioConfig(max_long=cfg.max_weight, max_short=cfg.max_weight, hold_decay=cfg.hold_decay))
    daily_returns = []
    prev_weights: Dict[str, float] = {}
    for date, grp in dfp.groupby("date"):
        raw_dict = {r["ticker"]: r["raw_score"] for _, r in grp.iterrows()}
        alloc = portfolio.allocate(raw_dict, prev_weights=prev_weights)
        new_weights = {t: info["weight"] for t, info in alloc.items()}
        day_r = 0.0
        for _, r in grp.iterrows():
            t = r["ticker"]
            day_r += new_weights.get(t, 0.0) * r["reward"]
        daily_returns.append(day_r)
        prev_weights = new_weights

    metrics = compute_metrics(np.array(daily_returns)) if daily_returns else {"cumulative_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    return metrics


def main():
    cfg = parse_args()
    device = torch.device(cfg.device)
    train_path = Path(cfg.replay_dir) / cfg.kol / "train.pt"
    val_path = Path(cfg.replay_dir) / cfg.kol / "val.pt"

    if not train_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {train_path}")

    dataset = ReplayDataset(train_path)
    state_dim = dataset.states.shape[1]
    LOGGER.info("Loaded buffer %s with %d samples, state_dim=%d", cfg.kol, len(dataset), state_dim)

    bc_loader = DataLoader(dataset, batch_size=cfg.bc_batch_size, shuffle=True, drop_last=True, pin_memory=True)
    iql_loader = DataLoader(dataset, batch_size=cfg.iql_batch_size, shuffle=True, drop_last=True, pin_memory=True)

    actor = ActorNetwork(state_dim).to(device)
    critic = CriticNetwork(state_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)

    bc_loss = behavior_cloning(actor, bc_loader, cfg, device) if cfg.bc_epochs > 0 else 0.0
    LOGGER.info("BC done loss=%.6f", bc_loss)

    iql_training(actor, critic, value_net, iql_loader, cfg, device)

    metrics = {}
    if val_path.exists():
        metrics = evaluate(actor, val_path, cfg, device)
        LOGGER.info("Val metrics: %s", metrics)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.kol}_{timestamp}"
    run_dir = Path(cfg.output_dir) / run_name
    ckpt_dir = run_dir / Path(cfg.checkpoints_dir).name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": actor.state_dict()}, ckpt_dir / "policy.pt")
    torch.save(actor.state_dict(), ckpt_dir / "actor.pt")
    torch.save(critic.state_dict(), ckpt_dir / "critic.pt")
    torch.save(value_net.state_dict(), ckpt_dir / "value.pt")
    summary = {
        "run_name": run_name,
        "timestamp": timestamp,
        "kol": cfg.kol,
        "train_samples": len(dataset),
        "bc_loss": bc_loss,
        "metrics": metrics,
        "config": asdict(cfg),
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    LOGGER.info("Saved checkpoints to %s and summary to run_summary.json", ckpt_dir)


if __name__ == "__main__":
    main()
