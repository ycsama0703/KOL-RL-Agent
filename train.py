"""
Train script for BC → IQL with explicit KOL-intent constraints.

Key conventions (from your ReplayDataset batch):
- batch["baseline_action"]: KOL intent anchor (shape [B,1])
- batch["action"]: behavior action used to learn critic/value (shape [B,1])
- batch["state"], batch["next_state"]: float32 tensors (shape [B,804])
- batch["reward"]: float32 (shape [B])
- batch["done"]: bool (shape [B])

Policy is residual:
    policy_action = baseline_action + delta
Then we enforce:
1) No reversal against baseline direction
2) No new entry when baseline_action ≈ 0
"""

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
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.portfolio.layer import PortfolioLayer
from src.training.data import ReplayDataset, load_buffer
from src.training.models import ActorNetwork, CriticNetwork, ValueNetwork
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


# -------------------------
# Config
# -------------------------
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
    bc_fit_behavior: bool = True
    bc_anchor_lambda: float = 0.1  # pulls policy toward baseline_action even when fitting behavior

    # IQL
    iql_steps: int = 200_000
    iql_batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    gamma: float = 0.99
    expectile: float = 0.7
    temperature_beta: float = 3.0

    # Faithfulness shaping (IQL)
    fidelity_lambda: float = 0.1
    # Soft alignment & soft intent penalties (actor update only)
    actor_align_lambda: float = 0.1
    entry_penalty_lambda: float = 0.1
    reversal_penalty_lambda: float = 0.1

    # Explicit intent constraints (the “my method” part)
    entry_threshold: float = 1e-3   # baseline_action abs below this => no entry allowed
    clamp_delta: float = 1.0        # delta is clamped to [-clamp_delta, +clamp_delta]

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainingConfig:
    p = argparse.ArgumentParser(description="Train KOL agent with BC + IQL (intent-constrained residual policy).")
    p.add_argument("--kol", default="Everything_Money")
    p.add_argument("--replay-dir", default="data/replay_buffer")
    p.add_argument("--checkpoints-dir", default="models/checkpoints")
    p.add_argument("--output-dir", default="outputs")

    p.add_argument("--bc-epochs", type=int, default=10)
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    p.add_argument("--bc-fit-behavior", action="store_true", default=True)
    p.add_argument("--bc-anchor-lambda", type=float, default=0.1)

    p.add_argument("--iql-steps", type=int, default=200_000)
    p.add_argument("--iql-batch-size", type=int, default=256)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--temperature-beta", type=float, default=3.0)
    p.add_argument("--fidelity-lambda", type=float, default=0.1)
    p.add_argument("--actor-align-lambda", type=float, default=0.1)
    p.add_argument("--entry-penalty-lambda", type=float, default=0.1)
    p.add_argument("--reversal-penalty-lambda", type=float, default=0.1)

    p.add_argument("--entry-threshold", type=float, default=1e-3)
    p.add_argument("--clamp-delta", type=float, default=1.0)

    args = p.parse_args()
    return TrainingConfig(
        kol=args.kol,
        replay_dir=args.replay_dir,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_lr=args.bc_lr,
        bc_fit_behavior=args.bc_fit_behavior,
        bc_anchor_lambda=args.bc_anchor_lambda,
        iql_steps=args.iql_steps,
        iql_batch_size=args.iql_batch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        gamma=args.gamma,
        expectile=args.expectile,
        temperature_beta=args.temperature_beta,
        fidelity_lambda=args.fidelity_lambda,
        actor_align_lambda=args.actor_align_lambda,
        entry_penalty_lambda=args.entry_penalty_lambda,
        reversal_penalty_lambda=args.reversal_penalty_lambda,
        entry_threshold=args.entry_threshold,
        clamp_delta=args.clamp_delta,
    )


# -------------------------
# Intent-constrained residual policy utilities
# -------------------------
def _extract_delta(actor_out, baseline_action: torch.Tensor) -> torch.Tensor:
    """
    Compatibility layer:
    - If ActorNetwork returns a Tensor: use it as delta directly (shape [B,1]).
    - If ActorNetwork returns a dict (e.g. {"delta_signal":..., "delta_decay":...}):
      choose delta_signal when baseline_action indicates "has signal",
      else choose delta_decay.
    """
    if isinstance(actor_out, torch.Tensor):
        return actor_out

    if isinstance(actor_out, dict):
        # Heuristic based on baseline magnitude: signal if baseline is not ~0
        has_signal = (baseline_action.abs() > 1e-6)
        delta_signal = actor_out.get("delta_signal", None)
        delta_decay = actor_out.get("delta_decay", None)
        if delta_signal is None or delta_decay is None:
            raise KeyError("ActorNetwork returned dict but missing 'delta_signal'/'delta_decay' keys.")
        return torch.where(has_signal, delta_signal, delta_decay)

    raise TypeError(f"Unsupported actor output type: {type(actor_out)}")


def apply_intent_constraints(
    baseline_action: torch.Tensor,
    delta: torch.Tensor,
    cfg: TrainingConfig,
) -> torch.Tensor:
    """
    Construct policy_action = baseline_action + delta, then enforce:
    1) No-entry when baseline_action ≈ 0
    2) No reversal against baseline direction (sign constraint)
    """
    # Clamp delta magnitude to keep residual interpretation stable
    delta = torch.clamp(delta, -cfg.clamp_delta, cfg.clamp_delta)

    # No entry zone: if baseline ≈ 0, force delta to 0 -> policy_action = 0
    no_entry = baseline_action.abs() < cfg.entry_threshold
    delta = torch.where(no_entry, torch.zeros_like(delta), delta)

    proposed = baseline_action + delta

    # Directional constraint: if baseline > 0 => policy >= 0; if baseline < 0 => policy <= 0
    sign = torch.sign(baseline_action + 1e-8)
    proposed = torch.where(sign > 0, torch.clamp(proposed, min=0.0), proposed)
    proposed = torch.where(sign < 0, torch.clamp(proposed, max=0.0), proposed)

    return proposed


def build_policy_action_for_training(
    baseline_action: torch.Tensor,
    delta: torch.Tensor,
    cfg: TrainingConfig,
) -> torch.Tensor:
    """
    Training-time action (SOFT):
    - residual form: a = baseline + clamp(delta)
    - NO hard no-entry / no-reversal here
    """
    delta = torch.clamp(delta, -cfg.clamp_delta, cfg.clamp_delta)
    return baseline_action + delta


def intent_penalties_soft(
    baseline_action: torch.Tensor,
    policy_action: torch.Tensor,
    cfg: TrainingConfig,
) -> Dict[str, torch.Tensor]:
    """
    Differentiable intent penalties (actor loss only).
    """
    # soft no-entry
    no_entry_mask = (baseline_action.abs() < cfg.entry_threshold).float()
    entry_pen = (no_entry_mask * policy_action.abs()).mean()

    # soft no-reversal (ignore baseline≈0)
    has_signal = (baseline_action.abs() >= cfg.entry_threshold).float()
    rev_pen = (has_signal * torch.relu(-(policy_action * baseline_action))).mean()

    return {"entry_pen": entry_pen, "rev_pen": rev_pen}

# -------------------------
# BC
# -------------------------
def behavior_cloning(
    actor: ActorNetwork,
    dataloader: DataLoader,
    cfg: TrainingConfig,
    device: torch.device,
) -> float:
    """
    BC stage:
    - If bc_fit_behavior=True: fit policy_action to behavior action, with an anchor penalty to baseline_action.
    - Else: fit policy_action to baseline_action (pure faithfulness warmstart).
    """
    mse = nn.MSELoss()
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.bc_lr)
    actor.train()

    total = 0.0
    steps = 0

    for ep in range(cfg.bc_epochs):
        ep_losses = []
        for batch in tqdm(dataloader, desc=f"BC {ep+1}/{cfg.bc_epochs}", leave=False):
            state = batch["state"].to(device)
            behavior_action = batch["action"].to(device)                 # [B,1]
            baseline_action = batch["baseline_action"].to(device)        # [B,1]

            actor_out = actor(state)
            delta = _extract_delta(actor_out, baseline_action)
            policy_action = apply_intent_constraints(baseline_action, delta, cfg)

            if cfg.bc_fit_behavior:
                loss_fit = mse(policy_action, behavior_action)
                loss_anchor = mse(policy_action, baseline_action)
                loss = loss_fit + cfg.bc_anchor_lambda * loss_anchor
            else:
                loss = mse(policy_action, baseline_action)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_losses.append(loss.item())
            total += loss.item()
            steps += 1

        LOGGER.info("BC epoch %d/%d - loss=%.6f", ep + 1, cfg.bc_epochs, float(np.mean(ep_losses)))

    return total / max(steps, 1)


# -------------------------
# IQL
# -------------------------
def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()


def iql_training(
    actor: ActorNetwork,
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

        behavior_action = batch["action"].to(device)              # [B,1]
        baseline_action = batch["baseline_action"].to(device)     # [B,1]
        next_baseline_action = batch["next_baseline_action"].to(device)  # [B,1]

        reward = batch["reward"].to(device)                       # [B]
        done = batch["done"].to(device).float()                   # [B]  (bool -> float)

        # Residual-aware value conditioning
        extended_state = torch.cat([state, baseline_action], dim=-1)
        extended_next_state = torch.cat([next_state, next_baseline_action], dim=-1)

        # Policy action = baseline + residual, with intent constraints
        actor_out = actor(state)
        delta = _extract_delta(actor_out, baseline_action)

        # Training-time policy action: SOFT (no hard gates)
        policy_action = build_policy_action_for_training(baseline_action, delta, cfg)

        delta_behavior = behavior_action - baseline_action
        delta_pi = policy_action - baseline_action

        # Faithfulness shaping: penalize deviation from baseline (detach to avoid shortcut)
        fidelity_penalty = (policy_action.detach() - baseline_action).pow(2).squeeze(-1)  # [B]
        reward_aug = reward - cfg.fidelity_lambda * fidelity_penalty

        with torch.no_grad():
            next_v = value_net(extended_next_state).squeeze(-1)  # [B]
            target_q = reward_aug + cfg.gamma * (1.0 - done) * next_v  # [B]

        # Critic regression on behavior data (stability)
        q_pred = critic(extended_state, delta_behavior).squeeze(-1)  # [B]
        critic_loss = mse(q_pred, target_q)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # Value via expectile regression toward Q(s, a_behavior)
        with torch.no_grad():
            q_sa = critic(extended_state, delta_behavior).squeeze(-1)  # [B]
        v_pred = value_net(extended_state).squeeze(-1)                  # [B]
        value_loss = expectile_loss(q_sa - v_pred, cfg.expectile)
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        # Actor update (AWAC-style) + soft alignment + soft intent penalties
        # -------------------------
        with torch.no_grad():
            # Advantage of BEHAVIOR action (in-distribution)
            q_b = critic(extended_state, delta_behavior).squeeze(-1)
            v = value_net(extended_state).squeeze(-1)
            adv_b = q_b - v
            weights = torch.clamp(torch.exp(cfg.temperature_beta * adv_b), max=100.0)

        # Weighted regression toward behavior action
        loss_fit = (weights * (policy_action - behavior_action).pow(2).squeeze(-1)).mean()

        # Soft alignment to baseline
        loss_align = (policy_action - baseline_action).pow(2).mean()

        # Soft intent penalties (replace hard gates during training)
        pens = intent_penalties_soft(baseline_action, policy_action, cfg)
        loss_entry = pens["entry_pen"]
        loss_rev = pens["rev_pen"]

        actor_loss = (
            loss_fit
            + cfg.actor_align_lambda * loss_align
            + cfg.entry_penalty_lambda * loss_entry
            + cfg.reversal_penalty_lambda * loss_rev
        )

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        if step % 1000 == 0:
            LOGGER.info(
                "IQL step %d/%d - critic=%.6f value=%.6f actor=%.6f",
                step,
                cfg.iql_steps,
                critic_loss.item(),
                value_loss.item(),
                actor_loss.item(),
            )


# -------------------------
# Evaluation (kept close to your original, but uses baseline_action if present)
# -------------------------
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


def evaluate(actor: ActorNetwork, buffer_path: Path, cfg: TrainingConfig, device: torch.device) -> Dict[str, float]:
    actor.eval()
    buffer = load_buffer(buffer_path)

    states = buffer["states"].float()
    rewards = buffer["rewards"].float().cpu().numpy()

    # Prefer baseline actions if available (robust to naming)
    if "baseline_actions" in buffer:
        baseline = buffer["baseline_actions"].float()
    elif "baseline_action" in buffer:
        baseline = buffer["baseline_action"].float()
    else:
        # Fallback: treat stored actions as baseline (backward compatibility)
        baseline = buffer["actions"].float()

    baseline_np = baseline.cpu().numpy().squeeze(-1)

    dates = buffer["meta"]["published_at"]
    tickers = buffer["meta"]["ticker"]

    deltas = []
    with torch.no_grad():
        for start in range(0, states.size(0), 1024):
            s = states[start : start + 1024].to(device)
            b = baseline[start : start + 1024].to(device)
            out = actor(s)
            d = _extract_delta(out, b)
            a = apply_intent_constraints(b, d, cfg)
            deltas.append((a.squeeze(-1).cpu().numpy() - b.squeeze(-1).cpu().numpy()))
    delta_np = np.concatenate(deltas, axis=0)

    raw_scores = baseline_np + delta_np

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


def save_checkpoints(actor: ActorNetwork, critic: CriticNetwork, value_net: ValueNetwork, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), checkpoint_dir / "actor.pt")
    torch.save(critic.state_dict(), checkpoint_dir / "critic.pt")
    torch.save(value_net.state_dict(), checkpoint_dir / "value.pt")
    torch.save({"actor_state_dict": actor.state_dict()}, checkpoint_dir / "policy.pt")
    LOGGER.info("Saved checkpoints to %s", checkpoint_dir)


# -------------------------
# Main
# -------------------------
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

    # file logger
    log_path = log_dir / "training.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(fh)

    LOGGER.info("Starting training run %s", run_name)
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Checkpoints will be saved under %s", checkpoint_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {train_path}")

    train_dataset = ReplayDataset(train_path)
    state_dim = int(train_dataset.states.shape[1])
    LOGGER.info("Loaded replay buffer for %s with %d samples, state_dim=%d", cfg.kol, len(train_dataset), state_dim)

    bc_loader = DataLoader(train_dataset, batch_size=cfg.bc_batch_size, shuffle=True, drop_last=True, pin_memory=True)
    iql_loader = DataLoader(train_dataset, batch_size=cfg.iql_batch_size, shuffle=True, drop_last=True, pin_memory=True)

    actor = ActorNetwork(state_dim).to(device)
    extended_state_dim = state_dim + 1
    critic = CriticNetwork(extended_state_dim).to(device)
    value_net = ValueNetwork(extended_state_dim).to(device)

    bc_loss = behavior_cloning(actor, bc_loader, cfg, device)
    LOGGER.info("Behavior cloning finished. Avg loss=%.6f", bc_loss)

    iql_training(actor, critic, value_net, iql_loader, cfg, device)

    metrics = {}
    if val_path.exists():
        metrics = evaluate(actor, val_path, cfg, device)
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
        "kol": cfg.kol,
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
