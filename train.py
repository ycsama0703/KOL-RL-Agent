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
import csv
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional

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
    kol: str = "Ale_s_World_of_Stocks"
    replay_dir: str = "data/buffer_22-24_end1231"
    checkpoints_dir: str = "models/checkpoints"
    output_dir: str = "outputs"

    # BC
    bc_epochs: int = 10
    bc_batch_size: int = 256
    bc_lr: float = 3e-4
    bc_fit_behavior: bool = True
    bc_anchor_lambda: float = 0.03  # lighter anchor under hard constraints

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
    fidelity_lambda: float = 0.03
    # Soft alignment & soft intent penalties (actor update only)
    actor_align_lambda: float = 0.04
    entry_penalty_lambda: float = 0.02
    reversal_penalty_lambda: float = 0.05

    # Explicit intent constraints (the “my method” part)
    hard_intent_constraints: bool = True
    entry_threshold: float = 5e-4   # baseline_action abs below this => no entry allowed
    clamp_delta: float = 1.8        # delta is clamped to [-clamp_delta, +clamp_delta]
    regime_split: bool = True       # if False, disable signal/silence routing and use one head
    zero_market_factors: bool = False
    market_factor_dim: int = 6      # tail dims in state reserved for market factors

    # Logging
    log_interval: int = 200
    write_iql_csv: bool = True
    progress_bar: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TrainingConfig:
    p = argparse.ArgumentParser(description="Train KOL agent with BC + IQL (intent-constrained residual policy).")
    p.add_argument("--kol", default="Ale_s_World_of_Stocks")
    p.add_argument("--replay-dir", default="data/buffer_22-24_end1231")
    p.add_argument("--checkpoints-dir", default="models/checkpoints")
    p.add_argument("--output-dir", default="outputs")

    p.add_argument("--bc-epochs", type=int, default=10)
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    p.add_argument("--bc-fit-behavior", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bc-anchor-lambda", type=float, default=0.03)

    p.add_argument("--iql-steps", type=int, default=200_000)
    p.add_argument("--iql-batch-size", type=int, default=256)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--temperature-beta", type=float, default=3.0)
    p.add_argument("--fidelity-lambda", type=float, default=0.03)
    p.add_argument("--actor-align-lambda", type=float, default=0.04)
    p.add_argument("--entry-penalty-lambda", type=float, default=0.02)
    p.add_argument("--reversal-penalty-lambda", type=float, default=0.05)

    p.add_argument("--entry-threshold", type=float, default=5e-4)
    p.add_argument("--clamp-delta", type=float, default=1.8)
    p.add_argument("--hard-intent-constraints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--regime-split", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--zero-market-factors", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--market-factor-dim", type=int, default=6)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--write-iql-csv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)

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
        hard_intent_constraints=args.hard_intent_constraints,
        entry_threshold=args.entry_threshold,
        clamp_delta=args.clamp_delta,
        regime_split=args.regime_split,
        zero_market_factors=args.zero_market_factors,
        market_factor_dim=args.market_factor_dim,
        log_interval=args.log_interval,
        write_iql_csv=args.write_iql_csv,
        progress_bar=args.progress_bar,
    )


# -------------------------
# Intent-constrained residual policy utilities
# -------------------------
def _extract_delta(actor_out, baseline_action: torch.Tensor, cfg: TrainingConfig) -> torch.Tensor:
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
        # Single-head ablation: bypass regime routing and always use one head.
        if not cfg.regime_split:
            delta_signal = actor_out.get("delta_signal", None)
            if delta_signal is None:
                raise KeyError("ActorNetwork returned dict but missing 'delta_signal' key.")
            return delta_signal
        # Heuristic based on baseline magnitude: signal if baseline is not ~0
        has_signal = (baseline_action.abs() > 1e-6)
        delta_signal = actor_out.get("delta_signal", None)
        delta_decay = actor_out.get("delta_decay", None)
        if delta_signal is None or delta_decay is None:
            raise KeyError("ActorNetwork returned dict but missing 'delta_signal'/'delta_decay' keys.")
        return torch.where(has_signal, delta_signal, delta_decay)

    raise TypeError(f"Unsupported actor output type: {type(actor_out)}")


def maybe_zero_market_factors(states: torch.Tensor, cfg: TrainingConfig) -> torch.Tensor:
    """Optionally zero out trailing market-factor dimensions for ablation."""
    if not cfg.zero_market_factors:
        return states
    dim = int(max(cfg.market_factor_dim, 0))
    if dim <= 0:
        return states
    tail = min(dim, int(states.shape[1]))
    states[..., -tail:] = 0.0
    return states


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

    # Vanilla path: keep residual form only, do not apply hard admissibility constraints.
    if not cfg.hard_intent_constraints:
        return baseline_action + delta

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
            delta = _extract_delta(actor_out, baseline_action, cfg)
            # Use the same hard intent constraints as evaluation to avoid train-test mismatch.
            policy_action = apply_intent_constraints(baseline_action, delta, cfg)
            pens = intent_penalties_soft(baseline_action, policy_action, cfg)

            if cfg.bc_fit_behavior:
                loss_fit = mse(policy_action, behavior_action)
                loss_anchor = mse(policy_action, baseline_action)
                loss = (
                    loss_fit
                    + cfg.bc_anchor_lambda * loss_anchor
                    + cfg.entry_penalty_lambda * pens["entry_pen"]
                    + cfg.reversal_penalty_lambda * pens["rev_pen"]
                )
            else:
                loss = (
                    mse(policy_action, baseline_action)
                    + cfg.entry_penalty_lambda * pens["entry_pen"]
                    + cfg.reversal_penalty_lambda * pens["rev_pen"]
                )

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
    iql_metrics_csv_path: Optional[Path] = None,
) -> None:
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    mse = nn.MSELoss()
    it = cycle(dataloader)

    actor.train()
    critic.train()
    value_net.train()

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
                "loss_fit",
                "loss_align",
                "loss_entry",
                "loss_rev",
                "reward_mean",
                "done_ratio",
                "adv_mean",
                "adv_std",
                "weight_mean",
                "weight_max",
                "policy_abs_mean",
                "baseline_abs_mean",
                "behavior_abs_mean",
            ]
        )
        csv_fp.flush()

    rolling = {
        "critic_loss": 0.0,
        "value_loss": 0.0,
        "actor_loss": 0.0,
        "loss_fit": 0.0,
        "loss_align": 0.0,
        "loss_entry": 0.0,
        "loss_rev": 0.0,
        "reward_mean": 0.0,
        "done_ratio": 0.0,
        "adv_mean": 0.0,
        "adv_std": 0.0,
        "weight_mean": 0.0,
        "weight_max": 0.0,
        "policy_abs_mean": 0.0,
        "baseline_abs_mean": 0.0,
        "behavior_abs_mean": 0.0,
    }
    rolling_n = 0

    steps_iter = range(1, cfg.iql_steps + 1)
    if cfg.progress_bar:
        steps_iter = tqdm(steps_iter, desc="IQL Training")

    for step in steps_iter:
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

        # Policy action = baseline + residual
        actor_out = actor(state)
        delta = _extract_delta(actor_out, baseline_action, cfg)

        # Training-time policy action: HARD (same constraints as evaluation)
        policy_action = apply_intent_constraints(baseline_action, delta, cfg)

        delta_behavior = behavior_action - baseline_action

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

        rolling["critic_loss"] += float(critic_loss.item())
        rolling["value_loss"] += float(value_loss.item())
        rolling["actor_loss"] += float(actor_loss.item())
        rolling["loss_fit"] += float(loss_fit.item())
        rolling["loss_align"] += float(loss_align.item())
        rolling["loss_entry"] += float(loss_entry.item())
        rolling["loss_rev"] += float(loss_rev.item())
        rolling["reward_mean"] += float(reward.mean().item())
        rolling["done_ratio"] += float(done.mean().item())
        rolling["adv_mean"] += float(adv_b.mean().item())
        rolling["adv_std"] += float(adv_b.std(unbiased=False).item())
        rolling["weight_mean"] += float(weights.mean().item())
        rolling["weight_max"] += float(weights.max().item())
        rolling["policy_abs_mean"] += float(policy_action.abs().mean().item())
        rolling["baseline_abs_mean"] += float(baseline_action.abs().mean().item())
        rolling["behavior_abs_mean"] += float(behavior_action.abs().mean().item())
        rolling_n += 1

        if step % max(cfg.log_interval, 1) == 0 or step == cfg.iql_steps:
            avg = {k: v / max(rolling_n, 1) for k, v in rolling.items()}
            LOGGER.info(
                "IQL step %d/%d | critic=%.6f value=%.6f actor=%.6f fit=%.6f align=%.6f entry=%.6f rev=%.6f reward=%.6f done=%.4f adv=%.6f(+/-%.6f) w=%.6f(max=%.4f) | |a|=%.6f |b|=%.6f |beh|=%.6f",
                step,
                cfg.iql_steps,
                avg["critic_loss"],
                avg["value_loss"],
                avg["actor_loss"],
                avg["loss_fit"],
                avg["loss_align"],
                avg["loss_entry"],
                avg["loss_rev"],
                avg["reward_mean"],
                avg["done_ratio"],
                avg["adv_mean"],
                avg["adv_std"],
                avg["weight_mean"],
                avg["weight_max"],
                avg["policy_abs_mean"],
                avg["baseline_abs_mean"],
                avg["behavior_abs_mean"],
            )
            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        step,
                        avg["critic_loss"],
                        avg["value_loss"],
                        avg["actor_loss"],
                        avg["loss_fit"],
                        avg["loss_align"],
                        avg["loss_entry"],
                        avg["loss_rev"],
                        avg["reward_mean"],
                        avg["done_ratio"],
                        avg["adv_mean"],
                        avg["adv_std"],
                        avg["weight_mean"],
                        avg["weight_max"],
                        avg["policy_abs_mean"],
                        avg["baseline_abs_mean"],
                        avg["behavior_abs_mean"],
                    ]
                )
                csv_fp.flush()

            for k in rolling:
                rolling[k] = 0.0
            rolling_n = 0

    if csv_fp is not None:
        csv_fp.close()


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
    states = maybe_zero_market_factors(states, cfg)
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
            d = _extract_delta(out, b, cfg)
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
    LOGGER.info("Training config: %s", json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True))
    if torch.cuda.is_available() and device.type == "cuda":
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(device))
    LOGGER.info("Device: %s", device)

    if not train_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {train_path}")

    train_dataset = ReplayDataset(train_path)
    state_dim = int(train_dataset.states.shape[1])
    LOGGER.info("Loaded replay buffer for %s with %d samples, state_dim=%d", cfg.kol, len(train_dataset), state_dim)
    baseline_abs = train_dataset.baseline_actions.abs()
    behavior_abs = train_dataset.actions.abs()
    baseline_zero_ratio = float((baseline_abs < cfg.entry_threshold).float().mean().item())
    LOGGER.info(
        "Dataset stats | baseline_abs_mean=%.6f behavior_abs_mean=%.6f baseline_zero_ratio(th=%.1e)=%.4f",
        float(baseline_abs.mean().item()),
        float(behavior_abs.mean().item()),
        cfg.entry_threshold,
        baseline_zero_ratio,
    )
    if cfg.zero_market_factors:
        # Apply once at dataset level to keep train/val/inference behavior consistent for this run.
        before_nonzero = float((train_dataset.states[:, -min(cfg.market_factor_dim, state_dim):].abs() > 0).float().mean().item()) if cfg.market_factor_dim > 0 else 0.0
        maybe_zero_market_factors(train_dataset.states, cfg)
        maybe_zero_market_factors(train_dataset.next_states, cfg)
        LOGGER.info(
            "Ablation: zero_market_factors enabled | market_factor_dim=%d | nonzero_ratio_before=%.4f",
            cfg.market_factor_dim,
            before_nonzero,
        )

    bc_loader = DataLoader(train_dataset, batch_size=cfg.bc_batch_size, shuffle=True, drop_last=True, pin_memory=True)
    iql_loader = DataLoader(train_dataset, batch_size=cfg.iql_batch_size, shuffle=True, drop_last=True, pin_memory=True)

    actor = ActorNetwork(state_dim).to(device)
    critic_state_dim = state_dim + 1
    critic = CriticNetwork(critic_state_dim).to(device)
    value_net = ValueNetwork(critic_state_dim).to(device)

    bc_loss = behavior_cloning(actor, bc_loader, cfg, device)
    LOGGER.info("Behavior cloning finished. Avg loss=%.6f", bc_loss)

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
