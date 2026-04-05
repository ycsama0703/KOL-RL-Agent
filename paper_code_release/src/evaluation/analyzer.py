"""Helper functions for evaluating trained agents and logging positions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from src.portfolio.layer import PortfolioLayer
from src.training.models import ActorNetwork
from train import TrainingConfig, apply_intent_constraints, compute_metrics


def load_actor(checkpoint_path: Path, state_dim: int, device: torch.device) -> ActorNetwork:
    """Load an actor network from a checkpoint or raw state dict."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("actor_state_dict", checkpoint)
    actor = ActorNetwork(state_dim).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def _extract_delta(actor_out: Any, baseline_action: torch.Tensor, cfg: TrainingConfig) -> torch.Tensor:
    if isinstance(actor_out, torch.Tensor):
        return actor_out
    if isinstance(actor_out, dict):
        if not cfg.regime_split:
            delta_signal = actor_out.get("delta_signal")
            if delta_signal is None:
                raise KeyError("ActorNetwork returned dict but missing 'delta_signal'.")
            return delta_signal
        has_signal = baseline_action.abs() > 1e-6
        delta_signal = actor_out.get("delta_signal")
        delta_decay = actor_out.get("delta_decay")
        if delta_signal is None or delta_decay is None:
            raise KeyError("ActorNetwork returned dict but missing 'delta_signal'/'delta_decay' keys.")
        return torch.where(has_signal, delta_signal, delta_decay)
    raise TypeError(f"Unsupported actor output type: {type(actor_out)}")


def _maybe_zero_market_factors(states: torch.Tensor, cfg: TrainingConfig) -> torch.Tensor:
    if not cfg.zero_market_factors:
        return states
    dim = int(max(cfg.market_factor_dim, 0))
    if dim <= 0:
        return states
    out = states.clone()
    tail = min(dim, int(out.shape[1]))
    out[..., -tail:] = 0.0
    return out


def _predict_policy_actions(
    actor: ActorNetwork,
    states: torch.Tensor,
    baseline_actions: torch.Tensor,
    device: torch.device,
    cfg: TrainingConfig,
    batch_size: int = 1024,
) -> np.ndarray:
    preds: list[torch.Tensor] = []
    states = _maybe_zero_market_factors(states, cfg)
    with torch.no_grad():
        for start in range(0, states.size(0), batch_size):
            batch = states[start : start + batch_size].to(device)
            baseline_batch = baseline_actions[start : start + batch_size].to(device)
            actor_out = actor(batch)
            delta = _extract_delta(actor_out, baseline_batch, cfg)
            policy_action = apply_intent_constraints(baseline_batch, delta, cfg)
            preds.append(policy_action.squeeze(-1).cpu())
    return torch.cat(preds).numpy()


def _classify_action(prev_weight: float, new_weight: float, delta: float, threshold: float) -> str:
    abs_prev = abs(prev_weight)
    abs_new = abs(new_weight)
    if abs_new < threshold and abs_prev < threshold:
        return "HOLD"
    if abs_new < threshold <= abs_prev:
        return "CLOSE"
    if abs_prev < threshold <= abs_new:
        return "OPEN"
    if delta > threshold:
        return "INCREASE"
    if delta < -threshold:
        return "DECREASE"
    return "HOLD"


def run_policy(
    actor: ActorNetwork,
    buffer: Dict[str, Any],
    device: torch.device,
    action_threshold: float = 0.01,
    cfg: TrainingConfig | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Replay a policy on a buffer and record metrics plus per-date positions.

    Assumptions:
    - Positions persist across dates; tickers with fresh signals overwrite the
      carried position, while untargeted holdings may continue through carry.
    - Long and short positions are both allowed; weights are normalized by
      absolute magnitude.
    - Daily reward is accumulated only for tickers with observed reward values;
      missing rewards are treated as zero on that date.
    """

    states = buffer["states"]
    rewards = buffer["rewards"].numpy()
    if "baseline_actions" in buffer:
        baseline_actions = buffer["baseline_actions"]  # Baseline anchor weights
    else:
        baseline_actions = buffer["actions"]
    dates = buffer["meta"]["published_at"]
    tickers = buffer["meta"]["ticker"]

    cfg = cfg or TrainingConfig()
    policy_actions = _predict_policy_actions(actor, states, baseline_actions, device, cfg)
    raw_scores = policy_actions  # Replay the constrained policy action as the raw score.
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
    position_rows: list[dict] = []
    prev_weights: Dict[str, float] = {}

    for date, group in df.groupby("date"):
        raw_dict = {row["ticker"]: row["raw_score"] for _, row in group.iterrows()}
        allocation = portfolio.allocate(raw_dict, prev_weights=prev_weights)
        new_weights = {ticker: info["weight"] for ticker, info in allocation.items()}

        # Map tickers with an observed reward on the current date.
        rewards_today: Dict[str, float] = {
            row["ticker"]: float(row["reward"]) for _, row in group.iterrows()
        }

        # Record all active holdings, including carried positions without a fresh signal.
        day_return = 0.0
        tickers_today = sorted(set(new_weights.keys()) | set(prev_weights.keys()))
        for ticker in tickers_today:
            prev_weight = float(prev_weights.get(ticker, 0.0))
            weight = float(new_weights.get(ticker, 0.0))
            delta = weight - prev_weight
            allocation = weight * portfolio.config.capital
            allocation_delta = delta * portfolio.config.capital
            reward = float(rewards_today.get(ticker, 0.0))
            raw_score = float(
                raw_dict.get(ticker, 0.0)
            )  # If a ticker has no fresh signal on this date, treat its raw score as zero.

            action = _classify_action(prev_weight, weight, delta, action_threshold)
            position_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "reward": reward,
                    "raw_score": raw_score,
                    "prev_weight": prev_weight,
                    "weight": weight,
                    "weight_delta": delta,
                    "allocation": allocation,
                    "allocation_delta": allocation_delta,
                    "action": action,
                }
            )
            day_return += weight * reward

        prev_weights = new_weights
        daily_returns.append(day_return)

    if not daily_returns:
        metrics = {"cumulative_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    else:
        metrics = compute_metrics(np.array(daily_returns))

    positions_df = pd.DataFrame(position_rows)
    return metrics, positions_df
