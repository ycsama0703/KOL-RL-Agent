"""Helper functions for evaluating trained agents and logging positions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from src.portfolio.layer import PortfolioLayer
from src.training.models import ActorNetwork
from train import compute_metrics


def load_actor(checkpoint_path: Path, state_dim: int, device: torch.device) -> ActorNetwork:
    """Load an actor network from a checkpoint or raw state dict."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("actor_state_dict", checkpoint)
    actor = ActorNetwork(state_dim).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def _predict_raw_scores(actor: ActorNetwork, states: torch.Tensor, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, states.size(0), batch_size):
            batch = states[start : start + batch_size].to(device)
            preds.append(actor(batch).squeeze(-1).cpu())
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
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Replay a policy on a buffer and record metrics plus per-date positions.

    逻辑假设：
    - 持仓连续：上一日持仓在下一日延续（可选轻微衰减），当日有新信号的 ticker 才被覆盖调整；
    - 允许多空：权重可正可负，按绝对值归一；未提到的 ticker 不视为负面，直接沿用昨仓；
    - 收益只在有 reward 记录的 ticker 上累计，其余当日视为 0 收益。
    """

    states = buffer["states"]
    rewards = buffer["rewards"].numpy()
    baseline_actions = buffer["actions"].numpy()  # 基线签名权重
    dates = buffer["meta"]["published_at"]
    tickers = buffer["meta"]["ticker"]

    delta = _predict_raw_scores(actor, states, device)
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
    position_rows: list[dict] = []
    prev_weights: Dict[str, float] = {}

    for date, group in df.groupby("date"):
        raw_dict = {row["ticker"]: row["raw_score"] for _, row in group.iterrows()}
        allocation = portfolio.allocate(raw_dict, prev_weights=prev_weights)
        new_weights = {ticker: info["weight"] for ticker, info in allocation.items()}

        # 当日有 reward 记录的 ticker→reward 映射
        rewards_today: Dict[str, float] = {
            row["ticker"]: float(row["reward"]) for _, row in group.iterrows()
        }

        # 记录所有当前持仓的变动（包括当日无新信号但延续持仓的 ticker）
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
            )  # 当日若无新信号，则 raw_score 视作 0

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
