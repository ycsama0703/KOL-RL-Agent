"""Evaluate residual policy with custom portfolio settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# add repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio.layer import PortfolioConfig, PortfolioLayer
from src.training.models import ActorNetwork
from src.training.data import load_buffer
from experiments.residual_sweep.train_residual import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval residual policy (baseline * (1 + scale*tanh(delta))).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--buffer", required=True, help="Replay buffer split (e.g., test.pt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", help="metrics json")
    p.add_argument("--positions-output", help="per-date positions csv")
    p.add_argument("--residual-scale", type=float, default=0.2)
    p.add_argument("--decay-scale", type=float, default=0.5)
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--hold-decay", type=float, default=1.0)
    p.add_argument("--action-threshold", type=float, default=0.01)
    return p.parse_args()


def load_actor(checkpoint_path: Path, state_dim: int, device: torch.device) -> ActorNetwork:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("actor_state_dict", ckpt)
    actor = ActorNetwork(state_dim).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    buf_path = Path(args.buffer)
    if not ckpt.exists() or not buf_path.exists():
        raise FileNotFoundError("Missing checkpoint or buffer")

    device = torch.device(args.device)
    buffer = load_buffer(buf_path)
    states = buffer["states"]
    rewards = buffer["rewards"].numpy()
    baseline_actions = buffer["actions"].numpy()
    dates = buffer["meta"]["published_at"]
    tickers = buffer["meta"]["ticker"]

    actor = load_actor(ckpt, states.shape[1], device)
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
    last_pos = states[:, -2].numpy()  # last_position
    decay = 1 / (1 + np.exp(-args.decay_scale * delta_dec))  # sigmoid
    policy_sig = baseline_actions.squeeze(-1) * (1 + args.residual_scale * delta_sig)
    policy_nosig = last_pos * decay
    raw_scores = has_signal * policy_sig + (1 - has_signal) * policy_nosig

    df = pd.DataFrame({"date": dates, "ticker": tickers, "reward": rewards, "raw_score": raw_scores})
    portfolio = PortfolioLayer(PortfolioConfig(max_long=args.max_weight, max_short=args.max_weight, hold_decay=args.hold_decay))
    prev_weights = {}
    daily_returns = []
    rows = []
    for date, grp in df.groupby("date"):
        raw_dict = {r["ticker"]: r["raw_score"] for _, r in grp.iterrows()}
        alloc = portfolio.allocate(raw_dict, prev_weights=prev_weights)
        new_weights = {t: info["weight"] for t, info in alloc.items()}
        rewards_today = {r["ticker"]: float(r["reward"]) for _, r in grp.iterrows()}
        day_r = 0.0
        tickers_today = sorted(set(new_weights) | set(prev_weights))
        for t in tickers_today:
            pw = float(prev_weights.get(t, 0.0))
            w = float(new_weights.get(t, 0.0))
            delta_w = w - pw
            reward = rewards_today.get(t, 0.0)
            row = {
                "date": date,
                "ticker": t,
                "prev_weight": pw,
                "weight": w,
                "weight_delta": delta_w,
                "allocation": w * portfolio.config.capital,
                "allocation_delta": delta_w * portfolio.config.capital,
                "reward": reward,
            }
            rows.append(row)
            day_r += w * reward
        prev_weights = new_weights
        daily_returns.append(day_r)

    metrics = compute_metrics(np.array(daily_returns)) if daily_returns else {"cumulative_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
    if args.positions_output:
        Path(args.positions_output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.positions_output, index=False)


if __name__ == "__main__":
    main()
