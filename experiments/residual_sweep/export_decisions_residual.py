"""Export per-video decisions for residual policy (baseline * (1 + scale*tanh(delta)))."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# add repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio.layer import PortfolioConfig, PortfolioLayer
from src.pipeline.replay_utils import build_states, load_ticker_embedder
from src.training.models import ActorNetwork


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export residual decisions (baseline * (1 + scale*tanh(delta))).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reward-csv", required=True)
    p.add_argument("--vocab-path", required=True)
    p.add_argument("--embedding-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def classify(prev_w: float, w: float, th: float) -> str:
    delta = w - prev_w
    if abs(w) < th and abs(prev_w) < th:
        return "HOLD"
    if abs(w) < th <= abs(prev_w):
        return "CLOSE"
    if abs(prev_w) < th <= abs(w):
        return "OPEN"
    if delta > th:
        return "INCREASE"
    if delta < -th:
        return "DECREASE"
    return "HOLD"


def format_portfolio(weights: Dict[str, float], cutoff: float = 1e-6) -> str:
    return ";".join(f"{t}:{w:.4f}" for t, w in sorted(weights.items()) if abs(w) > cutoff)


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = Path(args.checkpoint)
    reward_csv = Path(args.reward_csv)
    if not ckpt.exists() or not reward_csv.exists():
        raise FileNotFoundError("Missing checkpoint or reward csv")

    df = pd.read_csv(reward_csv, parse_dates=["published_at"])
    required = {"ticker", "text", "sentiment", "confidence", "reward_1d", "baseline_raw_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Reward CSV missing columns: {missing}")

    # 重建 last_position / baseline_weight / silence_days 以满足 build_states
    portfolio = PortfolioLayer(PortfolioConfig(max_long=args.max_weight, max_short=args.max_weight, hold_decay=args.hold_decay))
    df = df.sort_values("published_at").reset_index(drop=True)
    prev_weights: Dict[str, float] = {}
    last_positions = []
    baseline_weights = []
    silence_days = []
    last_dates: Dict[str, object] = {}
    for date, group in df.groupby("published_at", sort=True):
        raw_base = {
            row["ticker"]: float(row["baseline_raw_score"]) * float(np.sign(row.get("sentiment", 0.0)))
            for _, row in group.iterrows()
        }
        weights = portfolio.allocate(raw_base, prev_weights=prev_weights)
        for _, row in group.iterrows():
            t = row["ticker"]
            last_positions.append(prev_weights.get(t, 0.0))
            baseline_weights.append(weights.get(t, {"weight": 0.0})["weight"])
            prev_date = last_dates.get(t)
            cur_date = row["published_at"]
            silence_days.append(float((cur_date - prev_date).days) if prev_date is not None else 0.0)
            last_dates[t] = cur_date
        prev_weights = {t: v["weight"] for t, v in weights.items()}
    df["last_position"] = last_positions
    df["baseline_weight"] = baseline_weights
    df["silence_days"] = silence_days

    ticker_embedder = load_ticker_embedder(Path(args.embedding_path), Path(args.vocab_path))
    states_np = build_states(df, ticker_embedder)
    states = torch.from_numpy(states_np)
    actor = load_actor(ckpt, states_np.shape[1], device)
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

    df = df.sort_values("published_at").reset_index(drop=True)
    df["baseline_signed"] = df["baseline_raw_score"] * np.sign(df["sentiment"].fillna(0.0))
    has_signal = (df["baseline_signed"].abs() > 1e-6).astype(float)
    last_pos = df["last_position"].fillna(0.0).astype(float).values
    decay = 1 / (1 + np.exp(-args.decay_scale * delta_dec))
    policy_sig = df["baseline_signed"].values * (1 + args.residual_scale * delta_sig)
    policy_nosig = last_pos * decay
    df["raw_trained"] = has_signal * policy_sig + (1 - has_signal) * policy_nosig

    portfolio = PortfolioLayer(PortfolioConfig(max_long=args.max_weight, max_short=args.max_weight, hold_decay=args.hold_decay))
    prev_base: Dict[str, float] = {}
    prev_train: Dict[str, float] = {}
    equity_base = 1.0
    equity_train = 1.0
    records: List[Dict[str, object]] = []

    for date, group in df.groupby("published_at", sort=True):
        raw_base = {r["ticker"]: r["baseline_signed"] for _, r in group.iterrows()}
        w_base_full = portfolio.allocate(raw_base, prev_weights=prev_base)
        w_base = {t: v["weight"] for t, v in w_base_full.items()}

        raw_train = {r["ticker"]: r["raw_trained"] for _, r in group.iterrows()}
        w_train_full = portfolio.allocate(raw_train, prev_weights=prev_train)
        w_train = {t: v["weight"] for t, v in w_train_full.items()}

        day_r_base = day_r_train = 0.0
        for _, r in group.iterrows():
            t = r["ticker"]
            rr = float(r["reward_1d"])
            day_r_base += w_base.get(t, 0.0) * rr
            day_r_train += w_train.get(t, 0.0) * rr
        equity_base *= 1 + day_r_base
        equity_train *= 1 + day_r_train

        video_groups = group.groupby("video_id", dropna=False) if "video_id" in group.columns else [(None, group)]
        for vid, vgroup in video_groups:
            video_tickers = []
            base_actions = []
            train_actions = []
            for _, r in vgroup.iterrows():
                t = r["ticker"]
                video_tickers.append(str(t))
                base_actions.append(f"{t}:{classify(prev_base.get(t,0.0), w_base.get(t,0.0), args.action_threshold)}")
                train_actions.append(f"{t}:{classify(prev_train.get(t,0.0), w_train.get(t,0.0), args.action_threshold)}")

            first = vgroup.iloc[0]
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "video_id": first.get("video_id", vid if vid is not None else ""),
                "text": first["text"],
                "tickers": ";".join(sorted(set(video_tickers))),
                "baseline_actions": ";".join(base_actions),
                "trained_actions": ";".join(train_actions),
                "portfolio_before_baseline": format_portfolio(prev_base),
                "portfolio_after_baseline": format_portfolio(w_base),
                "portfolio_before_trained": format_portfolio(prev_train),
                "portfolio_after_trained": format_portfolio(w_train),
                "equity_baseline": equity_base,
                "equity_trained": equity_train,
                "cum_return_baseline": equity_base - 1.0,
                "cum_return_trained": equity_train - 1.0,
            })

        prev_base = w_base
        prev_train = w_train

    out_df = pd.DataFrame(records)
    out_df.sort_values(["date", "video_id"], inplace=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
