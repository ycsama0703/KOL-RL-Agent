"""Export per-video decision trace for baseline vs trained policy.

每一行对应 KOL 的一次发文（一个视频），包括：
- 日期、video_id、原文 text
- 该视频涉及到的所有 ticker 及其在训练前/后的动作标签（加仓/减仓/平仓等）
- 两个策略在该视频对应交易日调仓前后的组合构成（以 ticker:weight 串表示）
- 两个策略从测试期开始到当前日期为止的净值和累计收益率

用法示例（Everything Money + test 集）：

  python scripts/export_signal_decisions.py \
    --checkpoint outputs/Everything_Money_20251119_171039/checkpoints/policy.pt \
    --reward-csv data/processed/reward/Everything_Money/test.csv \
    --vocab-path models/embedding/ticker_vocab.json \
    --embedding-path models/embedding/ticker_embedding.pt \
    --output outputs/Everything_Money_20251119_171039/signal_decisions_test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio.layer import PortfolioLayer
from src.pipeline.replay_utils import annotate_positions, build_states, load_ticker_embedder
from src.training.models import ActorNetwork
from src.evaluation.analyzer import run_policy
from src.training.data import load_buffer
from train import TrainingConfig, apply_intent_constraints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-signal decisions for baseline vs trained policy.")
    parser.add_argument("--checkpoint", required=True, help="Trained policy checkpoint (policy.pt or actor.pt).")
    parser.add_argument("--reward-csv", required=True, help="Reward CSV for the split to analyse (e.g., test.csv).")
    parser.add_argument("--buffer", help="Optional replay buffer to align equity with evaluation.")
    parser.add_argument("--vocab-path", required=True, help="Path to ticker_vocab.json.")
    parser.add_argument("--embedding-path", required=True, help="Path to ticker_embedding.pt.")
    parser.add_argument("--output", required=True, help="Output CSV path for the decision trace.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device.")
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=0.01,
        help="最小权重/权重变化阈值；低于该值视为无仓位或小幅变动。",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=1e-3,
        help="Baseline weight abs threshold for intent constraints.",
    )
    parser.add_argument(
        "--clamp-delta",
        type=float,
        default=1.0,
        help="Clamp range for residual delta in intent constraints.",
    )
    return parser.parse_args()



def load_actor(checkpoint_path: Path, state_dim: int, device: torch.device) -> ActorNetwork:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("actor_state_dict", ckpt)
    actor = ActorNetwork(state_dim).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def _extract_delta(actor_out: Any, baseline_action: torch.Tensor) -> torch.Tensor:
    if isinstance(actor_out, torch.Tensor):
        return actor_out
    if isinstance(actor_out, dict):
        has_signal = baseline_action.abs() > 1e-6
        delta_signal = actor_out.get("delta_signal")
        delta_decay = actor_out.get("delta_decay")
        if delta_signal is None or delta_decay is None:
            raise KeyError("ActorNetwork returned dict but missing 'delta_signal'/'delta_decay' keys.")
        return torch.where(has_signal, delta_signal, delta_decay)
    raise TypeError(f"Unsupported actor output type: {type(actor_out)}")


def predict_raw_scores(
    actor: ActorNetwork,
    states: torch.Tensor,
    baseline_actions: torch.Tensor,
    device: torch.device,
    cfg: TrainingConfig,
    batch_size: int = 1024,
) -> np.ndarray:
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, states.size(0), batch_size):
            batch = states[start : start + batch_size].to(device)
            baseline_batch = baseline_actions[start : start + batch_size].to(device)
            actor_out = actor(batch)
            delta = _extract_delta(actor_out, baseline_batch)
            policy_action = apply_intent_constraints(baseline_batch, delta, cfg)
            preds.append(policy_action.squeeze(-1).cpu())
    return torch.cat(preds).numpy()


def classify_action(prev_weight: float, new_weight: float, threshold: float) -> str:
    abs_prev = abs(prev_weight)
    abs_new = abs(new_weight)
    delta = new_weight - prev_weight
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


def format_portfolio(weights: Dict[str, float], cutoff: float = 1e-6) -> str:
    if not weights:
        return ""
    items = [
        f"{ticker}:{weight:.4f}"
        for ticker, weight in sorted(weights.items())
        if abs(weight) > cutoff
    ]
    return ";".join(items)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    reward_csv = Path(args.reward_csv)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not reward_csv.exists():
        raise FileNotFoundError(f"Reward CSV not found: {reward_csv}")

    df = pd.read_csv(reward_csv, parse_dates=["published_at"])
    required_cols = {"ticker", "text", "sentiment", "confidence", "reward_1d", "baseline_raw_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Reward CSV missing columns: {missing}")

    ticker_embedder = load_ticker_embedder(Path(args.embedding_path), Path(args.vocab_path))
    # Skip carry rows for faster export; we only need signal rows for per-video analysis.
    df = annotate_positions(df, include_carry=False).sort_values("published_at").reset_index(drop=True)

    states_np = build_states(df, ticker_embedder)
    states = torch.from_numpy(states_np)
    baseline_actions = torch.from_numpy(
        df["baseline_weight"].fillna(0.0).values.astype(np.float32)
    ).unsqueeze(-1)
    state_dim = states_np.shape[1]
    actor = load_actor(checkpoint_path, state_dim, device)
    cfg = TrainingConfig()
    cfg.entry_threshold = args.entry_threshold
    cfg.clamp_delta = args.clamp_delta
    policy_actions = predict_raw_scores(actor, states, baseline_actions, device, cfg)

    # 基线签名权重（含情感符号），训练输出为残差 delta
    df["baseline_signed"] = df["baseline_weight"]
    df["raw_trained"] = policy_actions

    def normalize_ts(value: object) -> pd.Timestamp:
        ts = pd.to_datetime(value, utc=True)
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(None)
        return ts

    equity_by_date = None
    if args.buffer:
        buffer_path = Path(args.buffer)
        if not buffer_path.exists():
            raise FileNotFoundError(f"Replay buffer not found: {buffer_path}")
        buffer = load_buffer(buffer_path)

        class ZeroActor(torch.nn.Module):
            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return torch.zeros((state.size(0), 1), device=state.device)

        _, base_positions = run_policy(ZeroActor().to(device), buffer, device)
        _, train_positions = run_policy(actor, buffer, device)

        def equity_series(positions: pd.DataFrame) -> pd.Series:
            df_pos = positions.copy()
            df_pos["date"] = df_pos["date"].apply(normalize_ts)
            df_pos["weighted_return"] = df_pos["weight"] * df_pos["reward"]
            daily = df_pos.groupby("date", as_index=True)["weighted_return"].sum().sort_index()
            equity = (1.0 + daily).cumprod()
            return equity

        eq_base = equity_series(base_positions)
        eq_train = equity_series(train_positions)
        equity_by_date = {
            "baseline": eq_base.to_dict(),
            "trained": eq_train.to_dict(),
        }

    portfolio = PortfolioLayer()
    prev_weights_baseline: Dict[str, float] = {}
    prev_weights_trained: Dict[str, float] = {}
    equity_baseline = 1.0
    equity_trained = 1.0

    records: List[Dict[str, object]] = []
    missing_equity = 0

    for date, group in df.groupby("published_at", sort=True):
        group_signals = group
        if "has_signal" in group.columns:
            group_signals = group[group["has_signal"] == 1]

        # baseline 组合：情感为正→多头，负→空头（仅信号行）
        raw_base = {
            row["ticker"]: float(row["baseline_raw_score"]) * float(np.sign(row.get("sentiment", 0.0)))
            for _, row in group_signals.iterrows()
        }
        weights_base_full = portfolio.allocate(raw_base, prev_weights=prev_weights_baseline)
        weights_base = {t: info["weight"] for t, info in weights_base_full.items()}

        raw_train = {row["ticker"]: row["raw_trained"] for _, row in group_signals.iterrows()}
        weights_train_full = portfolio.allocate(raw_train, prev_weights=prev_weights_trained)
        weights_train = {t: info["weight"] for t, info in weights_train_full.items()}

        if equity_by_date is None:
            # 当日收益 & 净值更新（fallback：仅用当前信号行）
            day_return_base = 0.0
            day_return_train = 0.0
            for _, row in group.iterrows():
                ticker = row["ticker"]
                r = float(row["reward_1d"])
                day_return_base += weights_base.get(ticker, 0.0) * r
                day_return_train += weights_train.get(ticker, 0.0) * r
            equity_baseline *= 1.0 + day_return_base
            equity_trained *= 1.0 + day_return_train
        else:
            key = normalize_ts(date)
            base_val = equity_by_date["baseline"].get(key)
            train_val = equity_by_date["trained"].get(key)
            if base_val is None or train_val is None:
                missing_equity += 1
                base_val = equity_baseline
                train_val = equity_trained
            equity_baseline = float(base_val)
            equity_trained = float(train_val)

        portfolio_before_base = format_portfolio(prev_weights_baseline)
        portfolio_after_base = format_portfolio(weights_base)
        portfolio_before_train = format_portfolio(prev_weights_trained)
        portfolio_after_train = format_portfolio(weights_train)

        # 以视频为粒度聚合：同一日期下按 video_id 分组
        if "video_id" in group.columns:
            video_groups = group.groupby("video_id", dropna=False)
        else:
            # 如果没有 video_id，就把当日所有样本视为一个“视频”
            video_groups = [(None, group)]

        for video_id, vgroup in video_groups:
            # 一个视频可能涉及多个 ticker，这里汇总成 ticker:ACTION 列表
            video_tickers: List[str] = []
            baseline_actions: List[str] = []
            trained_actions: List[str] = []

            for _, row in vgroup.iterrows():
                ticker = row["ticker"]
                video_tickers.append(str(ticker))
                base_prev_w = float(prev_weights_baseline.get(ticker, 0.0))
                base_w = float(weights_base.get(ticker, 0.0))
                train_prev_w = float(prev_weights_trained.get(ticker, 0.0))
                train_w = float(weights_train.get(ticker, 0.0))
                base_act = classify_action(base_prev_w, base_w, args.action_threshold)
                train_act = classify_action(train_prev_w, train_w, args.action_threshold)
                baseline_actions.append(f"{ticker}:{base_act}")
                trained_actions.append(f"{ticker}:{train_act}")

            first = vgroup.iloc[0]
            text = first["text"]
            video_id_val = first.get("video_id", video_id if video_id is not None else "")

            record: Dict[str, object] = {
                "date": pd.to_datetime(date).isoformat(),
                "video_id": video_id_val,
                "text": text,
                "tickers": ";".join(sorted(set(video_tickers))),
                "baseline_actions": ";".join(baseline_actions),
                "trained_actions": ";".join(trained_actions),
                "portfolio_before_baseline": portfolio_before_base,
                "portfolio_after_baseline": portfolio_after_base,
                "portfolio_before_trained": portfolio_before_train,
                "portfolio_after_trained": portfolio_after_train,
                "equity_baseline": equity_baseline,
                "equity_trained": equity_trained,
                "cum_return_baseline": equity_baseline - 1.0,
                "cum_return_trained": equity_trained - 1.0,
            }
            records.append(record)

        prev_weights_baseline = weights_base
        prev_weights_trained = weights_train

    if missing_equity:
        print(f"[WARN] Missing equity lookups for {missing_equity} dates; check timestamp alignment.")

    out_df = pd.DataFrame(records)
    if "video_id" in out_df.columns:
        out_df.sort_values(["date", "video_id"], inplace=True)
    else:
        out_df.sort_values(["date"], inplace=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved signal-level decision trace to {output_path}")


if __name__ == "__main__":
    main()
