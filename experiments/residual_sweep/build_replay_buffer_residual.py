"""Build replay buffers with configurable single-stock cap and hold decay (experimental).

This script is self-contained to avoid modifying the main pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import sys

# add repo root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pipeline.replay_utils import build_states, compute_portfolio_rewards, load_ticker_embedder
from src.portfolio.layer import PortfolioConfig, PortfolioLayer
from src.state.ticker_embedding import TickerEmbedding


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build replay buffers (experimental residual sweep).")
    p.add_argument("--reward-dir", default="data/processed/reward")
    p.add_argument("--output-dir", default="data/replay_buffer_residual")
    p.add_argument("--ticker-embedding", default="models/embedding/ticker_embedding.pt")
    p.add_argument("--ticker-vocab", default="models/embedding/ticker_vocab.json")
    p.add_argument("--max-weight", type=float, default=0.2, help="Single-stock cap for baseline allocation.")
    p.add_argument("--hold-decay", type=float, default=1.0, help="Decay for tickers not mentioned today.")
    return p.parse_args()


def annotate_positions(df: pd.DataFrame, max_weight: float, hold_decay: float) -> pd.DataFrame:
    """Compute last_position, baseline_weight, silence_days; add carry rows for unmentioned tickers."""
    df = df.sort_values("published_at").reset_index(drop=True)
    portfolio = PortfolioLayer(PortfolioConfig(max_long=max_weight, max_short=max_weight, hold_decay=hold_decay))

    embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
    base_defaults = {col: 0 for col in df.columns}
    if "text" in base_defaults:
        base_defaults["text"] = ""
    if "video_id" in base_defaults:
        base_defaults["video_id"] = ""
    if "company" in base_defaults:
        base_defaults["company"] = ""

    prev_weights: Dict[str, float] = {}
    last_dates: Dict[str, pd.Timestamp] = {}
    rows: list[dict] = []

    for date, group in df.groupby("published_at", sort=True):
        raw_dict = {
            row["ticker"]: float(row["baseline_raw_score"]) * float(np.sign(row.get("sentiment", 0.0)))
            for _, row in group.iterrows()
        }
        weights = portfolio.allocate(raw_dict, prev_weights=prev_weights)

        # signal rows
        for _, row in group.iterrows():
            ticker = row["ticker"]
            cur_date = row["published_at"]
            prev_date = last_dates.get(ticker)
            silence = float((cur_date - prev_date).days) if prev_date is not None else 0.0
            last_dates[ticker] = cur_date
            enriched = row.to_dict()
            enriched["last_position"] = prev_weights.get(ticker, 0.0)
            enriched["baseline_weight"] = float(weights.get(ticker, {"weight": 0.0})["weight"])
            enriched["silence_days"] = silence
            enriched["has_signal"] = 1
            rows.append(enriched)

        # carry rows
        carry = [t for t in prev_weights.keys() if t not in raw_dict]
        for ticker in carry:
            cur_date = date
            prev_date = last_dates.get(ticker, cur_date)
            silence = float((cur_date - prev_date).days)
            last_dates[ticker] = cur_date
            enriched = base_defaults.copy()
            enriched["ticker"] = ticker
            enriched["published_at"] = cur_date
            enriched["sentiment"] = 0.0
            enriched["confidence"] = 0.0
            enriched["baseline_raw_score"] = 0.0
            enriched["reward_1d"] = 0.0
            enriched["done"] = False
            for col in embedding_cols:
                enriched[col] = 0.0
            enriched["last_position"] = prev_weights.get(ticker, 0.0)
            enriched["baseline_weight"] = 0.0
            enriched["silence_days"] = silence
            enriched["has_signal"] = 0
            rows.append(enriched)

        prev_weights = {t: v["weight"] for t, v in weights.items()}

    return pd.DataFrame(rows)


def collect_reward_files(reward_dir: Path) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {}
    for csv in reward_dir.rglob("*.csv"):
        kol = csv.parent.name
        files.setdefault(kol, []).append(csv)
    return files


def compute_next_indices(df: pd.DataFrame) -> np.ndarray:
    next_idx = np.full(len(df), -1, dtype=np.int64)
    grouped = df.groupby("ticker", sort=False)
    for _, group in grouped:
        indices = group.sort_values("published_at").index.to_list()
        for current, nxt in zip(indices[:-1], indices[1:]):
            next_idx[current] = nxt
    return next_idx


def build_buffer(df: pd.DataFrame, ticker_embedder: TickerEmbedding) -> Dict[str, torch.Tensor | List[str]]:
    df = df.sort_values(["ticker", "published_at"]).reset_index(drop=True)
    states = build_states(df, ticker_embedder)
    next_states = np.zeros_like(states)

    next_indices = compute_next_indices(df)
    dones = df["done"].astype(bool).values.copy()
    for idx, nxt in enumerate(next_indices):
        if nxt >= 0:
            next_states[idx] = states[nxt]
        else:
            dones[idx] = True

    buffer = {
        "states": torch.from_numpy(states),
        "rewards": torch.from_numpy(df["reward_1d"].fillna(0.0).values.astype(np.float32)),
        "portfolio_rewards": torch.from_numpy(df["portfolio_reward"].fillna(0.0).values.astype(np.float32)),
        # 行为动作：基线签名权重
        "actions": torch.from_numpy(df["baseline_weight"].fillna(0.0).values.astype(np.float32)).unsqueeze(-1),
        "next_states": torch.from_numpy(next_states),
        "dones": torch.from_numpy(dones.astype(np.bool_)),
        "meta": {
            "ticker": df["ticker"].astype(str).tolist(),
            "video_id": df["video_id"].astype(str).tolist(),
            "published_at": df["published_at"].astype(str).tolist(),
            "baseline_raw_score": df["baseline_raw_score"].astype(float).tolist(),
        },
    }
    return buffer


def process_file(csv_path: Path, output_path: Path, ticker_embedder: TickerEmbedding, max_weight: float, hold_decay: float) -> None:
    df = pd.read_csv(csv_path, parse_dates=["published_at"])
    required = {"sentiment", "confidence", "reward_1d", "baseline_raw_score", "ticker"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] {csv_path} missing columns: {missing}; skipping")
        return

    df = annotate_positions(df, max_weight=max_weight, hold_decay=hold_decay)
    df["portfolio_reward"] = compute_portfolio_rewards(df).astype(np.float32)

    buffer = build_buffer(df, ticker_embedder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(buffer, output_path)
    print(f"{csv_path.name}: saved replay buffer with {len(df)} samples -> {output_path}")


def main() -> None:
    args = parse_args()
    reward_dir = Path(args.reward_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ticker_embedder = load_ticker_embedder(Path(args.ticker_embedding), Path(args.ticker_vocab))
    files_by_kol = collect_reward_files(reward_dir)
    if not files_by_kol:
        print(f"No reward files found in {reward_dir}")
        return

    for kol, csv_files in files_by_kol.items():
        for csv_path in csv_files:
            split = csv_path.stem
            out_path = output_dir / kol / f"{split}.pt"
            process_file(csv_path, out_path, ticker_embedder, max_weight=args.max_weight, hold_decay=args.hold_decay)


if __name__ == "__main__":
    main()
