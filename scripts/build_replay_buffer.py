"""Construct replay buffers from reward CSVs and embeddings (with silence_days feature)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pipeline.replay_utils import annotate_positions, build_states, compute_portfolio_rewards, load_ticker_embedder
from src.state.ticker_embedding import TickerEmbedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build replay buffers for Offline RL.")
    parser.add_argument(
        "--reward-dir",
        default="data/processed/reward",
        help="Directory containing reward CSVs (split by KOL).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/replay_buffer",
        help="Directory to store serialized replay buffers.",
    )
    parser.add_argument(
        "--ticker-embedding",
        default="models/embedding/ticker_embedding.pt",
        help="Path to ticker embedding weights (.pt).",
    )
    parser.add_argument(
        "--ticker-vocab",
        default="models/embedding/ticker_vocab.json",
        help="Path to ticker vocab json.",
    )
    return parser.parse_args()


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


def build_buffer(
    df: pd.DataFrame,
    ticker_embedder,
) -> Dict[str, torch.Tensor | List[str]]:
    df = df.sort_values(["ticker", "published_at"]).reset_index(drop=True)
    states = build_states(df, ticker_embedder)
    next_states = np.zeros_like(states)

    next_indices = compute_next_indices(df)
    dones = df["done"].astype(bool).values.copy()
    for idx, next_idx in enumerate(next_indices):
        if next_idx >= 0:
            next_states[idx] = states[next_idx]
        else:
            dones[idx] = True

    buffer = {
        "states": torch.from_numpy(states),
        # 单票 reward_1d 仍然保留，用于评估/回放；组合级 reward 存在 portfolio_rewards 中供训练使用。
        "rewards": torch.from_numpy(df["reward_1d"].fillna(0.0).values.astype(np.float32)),
        "portfolio_rewards": torch.from_numpy(df["portfolio_reward"].fillna(0.0).values.astype(np.float32)),
        # 动作（基线签名权重，行为策略）
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


def process_file(csv_path: Path, output_path: Path, ticker_embedder: TickerEmbedding) -> None:
    df = pd.read_csv(csv_path, parse_dates=["published_at"])
    required_cols = {"sentiment", "confidence", "reward_1d", "baseline_raw_score", "ticker"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] {csv_path} missing columns: {missing}; skipping")
        return

    df = annotate_positions(df)
    # 组合层 reward：多空权重 * 单票收益（无额外成本）
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
            split = csv_path.stem  # train/val/test
            out_path = output_dir / kol / f"{split}.pt"
            process_file(csv_path, out_path, ticker_embedder)


if __name__ == "__main__":
    main()
