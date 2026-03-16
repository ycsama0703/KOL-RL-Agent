"""Construct replay buffers from reward CSVs and embeddings (with behavior policy)."""

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

from src.pipeline.replay_utils import (
    annotate_positions,
    build_behavior_weights,
    build_states,
    compute_portfolio_rewards,
    load_ticker_embedder,
)
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
    parser.add_argument(
        "--behavior-alpha",
        type=float,
        default=0.3,
        help="Entry smoothing factor for behavior policy.",
    )
    parser.add_argument(
        "--behavior-decay",
        type=float,
        default=0.2,
        help="Exit decay factor for behavior policy.",
    )
    parser.add_argument(
        "--behavior-entry-threshold",
        type=float,
        default=1e-3,
        help="Baseline weight abs threshold for signal vs no-signal.",
    )
    parser.add_argument(
        "--next-state-mode",
        choices=["next_event", "next_date"],
        default="next_event",
        help="How to build next_state links: next mention event or by next_date column.",
    )
    return parser.parse_args()


def collect_reward_files(reward_dir: Path) -> Dict[str, List[Path]]:
    """Group reward CSVs by their relative parent path under reward_dir.

    Examples:
    - reward_dir/<kol>/<split>.csv              -> group "<kol>"
    - reward_dir/<source>/<kol>/<split>.csv     -> group "<source>/<kol>"
    """

    files: Dict[str, List[Path]] = {}
    for csv in sorted(reward_dir.rglob("*.csv")):
        rel_parent = csv.relative_to(reward_dir).parent
        group = str(rel_parent) if str(rel_parent) != "." else csv.parent.name
        files.setdefault(group, []).append(csv)
    return files


def compute_next_indices(df: pd.DataFrame) -> np.ndarray:
    next_idx = np.full(len(df), -1, dtype=np.int64)
    grouped = df.groupby("ticker", sort=False)
    for _, group in grouped:
        indices = group.sort_values("published_at").index.to_list()
        for current, nxt in zip(indices[:-1], indices[1:]):
            next_idx[current] = nxt
    return next_idx


def compute_next_indices_by_next_date(df: pd.DataFrame) -> np.ndarray:
    """Map each row to same-ticker row located on `next_date` (date-level match)."""
    next_idx = np.full(len(df), -1, dtype=np.int64)
    if "next_date" not in df.columns:
        return next_idx

    day_col = "trading_day" if "trading_day" in df.columns else "__event_day"
    if day_col == "__event_day":
        df = df.copy()
        df[day_col] = pd.to_datetime(df["published_at"], errors="coerce").dt.strftime("%Y-%m-%d")

    # first row index for each (ticker, day)
    first_index: Dict[tuple[str, str], int] = {}
    for i, row in df.reset_index().iterrows():
        ticker = str(row["ticker"])
        day = str(row[day_col]) if pd.notna(row[day_col]) else ""
        if not day:
            continue
        key = (ticker, day)
        if key not in first_index:
            first_index[key] = int(row["index"])

    for i, row in df.iterrows():
        ticker = str(row["ticker"])
        nd = row.get("next_date", None)
        if not isinstance(nd, str) or not nd:
            continue
        day = nd[:10]
        j = first_index.get((ticker, day), -1)
        if j >= 0:
            next_idx[i] = j
    return next_idx


def build_buffer(
    df: pd.DataFrame,
    ticker_embedder,
    *,
    next_state_mode: str = "next_event",
) -> Dict[str, torch.Tensor | List[str]]:
    df = df.sort_values(["ticker", "published_at"]).reset_index(drop=True)
    states = build_states(df, ticker_embedder)
    next_states = np.zeros_like(states)
    baseline = df["baseline_weight"].fillna(0.0).values.astype(np.float32)
    behavior = df["behavior_weight"].fillna(0.0).values.astype(np.float32)
    next_baseline = np.zeros_like(baseline)

    if next_state_mode == "next_date":
        next_indices = compute_next_indices_by_next_date(df)
    else:
        next_indices = compute_next_indices(df)
    dones = df["done"].astype(bool).values.copy()
    for idx, next_idx in enumerate(next_indices):
        if next_idx >= 0:
            next_states[idx] = states[next_idx]
            next_baseline[idx] = baseline[next_idx]
        else:
            dones[idx] = True

    event_ids = (
        df["event_id"].astype(str)
        if "event_id" in df.columns
        else df["video_id"].astype(str)
        if "video_id" in df.columns
        else pd.Series([""] * len(df))
    )

    buffer = {
        "states": torch.from_numpy(states),
        # 单票 reward_1d 仍然保留，用于评估/回放；组合级 reward 存在 portfolio_rewards 中供训练使用。
        "rewards": torch.from_numpy(df["reward_1d"].fillna(0.0).values.astype(np.float32)),
        "portfolio_rewards": torch.from_numpy(df["portfolio_reward"].fillna(0.0).values.astype(np.float32)),
        # 行为动作（用于训练 actor/critic）
        "actions": torch.from_numpy(behavior).unsqueeze(-1),
        # 基线动作（意图锚）
        "baseline_actions": torch.from_numpy(baseline).unsqueeze(-1),
        # 下一步的基线动作（用于 residual-aware value）
        "next_baseline_action": torch.from_numpy(next_baseline).unsqueeze(-1),
        "next_states": torch.from_numpy(next_states),
        "dones": torch.from_numpy(dones.astype(np.bool_)),
        "meta": {
            "ticker": df["ticker"].astype(str).tolist(),
            "event_id": event_ids.tolist(),
            "published_at": df["published_at"].astype(str).tolist(),
            "baseline_raw_score": df["baseline_raw_score"].astype(float).tolist(),
        },
    }
    return buffer


def process_file(
    csv_path: Path,
    output_path: Path,
    ticker_embedder: TickerEmbedding,
    *,
    behavior_alpha: float,
    behavior_decay: float,
    behavior_entry_threshold: float,
    next_state_mode: str,
) -> None:
    df = pd.read_csv(csv_path, parse_dates=["published_at"])
    required_cols = {"sentiment", "confidence", "reward_1d", "baseline_raw_score", "ticker"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] {csv_path} missing columns: {missing}; skipping")
        return

    df = annotate_positions(df)
    df = build_behavior_weights(
        df,
        alpha_entry=behavior_alpha,
        decay_exit=behavior_decay,
        entry_threshold=behavior_entry_threshold,
    )
    # 组合层 reward：多空权重 * 单票收益（无额外成本），基于行为权重
    df["portfolio_reward"] = compute_portfolio_rewards(df, weight_col="behavior_weight").astype(np.float32)
    buffer = build_buffer(df, ticker_embedder, next_state_mode=next_state_mode)
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
        split_seen = set()
        for csv_path in csv_files:
            split = csv_path.stem  # train/val/test
            if split in split_seen:
                raise ValueError(
                    f"Duplicate split '{split}' under group '{kol}'. "
                    f"Please ensure only one {split}.csv per group."
                )
            split_seen.add(split)
            out_path = output_dir / kol / f"{split}.pt"
            process_file(
                csv_path,
                out_path,
                ticker_embedder,
                behavior_alpha=args.behavior_alpha,
                behavior_decay=args.behavior_decay,
                behavior_entry_threshold=args.behavior_entry_threshold,
                next_state_mode=args.next_state_mode,
            )


if __name__ == "__main__":
    main()
