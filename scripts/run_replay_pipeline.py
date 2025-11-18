"""Run the full pipeline to create replay buffers in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reward → action → ticker → replay pipeline.")
    parser.add_argument("--reward-dir", default="data/processed/reward", help="Path to reward CSV directory.")
    parser.add_argument(
        "--vocab-path",
        default="models/embedding/ticker_vocab.json",
        help="Ticker vocabulary output path.",
    )
    parser.add_argument(
        "--embedding-path",
        default="models/embedding/ticker_embedding.pt",
        help="Ticker embedding weights output path.",
    )
    parser.add_argument(
        "--replay-dir",
        default="data/replay_buffer",
        help="Where to store serialized replay buffers.",
    )
    parser.add_argument("--embedding-dim", type=int, default=32, help="Ticker embedding dimension.")
    return parser.parse_args()


def run_step(cmd: list[str]) -> None:
    print(f"[PIPELINE] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    reward_dir = Path(args.reward_dir)
    reward_dir.mkdir(parents=True, exist_ok=True)

    run_step(
        [
            sys.executable,
            "scripts/add_baseline_action.py",
            "--input",
            str(reward_dir),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/build_ticker_embedding.py",
            "--input",
            str(reward_dir),
            "--vocab-path",
            str(args.vocab_path),
            "--embedding-path",
            str(args.embedding_path),
            "--embedding-dim",
            str(args.embedding_dim),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/build_replay_buffer.py",
            "--reward-dir",
            str(reward_dir),
            "--output-dir",
            str(args.replay_dir),
            "--ticker-embedding",
            str(args.embedding_path),
            "--ticker-vocab",
            str(args.vocab_path),
        ]
    )
    print("[PIPELINE] Replay buffer pipeline completed.")


if __name__ == "__main__":
    main()
