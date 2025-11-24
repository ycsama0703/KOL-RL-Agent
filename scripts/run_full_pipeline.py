"""Run end-to-end pipeline for all KOLs: cleaned -> embeddings -> enriched -> reward -> replay buffer.

Usage:
  python scripts/run_full_pipeline.py \
    --model answerdotai/modernbert-base \
    --cleaned data/processed/cleaned \
    --embeddings data/processed/embeddings \
    --enriched data/processed/enriched \
    --reward data/processed/reward \
    --replay data/replay_buffer \
    --vocab models/embedding/ticker_vocab.json \
    --ticker-emb models/embedding/ticker_embedding.pt \
    --price-days 5 \
    --batch-size 32
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full data pipeline for all KOLs.")
    p.add_argument("--model", default="answerdotai/modernbert-base", help="SentenceTransformer model or local path.")
    p.add_argument("--cleaned", default="data/processed/cleaned", help="Root dir of cleaned CSVs (<KOL>/<split>.csv).")
    p.add_argument("--embeddings", default="data/processed/embeddings", help="Output root for embeddings.")
    p.add_argument("--enriched", default="data/processed/enriched", help="Output root for enriched CSVs.")
    p.add_argument("--reward", default="data/processed/reward", help="Output root for reward CSVs.")
    p.add_argument("--replay", default="data/replay_buffer", help="Output root for replay buffers.")
    p.add_argument("--vocab", default="models/embedding/ticker_vocab.json", help="Ticker vocab path.")
    p.add_argument("--ticker-emb", default="models/embedding/ticker_embedding.pt", help="Ticker embedding weights.")
    p.add_argument("--price-days", type=int, default=5, help="Price history window.")
    p.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    p.add_argument("--device", default=None, help="Embedding device (e.g. cuda, cpu).")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"[PIPELINE] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()

    # 1) embeddings
    cmd = [
        sys.executable,
        "scripts/generate_embeddings.py",
        "--input",
        args.cleaned,
        "--output",
        args.embeddings,
        "--model",
        args.model,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.device:
        cmd += ["--device", args.device]
    run(cmd)

    # 2) enriched
    run(
        [
            sys.executable,
            "scripts/augment_with_market_data.py",
            "--input",
            args.cleaned,
            "--embeddings",
            args.embeddings,
            "--output",
            args.enriched,
            "--price-days",
            str(args.price_days),
        ]
    )

    # 3) reward
    run(
        [
            sys.executable,
            "scripts/generate_reward.py",
            "--input",
            args.enriched,
            "--output",
            args.reward,
        ]
    )

    # 4) baseline + ticker vocab/emb + replay buffer
    run(
        [
            sys.executable,
            "scripts/run_replay_pipeline.py",
            "--reward-dir",
            args.reward,
            "--vocab-path",
            args.vocab,
            "--embedding-path",
            args.ticker_emb,
            "--replay-dir",
            args.replay,
        ]
    )


if __name__ == "__main__":
    main()
