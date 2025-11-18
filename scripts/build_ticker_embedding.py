"""Build ticker vocabulary and initialize ticker embedding weights."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.state.ticker_embedding import TickerEmbedding, TickerVocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ticker vocab and embedding matrix.")
    parser.add_argument(
        "--input",
        default="data/processed/reward",
        help="Directory containing reward CSVs (or a single CSV).",
    )
    parser.add_argument(
        "--vocab-path",
        default="models/embedding/ticker_vocab.json",
        help="Output path for ticker vocabulary JSON.",
    )
    parser.add_argument(
        "--embedding-path",
        default="models/embedding/ticker_embedding.pt",
        help="Output path for embedding weights (.pt).",
    )
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension.")
    return parser.parse_args()


def collect_csv_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    if path.suffix == ".csv":
        return [path]
    raise ValueError(f"Unsupported input path: {path}")


def collect_tickers(files: Iterable[Path]) -> Set[str]:
    tickers: Set[str] = set()
    for csv_path in files:
        df = pd.read_csv(csv_path, usecols=["ticker"])
        tickers.update(df["ticker"].dropna().astype(str).tolist())
    return tickers


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    vocab_path = Path(args.vocab_path)
    embedding_path = Path(args.embedding_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = collect_csv_files(input_path)
    if not csv_files:
        print(f"No CSV files found under {input_path}")
        return

    tickers = collect_tickers(csv_files)
    vocab = TickerVocabulary(tokens=tickers)
    vocab.save(vocab_path)
    embedder = TickerEmbedding(vocab, embedding_dim=args.embedding_dim)
    embedder.save(embedding_path, vocab_path)

    print(
        f"Saved ticker vocab ({len(vocab)} tokens) to {vocab_path} "
        f"and embedding weights (dim={args.embedding_dim}) to {embedding_path}"
    )


if __name__ == "__main__":
    main()
