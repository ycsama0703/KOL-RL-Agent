"""Generate ModernBERT embeddings for cleaned KOL datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode text using ModernBERT.")
    parser.add_argument(
        "--model",
        default="answerdotai/modernbert-base",
        help="SentenceTransformer compatible model name.",
    )
    parser.add_argument(
        "--input",
        default="data/processed/cleaned",
        help="CSV file or directory containing cleaned datasets.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/embeddings",
        help="Directory to store embedding tensors mirroring input structure.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Force model device (e.g. cuda, cpu). Defaults to SentenceTransformer logic.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings via SentenceTransformer.",
    )
    return parser.parse_args()


def collect_csv_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    if path.suffix == ".csv":
        return [path]
    raise ValueError(f"Unsupported input path: {path}")


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    ).astype("float32")


def save_embeddings(
    embeddings: np.ndarray,
    output_root: Path,
    input_root: Path,
    source_csv: Path,
    model_name: str,
) -> Path:
    try:
        relative = source_csv.relative_to(input_root)
    except ValueError:
        relative = source_csv.name
    output_path = (output_root / relative).with_suffix(".pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(embeddings)
    payload = {
        "embeddings": tensor,
        "rows": embeddings.shape[0],
        "dim": embeddings.shape[1],
        "model": model_name,
        "source_csv": str(source_csv),
    }
    torch.save(payload, output_path)
    return output_path


def process_file(
    model: SentenceTransformer,
    model_name: str,
    csv_path: Path,
    input_root: Path,
    output_root: Path,
    batch_size: int,
    normalize: bool,
) -> None:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        print(f"[WARN] {csv_path} missing `text` column; skipping.")
        return
    texts = df["text"].fillna("").astype(str).tolist()
    if not texts:
        print(f"[WARN] {csv_path} contains no rows; skipping.")
        return
    embeddings = encode_texts(model, texts, batch_size, normalize)
    output_path = save_embeddings(embeddings, output_root, input_root, csv_path, model_name=model_name)
    print(f"Saved {embeddings.shape[0]} embeddings -> {output_path}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)

    model_kwargs = {}
    if args.device:
        model_kwargs["device"] = args.device
    model = SentenceTransformer(args.model, **model_kwargs)

    csv_files = collect_csv_files(input_root)
    for csv_path in csv_files:
        process_file(
            model=model,
            model_name=args.model,
            csv_path=csv_path,
            input_root=input_root,
            output_root=output_root,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )


if __name__ == "__main__":
    main()
