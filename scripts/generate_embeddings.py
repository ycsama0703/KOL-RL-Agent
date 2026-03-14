"""Generate embeddings for cleaned KOL datasets."""

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
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading custom HF models (e.g., Qwen3 embedding).",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help='Optional attention implementation passed to model_kwargs (e.g. "flash_attention_2").',
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Optional torch dtype passed to model_kwargs.",
    )
    parser.add_argument(
        "--padding-side",
        default=None,
        choices=["left", "right"],
        help="Optional tokenizer padding side. Qwen3 recommends left padding.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional max sequence length for SentenceTransformer (model.max_seq_length).",
    )
    parser.add_argument(
        "--prompt-name",
        default=None,
        help='Optional SentenceTransformer prompt name (e.g. "query").',
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional raw prompt text prepended by SentenceTransformer encode.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=None,
        help="Optional output embedding dim via tail truncation (useful for MRL-style tradeoffs).",
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
    prompt_name: str | None,
    prompt: str | None,
    output_dim: int | None,
) -> np.ndarray:
    encode_kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "show_progress_bar": True,
        "normalize_embeddings": normalize,
    }
    if prompt_name:
        encode_kwargs["prompt_name"] = prompt_name
    if prompt:
        encode_kwargs["prompt"] = prompt

    embeddings = model.encode(
        texts,
        **encode_kwargs,
    ).astype("float32")

    if output_dim is not None:
        if output_dim <= 0:
            raise ValueError("--output-dim must be a positive integer.")
        if output_dim > embeddings.shape[1]:
            raise ValueError(f"--output-dim={output_dim} exceeds embedding dim={embeddings.shape[1]}.")
        embeddings = embeddings[:, :output_dim]
        # Re-normalize after truncation if user requests normalized embeddings.
        if normalize:
            denom = np.linalg.norm(embeddings, axis=1, keepdims=True)
            denom = np.clip(denom, 1e-12, None)
            embeddings = embeddings / denom
    return embeddings


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
    prompt_name: str | None,
    prompt: str | None,
    output_dim: int | None,
) -> None:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        print(f"[WARN] {csv_path} missing `text` column; skipping.")
        return
    texts = df["text"].fillna("").astype(str).tolist()
    if not texts:
        print(f"[WARN] {csv_path} contains no rows; skipping.")
        return
    embeddings = encode_texts(
        model=model,
        texts=texts,
        batch_size=batch_size,
        normalize=normalize,
        prompt_name=prompt_name,
        prompt=prompt,
        output_dim=output_dim,
    )
    output_path = save_embeddings(embeddings, output_root, input_root, csv_path, model_name=model_name)
    print(f"Saved {embeddings.shape[0]} embeddings -> {output_path}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)

    st_kwargs = {}
    if args.device:
        st_kwargs["device"] = args.device
    if args.trust_remote_code:
        st_kwargs["trust_remote_code"] = True

    model_kwargs = {}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)
    if model_kwargs:
        st_kwargs["model_kwargs"] = model_kwargs

    tokenizer_kwargs = {}
    if args.padding_side:
        tokenizer_kwargs["padding_side"] = args.padding_side
    if tokenizer_kwargs:
        st_kwargs["tokenizer_kwargs"] = tokenizer_kwargs

    model = SentenceTransformer(args.model, **st_kwargs)
    if args.max_length is not None:
        model.max_seq_length = args.max_length

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
            prompt_name=args.prompt_name,
            prompt=args.prompt,
            output_dim=args.output_dim,
        )


if __name__ == "__main__":
    main()
