"""Generate embeddings for cleaned KOL datasets."""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger("generate_embeddings")


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
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Default: <output>/generate_embeddings.log",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first file-level error.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable sentence-transformers progress bar (recommended for detached logs).",
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
    show_progress_bar: bool,
) -> np.ndarray:
    encode_kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "show_progress_bar": show_progress_bar,
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
    show_progress_bar: bool,
    index: int,
    total: int,
) -> bool:
    started = time.time()
    LOGGER.info("[%d/%d] Start file: %s", index, total, csv_path)
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        LOGGER.warning("[%d/%d] Missing `text` column, skipped: %s", index, total, csv_path)
        return False
    texts = df["text"].fillna("").astype(str).tolist()
    if not texts:
        LOGGER.warning("[%d/%d] Empty rows, skipped: %s", index, total, csv_path)
        return False

    LOGGER.info(
        "[%d/%d] Encoding rows=%d batch=%d",
        index,
        total,
        len(texts),
        batch_size,
    )
    embeddings = encode_texts(
        model=model,
        texts=texts,
        batch_size=batch_size,
        normalize=normalize,
        prompt_name=prompt_name,
        prompt=prompt,
        output_dim=output_dim,
        show_progress_bar=show_progress_bar,
    )
    output_path = save_embeddings(embeddings, output_root, input_root, csv_path, model_name=model_name)
    elapsed = time.time() - started
    LOGGER.info(
        "[%d/%d] Done rows=%d dim=%d elapsed=%.1fs -> %s",
        index,
        total,
        embeddings.shape[0],
        embeddings.shape[1],
        elapsed,
        output_path,
    )
    return True


def configure_logging(output_root: Path, log_file: str | None, level: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file) if log_file else (output_root / "generate_embeddings.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return log_path


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    log_path = configure_logging(output_root=output_root, log_file=args.log_file, level=args.log_level)
    LOGGER.info("Log file: %s", log_path)
    LOGGER.info("Input: %s", input_root)
    LOGGER.info("Output: %s", output_root)

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

    LOGGER.info("Loading model: %s", args.model)
    LOGGER.info("SentenceTransformer kwargs: %s", st_kwargs)
    model = SentenceTransformer(args.model, **st_kwargs)
    if args.max_length is not None:
        model.max_seq_length = args.max_length
    LOGGER.info("Model loaded. max_seq_length=%s", model.max_seq_length)

    csv_files = collect_csv_files(input_root)
    LOGGER.info("Discovered %d csv file(s).", len(csv_files))

    ok_count = 0
    skipped_or_failed = 0
    run_started = time.time()

    for idx, csv_path in enumerate(csv_files, start=1):
        try:
            ok = process_file(
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
                show_progress_bar=not args.no_progress_bar,
                index=idx,
                total=len(csv_files),
            )
            if ok:
                ok_count += 1
            else:
                skipped_or_failed += 1
        except Exception as exc:
            skipped_or_failed += 1
            LOGGER.error("[%d/%d] Failed file: %s", idx, len(csv_files), csv_path)
            LOGGER.error("Exception: %s", exc)
            LOGGER.error(traceback.format_exc())
            if args.fail_fast:
                raise

    total_elapsed = time.time() - run_started
    LOGGER.info(
        "Finished. success=%d skipped_or_failed=%d elapsed=%.1fs",
        ok_count,
        skipped_or_failed,
        total_elapsed,
    )


if __name__ == "__main__":
    main()
