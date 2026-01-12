"""Export replay buffer .pt files to CSV for inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export replay buffer .pt files to CSV.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a .pt file or a directory containing .pt files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/replay_buffer_csv",
        help="Directory to write CSVs (mirrors input structure).",
    )
    parser.add_argument(
        "--include-states",
        action="store_true",
        help="Include state and next_state vectors as expanded columns.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSVs if they already exist.",
    )
    return parser.parse_args()


def iter_pt_files(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix == ".pt":
        yield path
        return
    if path.is_dir():
        yield from sorted(path.rglob("*.pt"))


def as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def add_tensor_columns(data: dict, prefix: str, array: np.ndarray) -> None:
    if array.ndim == 1:
        data[prefix] = array
        return
    for idx in range(array.shape[1]):
        data[f"{prefix}_{idx}"] = array[:, idx]


def build_dataframe(buffer: dict, include_states: bool) -> pd.DataFrame:
    data: dict = {}

    for key in (
        "actions",
        "baseline_actions",
        "baseline_action",
        "next_baseline_action",
        "rewards",
        "portfolio_rewards",
        "dones",
    ):
        if key in buffer:
            arr = as_numpy(buffer[key])
            arr = np.squeeze(arr, axis=-1) if arr.ndim == 2 and arr.shape[1] == 1 else arr
            data[key] = arr

    meta = buffer.get("meta", {})
    if isinstance(meta, dict):
        for key, value in meta.items():
            data[f"meta_{key}"] = as_numpy(value)

    if include_states:
        if "states" in buffer:
            add_tensor_columns(data, "state", as_numpy(buffer["states"]))
        if "next_states" in buffer:
            add_tensor_columns(data, "next_state", as_numpy(buffer["next_states"]))

    return pd.DataFrame(data)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    files = list(iter_pt_files(input_path))
    if not files:
        raise FileNotFoundError(f"No .pt files found under {input_path}")

    for pt_path in files:
        try:
            relative = pt_path.relative_to(input_path)
        except ValueError:
            relative = pt_path.name
        output_path = (output_root / relative).with_suffix(".csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

        buffer = torch.load(pt_path, map_location="cpu")
        df = build_dataframe(buffer, include_states=args.include_states)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
