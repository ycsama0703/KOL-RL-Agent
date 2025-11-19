"""Evaluate a trained policy checkpoint on a replay buffer split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.analyzer import load_actor, run_policy
from src.training.data import load_buffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy on a replay buffer split.")
    parser.add_argument("--checkpoint", required=True, help="Path to actor or policy checkpoint (pt file).")
    parser.add_argument("--buffer", required=True, help="Replay buffer file to evaluate against (e.g., test.pt).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference.")
    parser.add_argument("--output", help="Optional path to dump metrics as JSON.")
    parser.add_argument("--positions-output", help="Optional CSV path to log per-date holdings/actions.")
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=0.01,
        help="Minimum absolute weight/weight delta treated as a position or action.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    buffer_path = Path(args.buffer)
    device = torch.device(args.device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not buffer_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {buffer_path}")

    buffer = load_buffer(buffer_path)
    state_dim = buffer["states"].shape[1]
    actor = load_actor(checkpoint_path, state_dim, device)
    metrics, positions_df = run_policy(actor, buffer, device, action_threshold=args.action_threshold)

    print(json.dumps(metrics, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"Saved metrics to {output_path}")
    if args.positions_output:
        positions_path = Path(args.positions_output)
        positions_path.parent.mkdir(parents=True, exist_ok=True)
        positions_df.to_csv(positions_path, index=False)
        print(f"Saved positions log to {positions_path}")


if __name__ == "__main__":
    main()
