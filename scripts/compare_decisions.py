"""Compare decisions between a trained policy and a baseline on the same buffer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.analyzer import load_actor, run_policy
from src.training.data import load_buffer
from src.training.models import ActorNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained vs. baseline policy decisions on a buffer.")
    parser.add_argument("--trained", required=True, help="Checkpoint path for the trained policy.")
    parser.add_argument("--buffer", required=True, help="Replay buffer file (e.g., test.pt).")
    parser.add_argument(
        "--baseline",
        help="Optional checkpoint for the baseline policy. If omitted, uses a randomly initialised actor.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device.")
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=0.01,
        help="Threshold for determining meaningful position/action changes.",
    )
    parser.add_argument("--output", required=True, help="CSV path to store decision comparison results.")
    parser.add_argument("--metrics-output", help="Optional JSON path to store metrics for both policies.")
    return parser.parse_args()


def _rename_columns(prefix: str, df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "raw_score": f"raw_score_{prefix}",
            "prev_weight": f"prev_weight_{prefix}",
            "weight": f"weight_{prefix}",
            "weight_delta": f"weight_delta_{prefix}",
            "allocation": f"allocation_{prefix}",
            "action": f"action_{prefix}",
        }
    )


def _classify_relative(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "MORE_LONG"
    if delta < -threshold:
        return "MORE_SHORT"
    return "SIMILAR"


def main() -> None:
    args = parse_args()
    buffer_path = Path(args.buffer)
    trained_path = Path(args.trained)
    baseline_path = Path(args.baseline) if args.baseline else None
    device = torch.device(args.device)

    if not buffer_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {buffer_path}")
    if not trained_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found: {trained_path}")
    if baseline_path and not baseline_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")

    buffer = load_buffer(buffer_path)
    state_dim = buffer["states"].shape[1]

    trained_actor = load_actor(trained_path, state_dim, device)
    trained_metrics, trained_positions = run_policy(
        trained_actor, buffer, device, action_threshold=args.action_threshold
    )

    if baseline_path is not None:
        baseline_actor = load_actor(baseline_path, state_dim, device)
    else:
        baseline_actor = ActorNetwork(state_dim).to(device)
        baseline_actor.eval()

    baseline_metrics, baseline_positions = run_policy(
        baseline_actor, buffer, device, action_threshold=args.action_threshold
    )

    trained_positions = _rename_columns("trained", trained_positions)
    baseline_positions = _rename_columns("baseline", baseline_positions)

    merged = pd.merge(
        trained_positions,
        baseline_positions,
        on=["date", "ticker", "reward"],
        how="outer",
        sort=True,
    )
    numeric_cols = [
        "raw_score_trained",
        "prev_weight_trained",
        "weight_trained",
        "weight_delta_trained",
        "allocation_trained",
        "raw_score_baseline",
        "prev_weight_baseline",
        "weight_baseline",
        "weight_delta_baseline",
        "allocation_baseline",
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    for col in ("action_trained", "action_baseline"):
        if col in merged.columns:
            merged[col] = merged[col].fillna("NONE")
    merged["weight_delta_vs_baseline"] = merged["weight_trained"] - merged["weight_baseline"]
    merged["relative_action"] = merged["weight_delta_vs_baseline"].apply(
        lambda delta: _classify_relative(delta, args.action_threshold)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["date", "ticker"], inplace=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved decision comparison to {output_path}")

    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump({"trained": trained_metrics, "baseline": baseline_metrics}, fp, indent=2)
        print(f"Saved metrics summary to {metrics_path}")


if __name__ == "__main__":
    main()
