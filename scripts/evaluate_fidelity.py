"""Evaluate how closely model actions replicate KOL baseline actions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predicted actions against baseline sentiment actions.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to signal_decisions_test.csv.",
    )
    parser.add_argument(
        "--prediction-column",
        default="trained_actions",
        help="Column to evaluate (e.g., trained_actions or baseline_actions).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save summary metrics as JSON/CSV.",
    )
    return parser.parse_args()


def parse_action_string(action_str: str) -> Dict[str, str]:
    actions: Dict[str, str] = {}
    if pd.isna(action_str) or not action_str:
        return actions
    parts = [item.strip() for item in action_str.split(";") if item.strip()]
    for part in parts:
        if ":" not in part:
            continue
        ticker, action = part.split(":", 1)
        actions[ticker.strip()] = action.strip()
    return actions


def evaluate(df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    total = 0
    exact_matches = 0
    per_label_totals: Counter[str] = Counter()
    per_label_correct: Counter[str] = Counter()
    mismatch_examples: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        baseline = parse_action_string(row["baseline_actions"])
        predicted = parse_action_string(row[pred_col])
        for ticker, base_action in baseline.items():
            pred_action = predicted.get(ticker, "MISSING")
            total += 1
            per_label_totals[base_action] += 1
            if base_action == pred_action:
                exact_matches += 1
                per_label_correct[base_action] += 1
            else:
                mismatch_examples.append(
                    {
                        "date": row["date"],
                        "ticker": ticker,
                        "baseline": base_action,
                        "predicted": pred_action,
                    }
                )

    accuracy = exact_matches / total if total else 0.0
    per_label_accuracy = {
        label: (per_label_correct[label] / per_label_totals[label])
        for label in per_label_totals
    }

    return {
        "total_samples": total,
        "matches": exact_matches,
        "accuracy": accuracy,
        "per_label_accuracy": per_label_accuracy,
        "mismatch_samples": mismatch_examples[:10],  # show up to 10 mismatches
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if "baseline_actions" not in df.columns:
        raise ValueError("Input file must contain baseline_actions column.")
    if args.prediction_column not in df.columns:
        raise ValueError(f"Column {args.prediction_column} not found in input file.")

    metrics = evaluate(df, args.prediction_column)
    print(f"Evaluating column: {args.prediction_column}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Per action accuracy:")
    for label, acc in metrics["per_label_accuracy"].items():
        print(f"  {label}: {acc:.4f}")
    if metrics["mismatch_samples"]:
        print("Sample mismatches (up to 10):")
        for sample in metrics["mismatch_samples"]:
            print(
                f"  {sample['date']} {sample['ticker']} - baseline={sample['baseline']} predicted={sample['predicted']}"
            )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(metrics).to_json(output_path)
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
