"""Compare two trained checkpoints on the same buffer and plot equity curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.analyzer import load_actor, run_policy
from src.training.data import load_buffer
from train import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two checkpoints on the same buffer.")
    parser.add_argument("--checkpoint-a", required=True, help="First policy checkpoint (policy.pt).")
    parser.add_argument("--checkpoint-b", required=True, help="Second policy checkpoint (policy.pt).")
    parser.add_argument("--buffer", required=True, help="Replay buffer (test.pt).")
    parser.add_argument("--output-dir", required=True, help="Output directory for metrics/plots.")
    parser.add_argument("--label-a", default="Model A", help="Label for checkpoint A in plots.")
    parser.add_argument("--label-b", default="Model B", help="Label for checkpoint B in plots.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-threshold", type=float, default=0.02)
    parser.add_argument("--include-baseline", action="store_true", help="Include baseline curve in plots.")
    parser.add_argument("--daily-plot", action="store_true", help="Also plot daily equity curves.")
    return parser.parse_args()


def equity_series(positions: pd.DataFrame, normalize_dates: bool = False) -> pd.Series:
    df = positions.copy()
    dates = pd.to_datetime(df["date"], errors="coerce")
    if normalize_dates:
        dates = dates.dt.normalize()
    df["date"] = dates
    df["weighted_return"] = df["weight"] * df["reward"]
    daily = df.groupby("date", as_index=True)["weighted_return"].sum().sort_index()
    return (1.0 + daily).cumprod()


def metrics_from_positions(positions: pd.DataFrame, normalize_dates: bool = False) -> dict:
    df = positions.copy()
    dates = pd.to_datetime(df["date"], errors="coerce")
    if normalize_dates:
        dates = dates.dt.normalize()
    df["date"] = dates
    df["weighted_return"] = df["weight"] * df["reward"]
    daily_returns = df.groupby("date", as_index=True)["weighted_return"].sum().sort_index().to_numpy()
    return compute_metrics(daily_returns) if len(daily_returns) else {"cumulative_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer = load_buffer(args.buffer)
    state_dim = buffer["states"].shape[1]
    device = torch.device(args.device)

    actor_a = load_actor(Path(args.checkpoint_a), state_dim, device)
    actor_b = load_actor(Path(args.checkpoint_b), state_dim, device)

    metrics_a, positions_a = run_policy(actor_a, buffer, device, action_threshold=args.action_threshold)
    metrics_b, positions_b = run_policy(actor_b, buffer, device, action_threshold=args.action_threshold)

    metrics = {
        "model_a": {"event": metrics_a, "daily": metrics_from_positions(positions_a, normalize_dates=True)},
        "model_b": {"event": metrics_b, "daily": metrics_from_positions(positions_b, normalize_dates=True)},
        "labels": {"model_a": args.label_a, "model_b": args.label_b},
    }
    metrics_path = output_dir / "compare_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved metrics to {metrics_path}")

    eq_a = equity_series(positions_a, normalize_dates=False)
    eq_b = equity_series(positions_b, normalize_dates=False)

    baseline_eq = None
    if args.include_baseline:
        class ZeroActor(torch.nn.Module):
            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return torch.zeros((state.size(0), 1), device=state.device)

        _, base_positions = run_policy(ZeroActor().to(device), buffer, device, action_threshold=args.action_threshold)
        baseline_eq = equity_series(base_positions, normalize_dates=False)

    event_df = pd.DataFrame({"equity_a": eq_a, "equity_b": eq_b}).dropna().reset_index()
    if baseline_eq is not None:
        event_df = event_df.merge(baseline_eq.rename("equity_baseline"), left_on="date", right_index=True, how="inner")
    event_csv = output_dir / "equity_event.csv"
    event_df.to_csv(event_csv, index=False)
    print(f"Saved event equity CSV to {event_csv}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting.") from exc

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(event_df["date"], event_df["equity_a"], label=args.label_a, linewidth=1.8)
    ax.plot(event_df["date"], event_df["equity_b"], label=args.label_b, linewidth=1.8)
    if "equity_baseline" in event_df.columns:
        ax.plot(event_df["date"], event_df["equity_baseline"], label="Baseline", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.set_title("Equity Comparison (Event-Time)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()

    fig_path = output_dir / "equity_event.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved event equity plot to {fig_path}")

    if args.daily_plot:
        daily_a = equity_series(positions_a, normalize_dates=True)
        daily_b = equity_series(positions_b, normalize_dates=True)
        daily_df = pd.DataFrame({"equity_a": daily_a, "equity_b": daily_b}).dropna().reset_index()
        daily_csv = output_dir / "equity_daily.csv"
        daily_df.to_csv(daily_csv, index=False)
        print(f"Saved daily equity CSV to {daily_csv}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_df["date"], daily_df["equity_a"], label=args.label_a, linewidth=1.8)
        ax.plot(daily_df["date"], daily_df["equity_b"], label=args.label_b, linewidth=1.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.set_title("Equity Comparison (Daily)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig_path = output_dir / "equity_daily.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved daily equity plot to {fig_path}")


if __name__ == "__main__":
    main()
