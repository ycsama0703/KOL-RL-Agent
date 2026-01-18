"""Evaluate a trained policy checkpoint on a replay buffer split."""

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
from train import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy on a replay buffer split.")
    parser.add_argument("--checkpoint", required=True, help="Path to actor or policy checkpoint (pt file).")
    parser.add_argument("--buffer", required=True, help="Replay buffer file to evaluate against (e.g., test.pt).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference.")
    parser.add_argument("--output", help="Optional path to dump metrics as JSON.")
    parser.add_argument("--positions-output", help="Optional CSV path to log per-date holdings/actions.")
    parser.add_argument(
        "--output-dir",
        help="Optional directory to write metrics_test.json and positions_test.csv.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate equity curve plot (requires --output-dir or --plot-output).",
    )
    parser.add_argument(
        "--plot-output",
        help="Optional path for equity curve plot (PNG).",
    )
    parser.add_argument(
        "--daily-output-dir",
        help="Optional directory to write daily metrics/plot (calendar-day aggregation).",
    )
    parser.add_argument(
        "--daily-benchmark-ticker",
        help="Optional benchmark ticker for daily plot (e.g., SPY or ^GSPC).",
    )
    parser.add_argument(
        "--daily-benchmark-label",
        help="Label for daily benchmark curve (default: same as ticker).",
    )
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

    def daily_equity(positions: pd.DataFrame) -> pd.DataFrame:
        df = positions.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["weighted_return"] = df["weight"] * df["reward"]
        daily = df.groupby("date", as_index=False)["weighted_return"].sum().sort_values("date")
        daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
        return daily

    # Daily (calendar) metrics: collapse multiple videos within a day.
    daily_train = daily_equity(positions_df)
    daily_returns = daily_train["weighted_return"].to_numpy()
    daily_metrics = compute_metrics(daily_returns) if len(daily_returns) else metrics
    metrics_out = {**metrics, "daily_metrics": daily_metrics}

    print(json.dumps(metrics_out, indent=2))

    output_path = Path(args.output) if args.output else None
    positions_path = Path(args.positions_output) if args.positions_output else None
    if args.output_dir:
        base = Path(args.output_dir)
        base.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            output_path = base / "metrics_test.json"
        if positions_path is None:
            positions_path = base / "positions_test.csv"

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics_out, fp, indent=2)
        print(f"Saved metrics to {output_path}")
    if positions_path:
        positions_path.parent.mkdir(parents=True, exist_ok=True)
        positions_df.to_csv(positions_path, index=False)
        print(f"Saved positions log to {positions_path}")

    base_positions = None
    if args.plot or args.plot_output or args.daily_output_dir:
        plot_path = Path(args.plot_output) if args.plot_output else None
        if plot_path is None:
            if not args.output_dir:
                raise SystemExit("Plot requested but no --output-dir or --plot-output provided.")
            plot_path = Path(args.output_dir) / "equity_test.png"

        class ZeroActor(torch.nn.Module):
            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return torch.zeros((state.size(0), 1), device=state.device)

        _, base_positions = run_policy(
            ZeroActor().to(device),
            buffer,
            device,
            action_threshold=args.action_threshold,
        )

        def equity_series(positions: pd.DataFrame) -> pd.DataFrame:
            df = positions.copy()
            df["date"] = pd.to_datetime(df["date"])
            df["weighted_return"] = df["weight"] * df["reward"]
            daily = df.groupby("date", as_index=False)["weighted_return"].sum()
            daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
            return daily

        base = equity_series(base_positions).rename(columns={"equity": "equity_baseline"})
        train = equity_series(positions_df).rename(columns={"equity": "equity_trained"})
        daily = pd.merge(base, train, on="date", how="inner")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]
        except ImportError as exc:
            raise SystemExit(
                "matplotlib is required for plotting. Please install it with `pip install matplotlib`."
            ) from exc

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily["date"], daily["equity_baseline"], label="Baseline", linewidth=1.8)
        ax.plot(daily["date"], daily["equity_trained"], label="Trained", linewidth=1.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.set_title("Baseline vs Trained Equity (Test)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved equity curve figure to {plot_path}")

    if args.daily_output_dir:
        daily_dir = Path(args.daily_output_dir)
        daily_dir.mkdir(parents=True, exist_ok=True)

        if base_positions is None:
            class ZeroActor(torch.nn.Module):
                def forward(self, state: torch.Tensor) -> torch.Tensor:
                    return torch.zeros((state.size(0), 1), device=state.device)

            _, base_positions = run_policy(
                ZeroActor().to(device),
                buffer,
                device,
                action_threshold=args.action_threshold,
            )

        daily_base = daily_equity(base_positions)
        daily_train = daily_equity(positions_df)

        metrics_daily = {
            "trained": compute_metrics(daily_train["weighted_return"].to_numpy())
            if len(daily_train)
            else metrics,
            "baseline": compute_metrics(daily_base["weighted_return"].to_numpy())
            if len(daily_base)
            else metrics,
        }
        metrics_path = daily_dir / "metrics_daily.json"
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics_daily, fp, indent=2)
        print(f"Saved daily metrics to {metrics_path}")

        daily_merge = pd.merge(
            daily_base.rename(columns={"equity": "equity_baseline"}),
            daily_train.rename(columns={"equity": "equity_trained"}),
            on="date",
            how="inner",
        )
        daily_csv = daily_dir / "equity_daily.csv"
        daily_merge.to_csv(daily_csv, index=False)
        print(f"Saved daily equity CSV to {daily_csv}")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]
        except ImportError as exc:
            raise SystemExit(
                "matplotlib is required for plotting. Please install it with `pip install matplotlib`."
            ) from exc

        bench_series = None
        if args.daily_benchmark_ticker:
            try:
                import yfinance as yf  # type: ignore[import]
            except ImportError as exc:
                raise SystemExit("yfinance is required for benchmark plots. Please install it.") from exc

            start = daily_merge["date"].min().date()
            end = daily_merge["date"].max().date()
            data = yf.download(args.daily_benchmark_ticker, start=start, end=end, auto_adjust=False)
            if not data.empty:
                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                bench_eq = close / float(close.iloc[0])
                bench_eq = bench_eq.reindex(pd.to_datetime(daily_merge["date"]), method="ffill")
                bench_series = bench_eq.values

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_merge["date"], daily_merge["equity_baseline"], label="Baseline", linewidth=1.8)
        ax.plot(daily_merge["date"], daily_merge["equity_trained"], label="Trained", linewidth=1.8)
        if bench_series is not None:
            bench_label = args.daily_benchmark_label or args.daily_benchmark_ticker
            ax.plot(
                daily_merge["date"],
                bench_series,
                label=bench_label,
                linewidth=1.5,
                linestyle="--",
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.set_title("Baseline vs Trained Equity (Daily)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()

        fig_path = daily_dir / "equity_daily.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved daily equity plot to {fig_path}")


if __name__ == "__main__":
    main()
