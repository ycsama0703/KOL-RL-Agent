"""Compare RORL benchmark performance vs our trained/baseline on the same dates."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Compare RORL benchmark vs our policy.")
    parser.add_argument("--checkpoint", required=True, help="Path to our policy checkpoint (policy.pt).")
    parser.add_argument("--buffer", required=True, help="Replay buffer (test.pt) for our policy.")
    parser.add_argument("--rorl-performance", required=True, help="RORL performance.csv path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=0.01,
        help="Minimum absolute weight/weight delta treated as a position or action.",
    )
    parser.add_argument("--align", choices=["intersection", "ours", "rorl"], default="intersection")
    parser.add_argument("--output-csv", required=True, help="Path to save comparison CSV.")
    parser.add_argument("--output-figure", help="Optional path to save comparison plot.")
    parser.add_argument("--no-rebase", action="store_true", help="Do not rebase curves to 1 in plot.")
    return parser.parse_args()


def positions_to_daily(positions: pd.DataFrame) -> pd.DataFrame:
    df = positions.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["weighted_return"] = df["weight"] * df["reward"]
    daily = df.groupby("date", as_index=True)["weighted_return"].sum().sort_index()
    out = pd.DataFrame({"date": daily.index, "daily_return": daily.values})
    out["nav"] = (1.0 + out["daily_return"]).cumprod()
    out["cumulative_return"] = out["nav"] - 1.0
    return out


def reindex_daily(daily: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    series = daily.set_index("date")["daily_return"].reindex(dates, fill_value=0.0)
    out = pd.DataFrame({"date": dates, "daily_return": series.values})
    out["nav"] = (1.0 + out["daily_return"]).cumprod()
    out["cumulative_return"] = out["nav"] - 1.0
    return out


def rebase(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    first = series.iloc[0]
    if pd.isna(first) or first == 0:
        return series
    return series / first


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    buffer = load_buffer(args.buffer)
    state_dim = buffer["states"].shape[1]
    actor = load_actor(Path(args.checkpoint), state_dim, device)

    class ZeroActor(torch.nn.Module):
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return torch.zeros((state.size(0), 1), device=state.device)

    _, positions_base = run_policy(
        ZeroActor().to(device),
        buffer,
        device,
        action_threshold=args.action_threshold,
    )
    _, positions_train = run_policy(
        actor,
        buffer,
        device,
        action_threshold=args.action_threshold,
    )

    daily_base = positions_to_daily(positions_base).rename(
        columns={
            "daily_return": "daily_return_baseline",
            "nav": "nav_baseline",
            "cumulative_return": "cumulative_return_baseline",
        }
    )
    daily_train = positions_to_daily(positions_train).rename(
        columns={
            "daily_return": "daily_return_trained",
            "nav": "nav_trained",
            "cumulative_return": "cumulative_return_trained",
        }
    )

    rorl = pd.read_csv(args.rorl_performance)
    if "Date" not in rorl.columns:
        raise ValueError("RORL performance.csv must have a Date column.")
    rorl["date"] = pd.to_datetime(rorl["Date"]).dt.normalize()
    rorl = rorl.rename(
        columns={
            "NAV": "nav_rorl",
            "Daily_Return": "daily_return_rorl",
            "Cumulative_Return": "cumulative_return_rorl",
        }
    )[["date", "nav_rorl", "daily_return_rorl", "cumulative_return_rorl"]]

    if args.align == "intersection":
        merged = rorl.merge(daily_train, on="date", how="inner").merge(daily_base, on="date", how="inner")
    elif args.align == "ours":
        merged = daily_train.merge(rorl, on="date", how="left").merge(daily_base, on="date", how="left")
    else:
        dates = pd.DatetimeIndex(rorl["date"])
        daily_train = reindex_daily(
            daily_train.rename(
                columns={
                    "daily_return_trained": "daily_return",
                    "nav_trained": "nav",
                    "cumulative_return_trained": "cumulative_return",
                }
            ),
            dates,
        ).rename(
            columns={
                "daily_return": "daily_return_trained",
                "nav": "nav_trained",
                "cumulative_return": "cumulative_return_trained",
            }
        )
        daily_base = reindex_daily(
            daily_base.rename(
                columns={
                    "daily_return_baseline": "daily_return",
                    "nav_baseline": "nav",
                    "cumulative_return_baseline": "cumulative_return",
                }
            ),
            dates,
        ).rename(
            columns={
                "daily_return": "daily_return_baseline",
                "nav": "nav_baseline",
                "cumulative_return": "cumulative_return_baseline",
            }
        )
        merged = rorl.merge(daily_train, on="date", how="left").merge(daily_base, on="date", how="left")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Saved comparison CSV to {out_path}")

    # Print summary metrics on aligned dates
    if "daily_return_trained" in merged.columns:
        metrics = compute_metrics(merged["daily_return_trained"].fillna(0.0).to_numpy())
        print("Trained metrics (aligned):", metrics)
    if "daily_return_baseline" in merged.columns:
        metrics = compute_metrics(merged["daily_return_baseline"].fillna(0.0).to_numpy())
        print("Baseline metrics (aligned):", metrics)
    if "daily_return_rorl" in merged.columns:
        metrics = compute_metrics(merged["daily_return_rorl"].fillna(0.0).to_numpy())
        print("RORL metrics (aligned):", metrics)

    if args.output_figure:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]
        except ImportError as exc:
            raise SystemExit("matplotlib is required for plotting. Please install it.") from exc

        plot_df = merged.copy()
        if not args.no_rebase:
            plot_df["nav_rorl"] = rebase(plot_df["nav_rorl"])
            plot_df["nav_trained"] = rebase(plot_df["nav_trained"])
            plot_df["nav_baseline"] = rebase(plot_df["nav_baseline"])

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df["date"], plot_df["nav_rorl"], label="RORL", linewidth=1.8)
        plt.plot(plot_df["date"], plot_df["nav_trained"], label="Trained", linewidth=1.8)
        plt.plot(plot_df["date"], plot_df["nav_baseline"], label="Baseline", linewidth=1.8)
        plt.title("RORL vs Trained vs Baseline")
        plt.xlabel("Date")
        plt.ylabel("NAV")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fig_path = Path(args.output_figure)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved comparison plot to {fig_path}")


if __name__ == "__main__":
    main()
