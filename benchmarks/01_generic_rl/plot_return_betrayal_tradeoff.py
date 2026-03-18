"""Plot return-vs-betrayal tradeoff for selected KOLs from compare outputs.

Inputs per KOL (under benchmarks/compare/<source>/<kol>/):
- daily_metrics_compare.csv
- betrayal_metrics_compare.csv

Outputs:
- <output_prefix>.png
- <output_prefix>.pdf
- <output_prefix>_points.csv
- <output_prefix>_method_mean.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_KOLS = [
    "Ale_s_World_of_Stocks",
    "Invest_with_Henry",
    "Dividend_Data",
    "The_Maverick_of_Wall_Street",
    "MarketBeat",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot return-vs-betrayal tradeoff.")
    p.add_argument(
        "--compare-root",
        default="benchmarks/compare/youtube",
        help="Root of per-KOL compare folders (e.g., benchmarks/compare/youtube).",
    )
    p.add_argument(
        "--kols",
        default=",".join(DEFAULT_KOLS),
        help="Comma-separated KOL list.",
    )
    p.add_argument(
        "--output-prefix",
        default="benchmarks/compare/youtube_tradeoff_selected5",
        help="Output file prefix.",
    )
    p.add_argument(
        "--ours-name",
        default="KICL",
        help="Method to emphasize in the plot.",
    )
    return p.parse_args()


def method_palette() -> dict:
    return {
        "KICL": "#ff7f0e",
        "BC": "#8c564b",
        "IQL": "#2ca02c",
        "CQL": "#d62728",
        "TD3BC": "#9467bd",
        "AWAC": "#17becf",
    }


def load_points(compare_root: Path, kols: List[str]) -> pd.DataFrame:
    rows = []
    for kol in kols:
        dpath = compare_root / kol / "daily_metrics_compare.csv"
        bpath = compare_root / kol / "betrayal_metrics_compare.csv"
        if not dpath.exists() or not bpath.exists():
            continue
        ddf = pd.read_csv(dpath)
        bdf = pd.read_csv(bpath)
        m = ddf.merge(
            bdf[["method", "mean_normalized_deviation", "entry_violation_rate", "reversal_rate"]],
            on="method",
            how="inner",
        )
        m["kol"] = kol
        # Higher is better; maps deviation in [0, +inf) to (0, 1].
        m["intent_consistency"] = 1.0 / (1.0 + pd.to_numeric(m["mean_normalized_deviation"], errors="coerce"))
        m["daily_return"] = pd.to_numeric(m["trained_cumulative_return"], errors="coerce")
        rows.append(
            m[
                [
                    "kol",
                    "method",
                    "daily_return",
                    "mean_normalized_deviation",
                    "intent_consistency",
                    "entry_violation_rate",
                    "reversal_rate",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["daily_return", "intent_consistency"])
    return out


def plot_tradeoff(points: pd.DataFrame, ours_name: str, output_prefix: Path) -> None:
    palette = method_palette()

    summary = (
        points.groupby("method", as_index=False)[
            [
                "daily_return",
                "intent_consistency",
                "mean_normalized_deviation",
                "entry_violation_rate",
                "reversal_rate",
            ]
        ]
        .mean()
        .rename(
            columns={
                "daily_return": "mean_daily_return",
                "intent_consistency": "mean_intent_consistency",
                "mean_normalized_deviation": "mean_dev",
                "entry_violation_rate": "mean_entry_violation_rate",
                "reversal_rate": "mean_reversal_rate",
            }
        )
    )

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    ax.set_facecolor("#fbfcff")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3)

    methods = sorted(points["method"].unique().tolist())
    for method in methods:
        sub = points[points["method"] == method]
        color = palette.get(method, "#4c4c4c")

        # KOL-level points (semi-transparent)
        ax.scatter(
            sub["intent_consistency"],
            sub["daily_return"],
            s=46,
            alpha=0.55,
            color=color,
            edgecolors="white",
            linewidths=0.5,
        )

        # Method mean point (bold)
        mean_row = summary[summary["method"] == method].iloc[0]
        marker = "X" if method == ours_name else "o"
        size = 240 if method == ours_name else 140
        edge = "black" if method == ours_name else "white"
        lw = 1.2 if method == ours_name else 0.8
        ax.scatter(
            [mean_row["mean_intent_consistency"]],
            [mean_row["mean_daily_return"]],
            s=size,
            marker=marker,
            color=color,
            edgecolors=edge,
            linewidths=lw,
            zorder=5,
            label=f"{method} mean",
        )
        ax.annotate(
            method,
            (mean_row["mean_intent_consistency"], mean_row["mean_daily_return"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=10,
            weight="bold" if method == ours_name else "normal",
        )

    ax.set_xlabel("Intent Consistency (higher is better)")
    ax.set_ylabel("Daily Cumulative Return (higher is better)")
    ax.set_title("Return vs Intent-Consistency Tradeoff (Selected YouTube KOLs)")
    ax.legend(loc="lower left", ncol=2, fontsize=9, frameon=True, framealpha=0.9)
    fig.tight_layout()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)

    points.to_csv(output_prefix.with_name(output_prefix.name + "_points.csv"), index=False)
    summary.to_csv(output_prefix.with_name(output_prefix.name + "_method_mean.csv"), index=False)


def main() -> None:
    args = parse_args()
    compare_root = Path(args.compare_root)
    kols = [x.strip() for x in args.kols.split(",") if x.strip()]
    output_prefix = Path(args.output_prefix)

    points = load_points(compare_root=compare_root, kols=kols)
    if points.empty:
        raise SystemExit("No valid points loaded. Check compare root / KOL names.")
    plot_tradeoff(points=points, ours_name=args.ours_name, output_prefix=output_prefix)
    print(f"Saved plot: {output_prefix.with_suffix('.png')}")
    print(f"Saved table: {output_prefix.with_name(output_prefix.name + '_method_mean.csv')}")


if __name__ == "__main__":
    main()

