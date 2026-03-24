"""Plot five-point progression figure (dual-axis line chart).

Order:
Baseline -> WO_RL_COMPLETION -> WO_REGIME_SPLIT -> KICL -> WO_HARD

Y1 (left): performance metric (default: daily_return_mean)
Y2 (right): hard betrayal index HVC = UER + DRR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["BASELINE", "WO_RL_COMPLETION", "WO_REGIME_SPLIT", "KICL", "WO_HARD"]
LABELS = {
    "BASELINE": "Baseline",
    "WO_RL_COMPLETION": "WO_RL_COMPLETION",
    "WO_REGIME_SPLIT": "WO_REGIME_SPLIT",
    "KICL": "KICL (Full)",
    "WO_HARD": "WO_HARD",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot five-point dual-axis progression figure.")
    p.add_argument(
        "--input-csv",
        default="ablation study/five_point_compare/five_point_summary_by_source.csv",
        help="Input five-point summary csv (with source/method rows).",
    )
    p.add_argument(
        "--output-prefix",
        default="ablation study/five_point_compare/five_point_progression_dual_axis",
        help="Output prefix (without extension).",
    )
    p.add_argument(
        "--scope",
        choices=["overall", "x", "youtube"],
        default="overall",
        help="Aggregate both sources or plot one source only.",
    )
    p.add_argument(
        "--metric",
        choices=["daily_return_mean", "daily_sharpe_mean"],
        default="daily_return_mean",
        help="Left-axis metric.",
    )
    p.add_argument("--dpi", type=int, default=260)
    return p.parse_args()


def _metric_label(metric: str) -> str:
    if metric == "daily_sharpe_mean":
        return "Sharpe (daily mean)"
    return "Cumulative Return (daily mean)"


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if args.scope == "overall":
        sdf = df.groupby("method", as_index=False).mean(numeric_only=True)
        title_suffix = "ALL20"
    else:
        sdf = df[df["source"] == args.scope].copy()
        title_suffix = args.scope.upper()

    sdf = sdf[sdf["method"].isin(ORDER)].copy()
    sdf["method"] = pd.Categorical(sdf["method"], categories=ORDER, ordered=True)
    sdf = sdf.sort_values("method")
    if len(sdf) != len(ORDER):
        missing = [m for m in ORDER if m not in set(sdf["method"].astype(str))]
        raise RuntimeError(f"Missing methods in plot scope={args.scope}: {missing}")

    sdf["hvc"] = sdf["UER_mean"].fillna(0.0) + sdf["DRR_mean"].fillna(0.0)

    x = np.arange(len(sdf))
    y_metric = sdf[args.metric].astype(float).to_numpy()
    y_hvc = sdf["hvc"].astype(float).to_numpy()
    xlabels = [LABELS[m] for m in sdf["method"].astype(str)]

    fig, ax1 = plt.subplots(figsize=(10.2, 4.8))
    ax2 = ax1.twinx()

    # performance line
    ax1.plot(
        x,
        y_metric,
        color="#F39C12",
        marker="o",
        markersize=6,
        linewidth=2.4,
        label=_metric_label(args.metric),
        zorder=4,
    )
    for i, v in enumerate(y_metric):
        ax1.text(i, v + (0.01 if v >= 0 else -0.02), f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color="#b45309")

    # hard betrayal line
    ax2.plot(
        x,
        y_hvc,
        color="#DC2626",
        marker="s",
        markersize=6,
        linewidth=2.2,
        linestyle="--",
        label="Hard Betrayal (UER + DRR)",
        zorder=5,
    )
    for i, v in enumerate(y_hvc):
        ax2.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#991b1b")

    # visual cue for "full -> wo_hard" step
    ax1.axvline(3.5, color="#6b7280", linestyle=":", linewidth=1.0, alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, rotation=12, ha="right")
    ax1.set_ylabel(_metric_label(args.metric))
    ax2.set_ylabel("Hard Betrayal Index")
    ax1.set_title(f"Five-Point Progression ({title_suffix})", fontsize=14, fontweight="semibold", pad=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # y-ranges with headroom
    y1_min = min(y_metric.min(), 0.0) - 0.03
    y1_max = y_metric.max() + 0.08
    ax1.set_ylim(y1_min, y1_max)
    ax2.set_ylim(0.0, max(0.05, y_hvc.max() * 1.25))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, framealpha=0.92, fontsize=10)

    fig.tight_layout()
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

