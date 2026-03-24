"""Plot 5-point ablation figure for paper use.

Methods:
- BASELINE
- KICL
- WO_HARD
- WO_RL_COMPLETION
- WO_REGIME_SPLIT

Per source (X/YouTube):
- Bar: daily mean cumulative return
- Line (2nd y-axis): hard betrayal index HVC = UER + DRR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["BASELINE", "KICL", "WO_HARD", "WO_RL_COMPLETION", "WO_REGIME_SPLIT"]
COLORS = {
    "BASELINE": "#4C78A8",
    "KICL": "#F39C12",
    "WO_HARD": "#E45756",
    "WO_RL_COMPLETION": "#72B7B2",
    "WO_REGIME_SPLIT": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot five-point ablation figure.")
    p.add_argument(
        "--input-csv",
        default="ablation study/five_point_compare/five_point_summary_by_source.csv",
        help="five_point_summary_by_source.csv path",
    )
    p.add_argument(
        "--output-prefix",
        default="ablation study/five_point_compare/five_point_ablation_main",
        help="Output figure prefix (without extension)",
    )
    p.add_argument("--dpi", type=int, default=260)
    return p.parse_args()


def _format_method_label(m: str) -> str:
    mapping = {
        "BASELINE": "Baseline",
        "KICL": "KICL",
        "WO_HARD": "w/o Hard",
        "WO_RL_COMPLETION": "w/o RL Comp.",
        "WO_REGIME_SPLIT": "w/o Regime Split",
    }
    return mapping.get(m, m)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = df[df["method"].isin(ORDER)].copy()
    df["method"] = pd.Categorical(df["method"], categories=ORDER, ordered=True)
    df = df.sort_values(["source", "method"])
    df["hvc"] = df["UER_mean"].fillna(0.0) + df["DRR_mean"].fillna(0.0)

    sources = [s for s in ["x", "youtube"] if s in set(df["source"])]
    if not sources:
        raise RuntimeError("No valid source rows found in input csv.")

    fig, axes = plt.subplots(
        1,
        len(sources),
        figsize=(7.2 * len(sources), 4.8),
        constrained_layout=True,
    )
    if len(sources) == 1:
        axes = [axes]

    for ax, src in zip(axes, sources):
        sdf = df[df["source"] == src].set_index("method").reindex(ORDER).reset_index()
        x = np.arange(len(sdf))
        returns = sdf["daily_return_mean"].astype(float).to_numpy()
        hvc = sdf["hvc"].astype(float).to_numpy()
        methods = sdf["method"].astype(str).tolist()

        bar_colors = [COLORS.get(m, "#888888") for m in methods]
        bars = ax.bar(
            x,
            returns,
            color=bar_colors,
            alpha=0.9,
            edgecolor="#2f2f2f",
            linewidth=0.8,
            zorder=2,
        )

        # Highlight KICL bar
        for i, m in enumerate(methods):
            if m == "KICL":
                bars[i].set_linewidth(2.4)
                bars[i].set_edgecolor("#111111")

        for i, v in enumerate(returns):
            ax.text(
                x[i],
                v + (0.008 if v >= 0 else -0.012),
                f"{v:.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9,
                color="#1f1f1f",
                zorder=5,
            )

        ax2 = ax.twinx()
        ax2.plot(
            x,
            hvc,
            color="#6B7280",
            linestyle="--",
            marker="o",
            markersize=5,
            linewidth=1.8,
            zorder=4,
            label="HVC (UER+DRR)",
        )
        for i, v in enumerate(hvc):
            ax2.text(
                x[i],
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#4b5563",
                zorder=6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([_format_method_label(m) for m in methods], rotation=18, ha="right")
        ax.set_title("X" if src == "x" else "YouTube", fontsize=16, fontweight="semibold", pad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.28, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        ax.set_ylim(min(0.0, returns.min() - 0.03), max(returns.max() + 0.08, 0.06))
        ax2.set_ylim(0.0, max(0.05, hvc.max() * 1.25))

        ax.set_ylabel("Daily mean cumulative return")
        ax2.set_ylabel("Hard betrayal index (UER + DRR)")

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color=COLORS[m], lw=8, label=_format_method_label(m))
        for m in ORDER
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="#6B7280",
            linestyle="--",
            marker="o",
            lw=1.8,
            label="HVC (UER+DRR)",
        )
    )
    fig.legend(handles=handles, ncol=3, loc="upper center", frameon=True, bbox_to_anchor=(0.5, 1.05))

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

