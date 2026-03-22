#!/usr/bin/env python3
"""Paper figure: hard-betrayal-only view (no soft deviation)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["KICL", "AWAC", "IQL", "BC", "CQL", "TD3BC"]
METHOD_COLORS = {
    "KICL": "#F39C12",
    "AWAC": "#17BECF",
    "IQL": "#2CA02C",
    "BC": "#8C564B",
    "CQL": "#D62728",
    "TD3BC": "#9467BD",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        default=(
            "benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/"
            "hard_only_betrayal_summary.csv"
        ),
        help="Hard-only summary CSV.",
    )
    p.add_argument(
        "--output-dir",
        default="benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20",
        help="Output directory.",
    )
    p.add_argument("--dpi", type=int, default=360)
    return p.parse_args()


def _methods_present(df: pd.DataFrame) -> List[str]:
    present = set(df["method"].astype(str))
    return [m for m in METHOD_ORDER if m in present]


def main() -> None:
    args = parse_args()
    in_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    sources = [s for s in ["x", "youtube"] if s in set(df["source"].astype(str))]

    fig, axes = plt.subplots(2, len(sources), figsize=(10.0, 6.5), squeeze=False)

    for col, src in enumerate(sources):
        sdf = df[df["source"] == src].copy()
        methods = _methods_present(sdf)
        sdf["method"] = pd.Categorical(sdf["method"], categories=methods, ordered=True)
        sdf = sdf.sort_values("method")
        x = np.arange(len(methods))
        colors = [METHOD_COLORS.get(m, "#4C72B0") for m in methods]

        # Row 1: overall hard betrayal probability
        ax1 = axes[0, col]
        bars = ax1.bar(
            x,
            sdf["p_hard_betrayal"].to_numpy(dtype=float),
            width=0.68,
            color=colors,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
        )
        for i, m in enumerate(methods):
            if m == "KICL":
                bars[i].set_edgecolor("#111111")
                bars[i].set_linewidth(2.0)
                ax1.text(i, bars[i].get_height() + 0.02, "KICL", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=20, fontsize=8.5)
        ax1.grid(axis="y", linestyle="--", alpha=0.30, linewidth=0.6)
        ax1.set_ylabel("Hard betrayal rate")
        ax1.set_title("X" if src == "x" else "YouTube", fontsize=11.5)

        # Row 2: uplift under profitable events (hard-only)
        ax2 = axes[1, col]
        vals = sdf["uplift_hard_only"].to_numpy(dtype=float)
        bars2 = ax2.bar(
            x,
            vals,
            width=0.68,
            color=colors,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
        )
        for i, m in enumerate(methods):
            if m == "KICL":
                bars2[i].set_edgecolor("#111111")
                bars2[i].set_linewidth(2.0)
        ax2.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.8)
        # auto-limits with small padding
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        pad = max(0.02, 0.08 * max(abs(vmin), abs(vmax), 1e-6))
        ax2.set_ylim(vmin - pad, vmax + pad)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=20, fontsize=8.5)
        ax2.grid(axis="y", linestyle="--", alpha=0.30, linewidth=0.6)
        ax2.set_ylabel("Hard-only uplift\n(P(hard|profit)-P(hard|non-profit))")

    fig.suptitle("Hard-Betrayal-Only View (Soft Deviation Excluded)", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.96), w_pad=0.9, h_pad=1.0)

    png = out_dir / "hard_betrayal_only_paper_figure.png"
    pdf = out_dir / "hard_betrayal_only_paper_figure.pdf"
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()

