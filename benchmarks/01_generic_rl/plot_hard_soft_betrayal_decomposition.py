#!/usr/bin/env python3
"""Plot hard-vs-soft betrayal decomposition for the profit-linked experiment.

This script is designed to support the paper narrative:
1) KICL profit-linked uplift is mostly from non-hard (soft) completion.
2) Other methods' betrayal profile is often hard-dominated.
3) Optional hard-only reporting (exclude soft deviation from betrayal).
"""

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
HARD_COLOR = "#C0392B"
SOFT_COLOR = "#2E86DE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        default=(
            "benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/"
            "excess_return_betrayal_hard_soft_decomposition.csv"
        ),
        help="Input decomposition CSV from the profit-linked betrayal analysis.",
    )
    p.add_argument(
        "--output-dir",
        default="benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20",
        help="Output directory for summary CSVs and figures.",
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=METHOD_ORDER,
        help="Method order in plots/tables.",
    )
    p.add_argument("--dpi", type=int, default=320)
    return p.parse_args()


def _ordered_methods(df: pd.DataFrame, methods: List[str]) -> List[str]:
    present = set(df["method"].astype(str).tolist())
    return [m for m in methods if m in present]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-12
    out["nonhard_rate"] = (out["any_rate"] - out["hard_rate"]).clip(lower=0.0)
    out["hard_share"] = np.where(out["any_rate"] > eps, out["hard_rate"] / out["any_rate"], 0.0)
    out["nonhard_share"] = 1.0 - out["hard_share"]
    out["nonhard_uplift_component"] = out["any_uplift"] - out["hard_uplift"]
    return out


def main() -> None:
    args = parse_args()
    in_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    df = _prepare(df)

    # 1) Hard-only summary table (if user wants to exclude soft from betrayal)
    hard_only = df[
        [
            "source",
            "method",
            "n_kols",
            "hard_rate",
            "hard_rate_pos",
            "hard_rate_nonpos",
            "hard_uplift",
            "rev_rate",
            "entry_rate",
        ]
    ].rename(
        columns={
            "hard_rate": "p_hard_betrayal",
            "hard_rate_pos": "p_hard_betrayal_given_profit_event",
            "hard_rate_nonpos": "p_hard_betrayal_given_nonprofit_event",
            "hard_uplift": "uplift_hard_only",
            "rev_rate": "p_reversal",
            "entry_rate": "p_unsupported_entry",
        }
    )
    hard_only.to_csv(out_dir / "hard_only_betrayal_summary.csv", index=False)

    # 2) Decomposition table
    decomp = df[
        [
            "source",
            "method",
            "n_kols",
            "hard_rate",
            "nonhard_rate",
            "hard_share",
            "nonhard_share",
            "hard_uplift",
            "nonhard_uplift_component",
            "any_uplift",
        ]
    ].copy()
    decomp.to_csv(out_dir / "hard_soft_betrayal_decomposition_summary.csv", index=False)

    # 3) Figure: top row = share composition, bottom row = uplift decomposition
    sources = [s for s in ["x", "youtube"] if s in set(df["source"].astype(str))]
    fig, axes = plt.subplots(2, len(sources), figsize=(10.0, 6.6), squeeze=False)

    for col, src in enumerate(sources):
        sdf = df[df["source"] == src].copy()
        methods = _ordered_methods(sdf, args.methods)
        sdf["method"] = pd.Categorical(sdf["method"], categories=methods, ordered=True)
        sdf = sdf.sort_values("method")
        x = np.arange(len(methods))

        # Top: 100% stacked share of betrayal forms (within betrayal_any)
        ax_top = axes[0, col]
        hard_share = sdf["hard_share"].to_numpy(dtype=float)
        nonhard_share = sdf["nonhard_share"].to_numpy(dtype=float)

        bars_h = ax_top.bar(
            x,
            hard_share,
            width=0.72,
            color=HARD_COLOR,
            alpha=0.88,
            label="Hard share (UER+DRR)",
        )
        bars_s = ax_top.bar(
            x,
            nonhard_share,
            width=0.72,
            bottom=hard_share,
            color=SOFT_COLOR,
            alpha=0.80,
            label="Non-hard share (mainly soft deviation)",
        )

        for i, m in enumerate(methods):
            if m == "KICL":
                bars_h[i].set_edgecolor("#111111")
                bars_h[i].set_linewidth(2.2)
                bars_s[i].set_edgecolor("#111111")
                bars_s[i].set_linewidth(2.2)
                ax_top.text(i, 1.03, "KICL", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

        ax_top.set_ylim(0.0, 1.08)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(methods, rotation=20, fontsize=8.5)
        ax_top.set_ylabel("Share within betrayal_any")
        ax_top.set_title("X" if src == "x" else "YouTube", fontsize=11.5)
        ax_top.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)

        # Bottom: uplift decomposition under profit-event condition
        ax_bot = axes[1, col]
        hard_uplift = sdf["hard_uplift"].to_numpy(dtype=float)
        nonhard_uplift = sdf["nonhard_uplift_component"].to_numpy(dtype=float)
        w = 0.34
        b1 = ax_bot.bar(
            x - w / 2,
            hard_uplift,
            width=w,
            color=HARD_COLOR,
            alpha=0.88,
            label="Hard uplift",
        )
        b2 = ax_bot.bar(
            x + w / 2,
            nonhard_uplift,
            width=w,
            color=SOFT_COLOR,
            alpha=0.82,
            label="Non-hard uplift component",
        )
        for i, m in enumerate(methods):
            if m == "KICL":
                b1[i].set_edgecolor("#111111")
                b1[i].set_linewidth(2.0)
                b2[i].set_edgecolor("#111111")
                b2[i].set_linewidth(2.0)

        ax_bot.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.8)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(methods, rotation=20, fontsize=8.5)
        ax_bot.set_ylabel("Uplift under profit events")
        ax_bot.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    handles.extend(h2)
    labels.extend(l2)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=8.8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Betrayal Decomposition: Hard vs Soft Components", y=0.995, fontsize=13.0)
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 0.97), h_pad=1.1, w_pad=0.95)

    fig.savefig(out_dir / "betrayal_hard_soft_decomposition_story.png", dpi=args.dpi)
    fig.savefig(out_dir / "betrayal_hard_soft_decomposition_story.pdf")
    plt.close(fig)

    # 4) short markdown note
    md = []
    md.append("# Hard vs Soft Betrayal Decomposition (Story Figure)")
    md.append("")
    md.append("- Hard betrayal: reversal + unsupported entry.")
    md.append("- Non-hard component: residual part of betrayal_any after removing hard part (mainly soft deviation effect).")
    md.append("- Profit-event condition: event_return > 0.")
    md.append("")
    md.append("Files:")
    md.append("- `hard_only_betrayal_summary.csv`")
    md.append("- `hard_soft_betrayal_decomposition_summary.csv`")
    md.append("- `betrayal_hard_soft_decomposition_story.png`")
    (out_dir / "README_hard_soft_decomposition.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Saved: {out_dir / 'hard_only_betrayal_summary.csv'}")
    print(f"Saved: {out_dir / 'hard_soft_betrayal_decomposition_summary.csv'}")
    print(f"Saved: {out_dir / 'betrayal_hard_soft_decomposition_story.png'}")
    print(f"Saved: {out_dir / 'betrayal_hard_soft_decomposition_story.pdf'}")
    print(f"Saved: {out_dir / 'README_hard_soft_decomposition.md'}")


if __name__ == "__main__":
    main()

