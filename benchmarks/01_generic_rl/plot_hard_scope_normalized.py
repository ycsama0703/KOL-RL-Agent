"""Plot normalized hard-scope ablation on a unified scale.

Normalization:
- Min-max per metric across variants
- Direction-aligned so that higher is always better
  - cumulative_return: higher better
  - max_drawdown, unsupported_entry_rate, direction_reversal_rate, baseline_deviation: lower better

Input:
  ablation study/hard_scope_test_selected20/_summary_hard_scope_means.csv
Output:
  ablation study/hard_scope_test_selected20/hard_scope_normalized_parallel.{png,pdf}
  ablation study/hard_scope_test_selected20/hard_scope_normalized_values.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT_ORDER = ["hard_train_only", "hard_none", "hard_infer_only", "hard_both"]
VARIANT_LABEL = {
    "hard_train_only": "train-only",
    "hard_none": "none",
    "hard_infer_only": "infer-only",
    "hard_both": "train+infer",
}

METRICS = [
    ("cumulative_return", "Return", "high"),
    ("max_drawdown", "MDD", "low"),
    ("unsupported_entry_rate", "UER", "low"),
    ("direction_reversal_rate", "DRR", "low"),
    ("baseline_deviation", "BD", "low"),
]

COLORS = {
    "hard_train_only": "#D16262",
    "hard_none": "#C98A5A",
    "hard_infer_only": "#7A99C4",
    "hard_both": "#D99058",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalized hard-scope ablation plot.")
    p.add_argument(
        "--input-csv",
        default="ablation study/hard_scope_test_selected20/_summary_hard_scope_means.csv",
        help="Input summary csv.",
    )
    p.add_argument(
        "--output-prefix",
        default="ablation study/hard_scope_test_selected20/hard_scope_normalized_parallel",
        help="Output prefix without extension.",
    )
    p.add_argument("--dpi", type=int, default=320)
    return p.parse_args()


def _normalize_directional(values: np.ndarray, direction: str) -> np.ndarray:
    arr = values.astype(float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if np.isclose(lo, hi):
        out = np.full_like(arr, 1.0)
    else:
        out = (arr - lo) / (hi - lo)
    if direction == "low":
        out = 1.0 - out
    return out


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = df.set_index("variant").loc[VARIANT_ORDER].reset_index()

    # build normalized table
    norm_df = pd.DataFrame({"variant": df["variant"]})
    for col, short, direction in METRICS:
        vals = pd.to_numeric(df[col], errors="coerce").values
        norm_df[short] = _normalize_directional(vals, direction)

    norm_csv = out_prefix.parent / "hard_scope_normalized_values.csv"
    norm_df.to_csv(norm_csv, index=False)

    # plot: parallel coordinates on unified 0-1 scale
    x = np.arange(len(METRICS))
    x_labels = [m[1] for m in METRICS]

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.set_facecolor("#f3f3f3")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.30)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Normalized score (0-1, higher is better)", fontsize=14, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for v in VARIANT_ORDER:
        row = norm_df[norm_df["variant"] == v].iloc[0]
        y = np.array([float(row[m[1]]) for m in METRICS], dtype=float)
        lw = 3.0 if v == "hard_both" else 2.4
        alpha = 0.98 if v == "hard_both" else 0.88
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7.5,
            linewidth=lw,
            color=COLORS[v],
            alpha=alpha,
            label=VARIANT_LABEL[v],
        )
        # annotate near final point to reduce clutter
        ax.text(
            x[-1] + 0.05,
            y[-1],
            VARIANT_LABEL[v],
            fontsize=11.5,
            va="center",
            color=COLORS[v],
            fontweight="bold" if v == "hard_both" else "normal",
        )

    # Do not show redundant legend since labels are at line tails.
    ax.set_xlim(-0.1, x[-1] + 0.9)
    fig.subplots_adjust(left=0.10, right=0.94, top=0.95, bottom=0.14)

    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {norm_csv}")


if __name__ == "__main__":
    main()

