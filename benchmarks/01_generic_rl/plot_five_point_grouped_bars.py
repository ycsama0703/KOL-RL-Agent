"""Plot five-point ablation as grouped bars (multi-metric per variant).

Order (left -> right):
Baseline -> WO_RL_COMPLETION -> WO_REGIME_SPLIT -> WO_HARD -> KICL (Full)

Bars are normalized within the selected scope so mixed metrics are comparable:
- Return/Sharpe: higher is better
- UER/DRR/BD: lower is better (inverted in normalization)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["BASELINE", "WO_RL_COMPLETION", "WO_REGIME_SPLIT", "WO_HARD", "KICL"]
LABELS = {
    "BASELINE": "Baseline",
    "WO_RL_COMPLETION": "WO_RL_COMPLETION",
    "WO_REGIME_SPLIT": "WO_REGIME_SPLIT",
    "WO_HARD": "WO_HARD",
    "KICL": "KICL (Full)",
}

COLORS = ["#F39C12", "#3498DB", "#E74C3C", "#9B59B6", "#2ECC71"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot five-point grouped bars (multi-metric).")
    p.add_argument(
        "--input-csv",
        default="ablation study/five_point_compare/five_point_summary_by_source.csv",
        help="Input five-point summary csv.",
    )
    p.add_argument(
        "--output-prefix",
        default="ablation study/five_point_compare/five_point_grouped_bars",
        help="Output prefix (without extension).",
    )
    p.add_argument(
        "--scope",
        choices=["overall", "x", "youtube"],
        default="overall",
        help="Plot overall or one source only.",
    )
    p.add_argument(
        "--eval-scope",
        choices=["event", "daily"],
        default="event",
        help="Use event-level or daily-level performance columns for Return/Sharpe.",
    )
    p.add_argument("--dpi", type=int, default=260)
    return p.parse_args()


def _normalize(series: pd.Series, direction: str) -> pd.Series:
    x = series.astype(float).copy()
    lo, hi = x.min(), x.max()
    if np.isclose(hi, lo):
        out = pd.Series(np.ones(len(x)) * 0.5, index=x.index)
    else:
        out = (x - lo) / (hi - lo)
    if direction == "down":
        out = 1.0 - out
    return out


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if args.scope == "overall":
        sdf = df.groupby("method", as_index=False).mean(numeric_only=True)
        title = "Five-Point Ablation (Overall)"
    else:
        sdf = df[df["source"] == args.scope].copy()
        title = f"Five-Point Ablation ({args.scope.upper()})"

    sdf = sdf[sdf["method"].isin(ORDER)].copy()
    sdf["method"] = pd.Categorical(sdf["method"], categories=ORDER, ordered=True)
    sdf = sdf.sort_values("method")
    if len(sdf) != len(ORDER):
        missing = [m for m in ORDER if m not in set(sdf["method"].astype(str))]
        raise RuntimeError(f"Missing methods in scope={args.scope}: {missing}")

    perf_cols = {
        "event": ("event_return_mean", "event_sharpe_mean"),
        "daily": ("daily_return_mean", "daily_sharpe_mean"),
    }
    ret_col, sha_col = perf_cols[args.eval_scope]
    metrics = [
        (ret_col, "Return", "up"),
        (sha_col, "Sharpe", "up"),
        ("UER_mean", "UER", "down"),
        ("DRR_mean", "DRR", "down"),
        ("BD_mean", "BD", "down"),
    ]

    for col, _, direction in metrics:
        if col not in sdf.columns:
            raise RuntimeError(f"Missing column: {col}")
        sdf[f"{col}_norm"] = _normalize(sdf[col], direction)

    x = np.arange(len(sdf))
    n = len(metrics)
    width = 0.15
    offsets = (np.arange(n) - (n - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    for i, ((col, label, _), color) in enumerate(zip(metrics, COLORS)):
        vals = sdf[f"{col}_norm"].to_numpy()
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=width,
            color=color,
            edgecolor="#1f2937",
            linewidth=0.7,
            label=label,
            alpha=0.92,
            zorder=3,
        )
        # show raw value on top of bars
        raw = sdf[col].to_numpy()
        for b, rv in zip(bars, raw):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.015,
                f"{rv:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#374151",
                rotation=90,
            )

    # Highlight KICL group
    kicl_idx = ORDER.index("KICL")
    ax.axvspan(kicl_idx - 0.45, kicl_idx + 0.45, color="#fde68a", alpha=0.16, zorder=1)
    ax.text(kicl_idx, 1.08, "KICL (Full)", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#92400e")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in sdf["method"].astype(str)], rotation=10, ha="right")
    ax.set_ylim(0.0, 1.18)
    ax.set_ylabel("Normalized Score (higher is better)")
    ax.set_title(f"{title} • {args.eval_scope.capitalize()} metrics", fontsize=14, fontweight="semibold", pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.24, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper left",
        ncol=5,
        frameon=True,
        framealpha=0.92,
        fontsize=9,
        bbox_to_anchor=(0.0, 1.02),
    )

    fig.tight_layout()
    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
