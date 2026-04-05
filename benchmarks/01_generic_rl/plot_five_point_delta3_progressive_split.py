"""Plot progressive 2D ablation bars with split panels and independent y-axes.

Input:
  ablation study/five_point_compare/five_point_summary_overall.csv

Output:
  five_point_delta3_grouped_progressive_split_nosharedy.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


BASELINE = "BASELINE"
PROGRESSIVE = ["WO_RL_COMPLETION", "WO_REGIME_SPLIT", "KICL"]
OUTLIER = ["WO_HARD"]
ALL = [BASELINE] + PROGRESSIVE + OUTLIER

LABELS = {
    "WO_RL_COMPLETION": "WO-RL-C",
    "WO_REGIME_SPLIT": "WO-RS",
    "KICL": "KICL",
    "WO_HARD": "WO-H",
}

METRICS = [
    ("event_return_mean", "Δ Return"),
    ("event_sharpe_mean", "Δ Sharpe"),
    ("BD_mean", "Δ BD"),
]
HATCHES = ["", "///", "xxx"]
METRIC_EDGE = ["#1E3A8A", "#0F766E", "#7C2D12"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Progressive split bar chart with independent y-axes.")
    p.add_argument(
        "--input-csv",
        default="ablation study/five_point_compare/five_point_summary_overall.csv",
        help="Overall summary csv path.",
    )
    p.add_argument(
        "--output-prefix",
        default="ablation study/five_point_compare/five_point_delta3_grouped_progressive_split_nosharedy",
        help="Output prefix (without suffix).",
    )
    p.add_argument("--dpi", type=int, default=280)
    return p.parse_args()


def _ylim(vals: np.ndarray) -> tuple[float, float]:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return (-0.1, 0.1)
    lo, hi = float(v.min()), float(v.max())
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.25, 0.03)
        return lo - pad, hi + pad
    span = hi - lo
    return lo - 0.18 * span, hi + 0.22 * span


def _collect_delta(df: pd.DataFrame, method: str, metric: str) -> float:
    base = float(df.loc[df["method"] == BASELINE, metric].iloc[0])
    cur = float(df.loc[df["method"] == method, metric].iloc[0])
    return cur - base


def _to_percent(v: float) -> float:
    return 100.0 * float(v)


def _draw_grouped(
    ax: plt.Axes,
    methods: list[str],
    vals: dict[str, list[float]],
    method_color: dict[str, str],
    *,
    bar_width: float = 0.22,
    offset_scale: float = 1.0,
) -> np.ndarray:
    x = np.arange(len(methods), dtype=float)
    width = bar_width
    offsets = np.array([-width * offset_scale, 0.0, width * offset_scale], dtype=float)
    all_y = []
    for i, (_, metric_label) in enumerate(METRICS):
        y = np.array(vals[metric_label], dtype=float)
        all_y.append(y)
        bars = ax.bar(
            x + offsets[i],
            y,
            width=width,
            color=[method_color[m] for m in methods],
            edgecolor=METRIC_EDGE[i],
            linewidth=1.15,
            hatch=HATCHES[i],
            alpha=0.95,
        )
        for idx, (b, yy) in enumerate(zip(bars, y)):
            # Stagger labels by metric so they don't sit directly on top of bars.
            # left/right bars get horizontal nudges; center gets extra vertical gap.
            if i == 0:
                x_nudge = -0.28 * width
                yoff = 0.85
            elif i == 1:
                x_nudge = 0.0
                yoff = 1.10
            else:
                x_nudge = 0.28 * width
                yoff = 0.85

            # Fine-tune specific WO-H labels requested by user:
            # -10.9% (ΔReturn) move right; +3.1% (ΔBD) move left.
            if methods[idx] == "WO_HARD":
                if i == 0:
                    x_nudge += 0.06
                elif i == 2:
                    x_nudge -= 0.06

            # KICL positive labels: force an up-down-up stack to avoid overlap.
            if methods[idx] == "KICL" and yy >= 0:
                # User-requested layout: down-up-down for the three KICL bars
                # (ΔReturn, ΔSharpe, ΔBD).
                if i in (0, 2):
                    y_text = -0.58
                    va = "top"
                else:
                    # Move +1.7% a bit higher to avoid overlap.
                    y_text = yy + 1.30
                    va = "bottom"
                ax.text(
                    b.get_x() + b.get_width() / 2 + x_nudge,
                    y_text,
                    f"{yy:+.1f}%",
                    ha="center",
                    va=va,
                    fontsize=24.0,
                    color="#111827",
                    fontweight="bold",
                )
                continue

            # Keep large negative labels close to bars (especially WO-H).
            if yy < 0:
                yoff = min(1.20, max(0.45, 0.08 * abs(float(yy)) + 0.15))

            ax.text(
                b.get_x() + b.get_width() / 2 + x_nudge,
                yy + (yoff if yy >= 0 else -yoff),
                f"{yy:+.1f}%",
                ha="center",
                va="bottom" if yy >= 0 else "top",
                fontsize=24.0,
                color="#111827",
                fontweight="bold",
            )
    ax.axhline(0.0, color="#374151", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=21.0, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return np.concatenate(all_y)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    missing = [m for m in ALL if m not in set(df["method"].astype(str))]
    if missing:
        raise RuntimeError(f"Missing methods in input: {missing}")

    method_color = {
        "WO_RL_COMPLETION": "#8DAAC8",
        "WO_REGIME_SPLIT": "#6F92BD",
        "KICL": "#D99058",
        "WO_HARD": "#D16262",
    }

    left_vals = {label: [] for _, label in METRICS}
    right_vals = {label: [] for _, label in METRICS}

    for metric, label in METRICS:
        for m in PROGRESSIVE:
            left_vals[label].append(_to_percent(_collect_delta(df, m, metric)))
        for m in OUTLIER:
            right_vals[label].append(_to_percent(_collect_delta(df, m, metric)))

    fig = plt.figure(figsize=(14.0, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.9], wspace=0.18)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    y_l = _draw_grouped(ax_l, PROGRESSIVE, left_vals, method_color, bar_width=0.22, offset_scale=1.0)
    # Right panel has only one method group, so use a larger bar width to keep
    # visual thickness comparable to the left panel.
    y_r = _draw_grouped(ax_r, OUTLIER, right_vals, method_color, bar_width=0.20, offset_scale=1.8)

    ax_l.set_ylim(*_ylim(y_l))
    ax_r.set_ylim(*_ylim(y_r))
    ax_r.set_xlim(-0.56, 0.56)

    ax_l.set_ylabel("Delta vs baseline (%)", fontsize=24, fontweight="semibold")
    ax_l.tick_params(axis="y", labelsize=19.0)
    ax_r.tick_params(axis="y", labelsize=19.0)

    bar_legend = [
        Patch(facecolor="#e5e7eb", edgecolor=METRIC_EDGE[i], hatch=HATCHES[i], label=label, linewidth=1.7)
        for i, (_, label) in enumerate(METRICS)
    ]
    fig.legend(
        handles=bar_legend,
        loc="upper center",
        ncol=3,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.93),
        prop={"size": 24.0, "weight": "bold"},
        handlelength=2.6,
        handleheight=1.2,
        borderpad=0.55,
        labelspacing=0.65,
        columnspacing=1.8,
    )
    fig.subplots_adjust(top=0.78, left=0.07, right=0.98, bottom=0.13, wspace=0.20)

    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
