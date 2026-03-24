"""Plot 5-point overall figures.

Input:
  ablation study/five_point_compare/five_point_summary_overall.csv

Outputs:
  - five_point_overall_event_2x2.{png,pdf}
  - five_point_overall_event_performance.{png,pdf}
  - five_point_overall_event_betrayal_nonzero.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["BASELINE", "WO_RL_COMPLETION", "WO_REGIME_SPLIT", "WO_HARD", "KICL"]
LABELS = {
    "BASELINE": "BL",
    "WO_RL_COMPLETION": "WO-RC",
    "WO_REGIME_SPLIT": "WO-RS",
    "KICL": "FULL",
    "WO_HARD": "WO-H",
}

# Style palette aligned with the hard-betrayal figure style.
BLUE_MAIN = "#5E7EB5"
ORANGE_DARK = "#C8743F"
AX_FACE = "#f1f1f1"
GRID_COLOR = "#b9b9b9"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot split overall 5-point figures.")
    p.add_argument(
        "--input-csv",
        default="ablation study/five_point_compare/five_point_summary_overall.csv",
        help="Input overall summary csv.",
    )
    p.add_argument(
        "--output-dir",
        default="ablation study/five_point_compare",
        help="Output directory.",
    )
    p.add_argument("--dpi", type=int, default=260)
    return p.parse_args()


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["method"].isin(ORDER)].copy()
    out["method"] = pd.Categorical(out["method"], categories=ORDER, ordered=True)
    out = out.sort_values("method").reset_index(drop=True)
    miss = [m for m in ORDER if m not in set(out["method"].astype(str))]
    if miss:
        raise RuntimeError(f"Missing methods in input: {miss}")
    return out


def _method_colors(methods: list[str]) -> list[str]:
    colors = []
    for m in methods:
        if m == "KICL":
            colors.append(ORANGE_DARK)
        else:
            colors.append(BLUE_MAIN)
    return colors


def _tight_ylim(vals: np.ndarray, *, lower_is_better: bool = False) -> tuple[float, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    lo = float(v.min())
    hi = float(v.max())
    if np.isclose(lo, hi):
        pad = max(0.02, abs(lo) * 0.15)
        return (lo - pad, hi + pad)
    span = hi - lo
    # keep a little more headroom above labels
    low_pad = 0.22 * span
    high_pad = 0.30 * span
    y0 = lo - low_pad
    y1 = hi + high_pad
    # For "lower is better" metrics, keep visible margin below the best bar.
    if lower_is_better:
        y0 = max(0.0, y0)
    return (y0, y1)


def _hide_y_ticks(ax: plt.Axes) -> None:
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0)


def _low_outlier_floor(vals: np.ndarray) -> float | None:
    """Return a zoom floor when there is a clear low outlier; otherwise None."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 4:
        return None
    s = np.sort(v)
    low = float(s[0])
    second = float(s[1])
    top = float(s[-1])
    core_span = max(1e-12, top - second)
    low_gap = second - low
    # strong low outlier
    if low_gap > 0.55 * core_span:
        return second - 0.20 * core_span
    return None


def plot_performance(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    metrics = [
        ("event_return_mean", "Return ↑"),
        ("event_sharpe_mean", "Sharpe ↑"),
        ("event_mdd_mean", "MDD ↓"),
    ]
    methods = df["method"].astype(str).tolist()
    x = np.arange(len(methods))
    colors = _method_colors(methods)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))
    for ax, (col, title) in zip(axes, metrics):
        ax.set_facecolor(AX_FACE)
        vals = df[col].astype(float).to_numpy()
        bars = ax.bar(x, vals, color=colors, edgecolor="#1f2937", linewidth=0.8, alpha=0.93)
        for i, (b, v) in enumerate(zip(bars, vals)):
            yoff = 0.015 if v >= 0 else -0.02
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + yoff,
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9.6,
                color="#111827",
                fontweight="bold" if methods[i] == "KICL" else "normal",
            )
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.grid(axis="y", linestyle="--", linewidth=0.85, color=GRID_COLOR, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha="right", fontsize=7.5)
        ax.set_ylim(*_tight_ylim(vals, lower_is_better=("MDD" in title)))
        _hide_y_ticks(ax)

    fig.tight_layout()
    png = out_dir / "five_point_overall_event_performance.png"
    pdf = out_dir / "five_point_overall_event_performance.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_betrayal_nonzero(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    metrics = [
        ("HVC_mean", "Hard Betrayal (UER+DRR, non-zero only)"),
        ("BD_mean", "BD (non-zero only)"),
        ("event_mdd_mean", "MDD"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))
    eps = 0.0

    for ax, (col, title) in zip(axes, metrics):
        ax.set_facecolor(AX_FACE)
        if col == "event_mdd_mean":
            sdf = df.copy()
        else:
            sdf = df[df[col].astype(float).abs() > eps].copy()
        if sdf.empty:
            ax.text(0.5, 0.5, "All zero", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_axis_off()
            continue

        methods = sdf["method"].astype(str).tolist()
        vals = sdf[col].astype(float).to_numpy()
        x = np.arange(len(methods))
        colors = _method_colors(methods)
        bars = ax.bar(x, vals, color=colors, edgecolor="#1f2937", linewidth=0.8, alpha=0.95)
        for i, (b, v) in enumerate(zip(bars, vals)):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.006,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.6,
                color="#111827",
                fontweight="bold" if methods[i] == "KICL" else "normal",
            )
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.grid(axis="y", linestyle="--", linewidth=0.85, color=GRID_COLOR, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha="right", fontsize=7.5)
        ax.set_ylim(*_tight_ylim(vals, lower_is_better=("MDD" in title)))
        _hide_y_ticks(ax)

    fig.tight_layout()
    png = out_dir / "five_point_overall_event_betrayal_nonzero.png"
    pdf = out_dir / "five_point_overall_event_betrayal_nonzero.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_compact_2x2(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """Single 2x2 panel figure for paper body.

    Panels:
    - Return ↑
    - Sharpe ↑
    - MDD ↓
    - BD + HVC (UER+DRR) ↓
    """
    main_metrics = [
        ("event_return_mean", "Return ↑"),
        ("event_sharpe_mean", "Sharpe ↑"),
        ("event_mdd_mean", "MDD ↓"),
    ]
    methods = df["method"].astype(str).tolist()
    x = np.arange(len(methods))
    colors = _method_colors(methods)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.6))
    axes = axes.ravel()
    # First 3 panels: single-metric bars
    for ax, (col, title) in zip(axes[:3], main_metrics):
        ax.set_facecolor(AX_FACE)
        vals = df[col].astype(float).to_numpy()
        floor = _low_outlier_floor(vals)
        clipped = np.zeros_like(vals, dtype=bool)
        vals_plot = vals.copy()
        if floor is not None:
            clipped = vals < floor
            vals_plot = np.where(clipped, floor, vals)
        bars = ax.bar(x, vals_plot, color=colors, edgecolor="#1f2937", linewidth=0.8, alpha=0.93)
        for i, (b, v) in enumerate(zip(bars, vals)):
            if clipped[i]:
                # clipped low outlier: show explicit down marker/value
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    vals_plot[i] + 0.004,
                    f"↓{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.4,
                    color="#7f1d1d",
                    fontweight="bold",
                )
                continue
            yoff = 0.012 if v >= 0 else -0.015
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + yoff,
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9.4,
                color="#111827",
                fontweight="bold" if methods[i] == "KICL" else "normal",
            )
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.grid(axis="y", linestyle="--", linewidth=0.85, color=GRID_COLOR, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha="right", fontsize=7.5)
        ax.set_ylim(*_tight_ylim(vals_plot, lower_is_better=("↓" in title)))
        if floor is not None:
            ax.text(
                0.03,
                0.96,
                "zoomed (low outlier clipped)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.1,
                color="#6b7280",
            )
        _hide_y_ticks(ax)

    # Panel 4: BD + HVC grouped bars
    ax = axes[3]
    ax.set_facecolor(AX_FACE)
    vals_bd = df["BD_mean"].astype(float).to_numpy()
    vals_hvc = df["HVC_mean"].astype(float).to_numpy()
    w = 0.36
    bd_bars = ax.bar(
        x - w / 2,
        vals_bd,
        width=w,
        color="#5E7EB5",
        edgecolor="#1f2937",
        linewidth=0.75,
        alpha=0.95,
        label="BD",
    )
    hvc_bars = ax.bar(
        x + w / 2,
        vals_hvc,
        width=w,
        color="#C8743F",
        edgecolor="#1f2937",
        linewidth=0.75,
        alpha=0.95,
        label="HVC",
    )
    for i, (b, v) in enumerate(zip(bd_bars, vals_bd)):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.006,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color="#111827",
            fontweight="bold" if methods[i] == "KICL" else "normal",
        )
    for i, (b, v) in enumerate(zip(hvc_bars, vals_hvc)):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.006,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color="#111827",
            fontweight="bold" if methods[i] == "KICL" else "normal",
        )
    ax.set_title("BD / HVC ↓", fontsize=12, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", linewidth=0.85, color=GRID_COLOR, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha="right", fontsize=7.5)
    ax.set_ylim(*_tight_ylim(np.concatenate([vals_bd, vals_hvc]), lower_is_better=True))
    _hide_y_ticks(ax)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)

    fig.tight_layout()
    png = out_dir / "five_point_overall_event_2x2.png"
    pdf = out_dir / "five_point_overall_event_2x2.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = _ordered(df)

    plot_compact_2x2(df=df, out_dir=out_dir, dpi=args.dpi)
    plot_performance(df=df, out_dir=out_dir, dpi=args.dpi)
    plot_betrayal_nonzero(df=df, out_dir=out_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
