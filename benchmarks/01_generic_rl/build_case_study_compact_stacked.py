#!/usr/bin/env python3
"""Build a compact stacked case-study figure.

Layout:
- 2 rows x 1 col (stacked)
- short title inside top-left corner of each subplot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class CaseSpec:
    title: str
    root: Path
    comp_label: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--top-root",
        type=Path,
        default=Path("benchmarks/compare/case_study/focused_case_single/baseline/x/Jake__Wujastyk"),
    )
    p.add_argument("--top-title", type=str, default="X · Jake W.")
    p.add_argument("--top-comp-label", type=str, default="Baseline")
    p.add_argument(
        "--bottom-root",
        type=Path,
        default=Path(
            "benchmarks/compare/case_study/focused_case_single/variant/youtube/The_Maverick_of_Wall_Street"
        ),
    )
    p.add_argument("--bottom-title", type=str, default="YouTube · Maverick WS")
    p.add_argument("--bottom-comp-label", type=str, default="WO_HARD")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/compare/case_study/focused_case_single/combined"),
    )
    p.add_argument("--output-name", type=str, default="case_compact_stacked")
    p.add_argument("--dpi", type=int, default=320)
    p.add_argument("--fig-width", type=float, default=11.2)
    p.add_argument("--fig-height", type=float, default=4.9)
    p.add_argument("--xmin", type=str, default="", help="Optional date lower bound YYYY-MM-DD.")
    p.add_argument("--xmax", type=str, default="", help="Optional date upper bound YYYY-MM-DD.")
    p.add_argument("--circle-width-days", type=float, default=24.0)
    p.add_argument("--max-bottom-nodes", type=int, default=2)
    return p.parse_args()


def load_case(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.read_csv(root / "case_single_timeseries.csv")
    nodes = pd.read_csv(root / "case_single_nodes.csv")
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    ts = ts.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return ts, nodes


def draw_case(
    ax: plt.Axes,
    ts: pd.DataFrame,
    nodes: pd.DataFrame,
    title: str,
    comp_label: str,
    circle_width_days: float,
    hide_xlabel: bool = False,
) -> None:
    ax.plot(ts["date"], ts["comp"], color="#2E86DE", lw=1.2, ls="--", label=comp_label)
    ax.plot(ts["date"], ts["ours"], color="#f39c12", lw=1.6, label="KICL (Ours)")
    ax.set_facecolor("white")
    ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.22, color="#bdbdbd")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.35, alpha=0.16, color="#d6d6d6")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    if hide_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("")
    ax.text(
        0.01,
        0.96,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.72),
        zorder=9,
    )

    y_all = pd.concat([ts["ours"], ts["comp"]], axis=0).dropna()
    ymin, ymax = float(y_all.min()), float(y_all.max())
    ypad = max(0.01, 0.05 * (ymax - ymin))
    ax.set_ylim(ymin - ypad, ymax + ypad)
    y_span = max(0.03, 0.30 * (ymax - ymin))

    for _, r in nodes.iterrows():
        d = pd.to_datetime(r["focus_day"], errors="coerce")
        if pd.isna(d):
            continue
        rr = ts[ts["date"] == d]
        if rr.empty:
            idx = (ts["date"] - d).abs().idxmin()
            rr = ts.loc[[idx]]
            d = pd.to_datetime(rr["date"].iloc[0])
        y = float(rr["ours"].iloc[0])
        e = Ellipse(
            (mdates.date2num(d), y),
            width=circle_width_days,
            height=y_span,
            fill=False,
            edgecolor="red",
            linewidth=1.6,
            alpha=0.9,
            zorder=6,
        )
        ax.add_patch(e)
        ax.text(
            d,
            y + y_span * 0.53,
            f"#{int(r['node_id'])}",
            color="red",
            fontsize=9,
            ha="center",
            fontweight="bold",
            zorder=7,
        )


def main() -> None:
    args = parse_args()

    top = CaseSpec(args.top_title, args.top_root, args.top_comp_label)
    bottom = CaseSpec(args.bottom_title, args.bottom_root, args.bottom_comp_label)

    ts_top, nodes_top = load_case(top.root)
    ts_bottom, nodes_bottom = load_case(bottom.root)
    if args.max_bottom_nodes > 0:
        nodes_bottom = nodes_bottom.head(args.max_bottom_nodes).copy()

    fig = plt.figure(figsize=(args.fig_width, args.fig_height), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.16)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    draw_case(
        ax1,
        ts_top,
        nodes_top,
        top.title,
        top.comp_label,
        args.circle_width_days,
        hide_xlabel=True,
    )
    draw_case(
        ax2,
        ts_bottom,
        nodes_bottom,
        bottom.title,
        bottom.comp_label,
        args.circle_width_days,
        hide_xlabel=False,
    )
    xmin = pd.to_datetime(args.xmin, errors="coerce") if args.xmin else pd.NaT
    xmax = pd.to_datetime(args.xmax, errors="coerce") if args.xmax else pd.NaT
    if pd.notna(xmin) or pd.notna(xmax):
        ax1.set_xlim(left=None if pd.isna(xmin) else xmin, right=None if pd.isna(xmax) else xmax)
        ax2.set_xlim(left=None if pd.isna(xmin) else xmin, right=None if pd.isna(xmax) else xmax)
    ax1.set_ylabel("")
    ax2.set_ylabel("")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        fontsize=10,
        frameon=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.output_dir / f"{args.output_name}.png"
    out_pdf = args.output_dir / f"{args.output_name}.pdf"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
