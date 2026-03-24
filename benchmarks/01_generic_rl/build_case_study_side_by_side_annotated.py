#!/usr/bin/env python3
"""Build a side-by-side Baseline vs KICL case-study figure with red-circle callouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd


DEFAULT_CALLOUT_WINDOWS = {
    ("youtube", "The_Maverick_of_Wall_Street"): ("2024-11-01", "2024-12-20"),
    ("x", "Jake__Wujastyk"): ("2024-11-15", "2024-12-25"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case-root",
        type=Path,
        default=Path("benchmarks/compare/case_study"),
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("benchmarks/compare/case_study/case_study_selected_kols_summary.csv"),
    )
    p.add_argument("--source", type=str, default="", help="Optional: single-case source (x or youtube).")
    p.add_argument("--kol", type=str, default="", help="Optional: single-case KOL name.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/compare/case_study/figures_baseline_vs_ours/case_study_side_by_side_annotated.png"),
    )
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("benchmarks/compare/case_study/figures_baseline_vs_ours/case_study_side_by_side_annotated.pdf"),
    )
    p.add_argument("--width", type=float, default=12.8, help="Figure width in inches.")
    p.add_argument("--height", type=float, default=4.5, help="Figure height in inches.")
    p.add_argument("--dpi", type=int, default=280)
    return p.parse_args()


def platform_label(source: str) -> str:
    return "YouTube" if source.lower() == "youtube" else "X"


def pretty_kol_name(kol: str) -> str:
    return kol.replace("_", " ")


def load_case_df(case_root: Path, source: str, kol: str) -> pd.DataFrame:
    p = case_root / "raw_kicl" / source / kol / "equity_daily.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing equity csv: {p}")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def draw_callout(ax: plt.Axes, df: pd.DataFrame, source: str, kol: str) -> None:
    key = (source, kol)
    if key not in DEFAULT_CALLOUT_WINDOWS:
        return

    s, e = DEFAULT_CALLOUT_WINDOWS[key]
    start = pd.Timestamp(s)
    end = pd.Timestamp(e)
    w = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if w.empty:
        return

    y_all = pd.concat([w["equity_baseline"], w["equity_trained"]], axis=0).dropna()
    if y_all.empty:
        return

    y_min, y_max = float(y_all.min()), float(y_all.max())
    y_center = (y_min + y_max) / 2.0
    y_height = max(0.04, (y_max - y_min) * 1.35)

    x_center = start + (end - start) / 2
    x_center_num = mdates.date2num(x_center)
    x_width_days = max(18, (end - start).days)

    ell = Ellipse(
        (x_center_num, y_center),
        width=x_width_days,
        height=y_height,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
        alpha=0.9,
        zorder=7,
    )
    ax.add_patch(ell)


def plot_one(
    ax: plt.Axes,
    df: pd.DataFrame,
    source: str,
    kol: str,
    show_legend: bool,
    show_ylabel: bool,
) -> None:
    ax.plot(
        df["date"],
        df["equity_baseline"],
        label="Baseline",
        color="#2E86DE",
        linewidth=1.2,
        linestyle="--",
        alpha=0.95,
    )
    ax.plot(
        df["date"],
        df["equity_trained"],
        label="KICL (Ours)",
        color="#f39c12",
        linewidth=1.6,
        alpha=0.98,
    )

    title = f"{platform_label(source)} \u00b7 {pretty_kol_name(kol)}"
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=10)
    if show_ylabel:
        ax.set_ylabel("Equity")
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.22, color="#bdbdbd")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.15, color="#d6d6d6")

    if show_legend:
        ax.legend(loc="upper left", fontsize=12, frameon=True)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for t in ax.get_xticklabels():
        t.set_fontsize(9)

    y = pd.concat([df["equity_baseline"], df["equity_trained"]], axis=0).dropna()
    if not y.empty:
        ymin, ymax = float(y.min()), float(y.max())
        pad = max(0.01, 0.05 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)

    draw_callout(ax, df, source, kol)


def main() -> None:
    args = parse_args()
    if args.source and args.kol:
        picked = pd.DataFrame([{"source": args.source, "kol": args.kol}])
    else:
        cases = pd.read_csv(args.summary_csv)[["source", "kol"]].drop_duplicates()
        if len(cases) < 2:
            raise RuntimeError("Need at least two case-study rows in summary csv.")
        # Keep one X and one YouTube if possible.
        x_case = cases[cases["source"].str.lower() == "x"].head(1)
        y_case = cases[cases["source"].str.lower() == "youtube"].head(1)
        if len(x_case) == 1 and len(y_case) == 1:
            picked = pd.concat([x_case, y_case], ignore_index=True)
        else:
            picked = cases.head(2).reset_index(drop=True)

    n = len(picked)
    fig, axes = plt.subplots(1, n, figsize=(args.width, args.height), sharey=False)
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (_, row) in enumerate(picked.iterrows()):
        source, kol = row["source"], row["kol"]
        df = load_case_df(args.case_root, source, kol)
        plot_one(
            axes[i],
            df,
            source,
            kol,
            show_legend=(n == 1 or i == 0),
            show_ylabel=(n == 1 or i == 0),
        )

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")
    print(f"Saved: {args.output_pdf}")


if __name__ == "__main__":
    main()
