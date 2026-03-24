#!/usr/bin/env python3
"""Plot case-study equity curves with only Baseline vs KICL."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case-root",
        type=Path,
        default=Path("benchmarks/compare/case_study"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/compare/case_study/figures_baseline_vs_ours"),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=260,
    )
    p.add_argument(
        "--width",
        type=float,
        default=7.6,
        help="Figure width in inches.",
    )
    p.add_argument(
        "--height",
        type=float,
        default=3.2,
        help="Figure height in inches.",
    )
    return p.parse_args()


def plot_one(case_root: Path, out_dir: Path, source: str, kol: str, dpi: int, w: float, h: float) -> None:
    csv_path = case_root / "raw_kicl" / source / kol / "equity_daily.csv"
    if not csv_path.exists():
        print(f"Skip missing: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    platform_label = "YouTube" if source.lower() == "youtube" else "X"

    ax.plot(
        df["date"],
        df["equity_baseline"],
        label="Baseline",
        color="#2E86DE",
        linewidth=1.15,
        linestyle="--",
        alpha=0.95,
    )
    ax.plot(
        df["date"],
        df["equity_trained"],
        label="KICL (Ours)",
        color="#f39c12",
        linewidth=1.55,
        alpha=0.98,
    )

    ax.set_title(f"{platform_label}-{kol}", fontsize=12, pad=6)
    ax.set_ylabel("Equity")
    ax.set_xlabel("")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.22, color="#bdbdbd")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.15, color="#d6d6d6")
    ax.legend(loc="best", fontsize=9, frameon=True)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for t in ax.get_xticklabels():
        t.set_fontsize(8.5)
        t.set_rotation(0)

    # Tight y-range for better visual readability.
    y = pd.concat([df["equity_baseline"], df["equity_trained"]], axis=0).dropna()
    if not y.empty:
        ymin, ymax = float(y.min()), float(y.max())
        pad = max(0.01, 0.045 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)

    target = out_dir / source / kol
    target.mkdir(parents=True, exist_ok=True)
    png = target / "equity_baseline_vs_kicl.png"
    pdf = target / "equity_baseline_vs_kicl.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main() -> None:
    args = parse_args()
    cases_csv = args.case_root / "case_study_selected_kols_summary.csv"
    if not cases_csv.exists():
        raise FileNotFoundError(cases_csv)
    cases = pd.read_csv(cases_csv)[["source", "kol"]]

    for _, r in cases.iterrows():
        plot_one(
            case_root=args.case_root,
            out_dir=args.output_dir,
            source=r["source"],
            kol=r["kol"],
            dpi=args.dpi,
            w=args.width,
            h=args.height,
        )


if __name__ == "__main__":
    main()
