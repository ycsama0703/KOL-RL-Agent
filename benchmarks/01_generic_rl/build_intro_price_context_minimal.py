#!/usr/bin/env python3
"""Regenerate compact intro price-context plots with publication-friendly styling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import dates as mdates
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class PlotSpec:
    ticker: str
    statement_day: str
    stem: str


DEFAULT_SPECS = (
    PlotSpec(ticker="FXI", statement_day="2024-12-02", stem="fxi_node_window_minimal"),
    PlotSpec(ticker="TSLA", statement_day="2024-11-11", stem="tsla_node_window_minimal"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/compare/case_study/intro_price_context"),
    )
    p.add_argument(
        "--window-trading-days",
        type=int,
        default=7,
        help="Trading-day window on each side of statement day.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=240,
    )
    return p.parse_args()


def _extract_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            close = df[("Close", ticker)].copy()
        elif ("Adj Close", ticker) in df.columns:
            close = df[("Adj Close", ticker)].copy()
        else:
            close = df.xs("Close", axis=1, level=0).iloc[:, 0].copy()
    else:
        if "Close" in df.columns:
            close = df["Close"].copy()
        elif "Adj Close" in df.columns:
            close = df["Adj Close"].copy()
        else:
            raise ValueError(f"Missing Close/Adj Close for {ticker}.")

    close = close.dropna().sort_index()
    if close.empty:
        raise ValueError(f"No valid close prices for {ticker}.")
    return close


def _download_window(ticker: str, statement_day: pd.Timestamp, width: int) -> pd.Series:
    # Add margin for weekends/holidays before slicing exact trading-day window.
    start = (statement_day - pd.tseries.offsets.BDay(width + 8)).date().isoformat()
    end = (statement_day + pd.tseries.offsets.BDay(width + 9)).date().isoformat()
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    close = _extract_close(raw, ticker)

    # If statement day is non-trading, map to nearest previous trading day.
    mapped_day = close.index.asof(statement_day)
    if pd.isna(mapped_day):
        mapped_day = close.index[0]

    center_idx = int(close.index.get_loc(mapped_day))
    lo = max(0, center_idx - width)
    hi = min(len(close) - 1, center_idx + width)
    return close.iloc[lo : hi + 1]


def _plot_one(
    series: pd.Series,
    ticker: str,
    statement_day: pd.Timestamp,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    # Landscape panel: better for paper layouts than a tall square.
    fig, ax = plt.subplots(figsize=(6.8, 2.65))
    _draw_axis(ax=ax, series=series, ticker=ticker, statement_day=statement_day, show_xlabel=False)
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _draw_axis(
    ax: plt.Axes,
    series: pd.Series,
    ticker: str,
    statement_day: pd.Timestamp,
    show_xlabel: bool,
    mode: str = "all_solid",
) -> None:
    style = {
        "line_color": "#2F6FAE",
        "line_width": 2.1,
        "event_color": "#C44E52",
        "grid_color": "#DDE3EB",
        "spine_color": "#AAB2BD",
        "deemph_color": "#9AA3AF",
    }

    mapped_day = series.index.asof(statement_day)
    if pd.isna(mapped_day):
        mapped_day = series.index[len(series) // 2]
    event_price = float(series.loc[mapped_day])
    y = series / event_price

    ax.set_facecolor("white")

    # Segment styling:
    # - all_solid: full line in primary color
    # - future_dashed: post-signal segment is gray dashed (future)
    # - history_dashed: pre-signal segment is gray dashed (history)
    pre_mask = y.index <= mapped_day
    post_mask = y.index >= mapped_day

    if mode == "future_dashed":
        ax.plot(
            y.index[pre_mask],
            y.values[pre_mask],
            color=style["line_color"],
            lw=style["line_width"],
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            y.index[post_mask],
            y.values[post_mask],
            color=style["deemph_color"],
            lw=style["line_width"] - 0.1,
            ls=(0, (4, 2)),
            alpha=0.95,
            zorder=3,
        )
    elif mode == "history_dashed":
        ax.plot(
            y.index[pre_mask],
            y.values[pre_mask],
            color=style["deemph_color"],
            lw=style["line_width"] - 0.1,
            ls=(0, (4, 2)),
            alpha=0.95,
            zorder=3,
        )
        ax.plot(
            y.index[post_mask],
            y.values[post_mask],
            color=style["line_color"],
            lw=style["line_width"],
            solid_capstyle="round",
            zorder=3,
        )
    else:
        ax.plot(y.index, y.values, color=style["line_color"], lw=style["line_width"], solid_capstyle="round", zorder=3)
    # Highlight signal day with both a thin band and a dashed line.
    ax.axvspan(
        mapped_day - pd.Timedelta(hours=12),
        mapped_day + pd.Timedelta(hours=12),
        color=style["event_color"],
        alpha=0.12,
        zorder=1,
    )
    ax.axvline(mapped_day, color=style["event_color"], lw=1.9, ls=(0, (4, 2)), alpha=0.85, zorder=2)
    ax.scatter([mapped_day], [1.0], s=34, color=style["event_color"], zorder=5)

    ax.set_title(ticker, fontsize=16, pad=6, fontweight="semibold")
    ax.set_xlabel("Date" if show_xlabel else "")
    ax.set_ylabel("")
    # Keep locators for visible helper grid, hide numeric labels.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, length=0)

    ax.grid(True, color=style["grid_color"], linestyle="--", linewidth=0.9, alpha=0.72)
    for spine in ax.spines.values():
        spine.set_color(style["spine_color"])
        spine.set_linewidth(0.9)
    ax.margins(x=0.03, y=0.14)


def _plot_stacked(
    data: list[tuple[pd.Series, str, pd.Timestamp]],
    out_png: Path,
    out_pdf: Path,
    dpi: int,
    mode: str = "all_solid",
) -> None:
    # Square canvas so two flat panels stack into a compact figure.
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.8, 4.75), sharex=False)
    fig.patch.set_facecolor("white")

    for i, (series, ticker, day) in enumerate(data):
        _draw_axis(
            ax=axes[i],
            series=series,
            ticker=ticker,
            statement_day=day,
            show_xlabel=False,
            mode=mode,
        )

    plt.subplots_adjust(hspace=0.16)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stacked_payload: list[tuple[pd.Series, str, pd.Timestamp]] = []

    for spec in DEFAULT_SPECS:
        statement_day = pd.Timestamp(spec.statement_day)
        series = _download_window(spec.ticker, statement_day, args.window_trading_days)
        png = args.output_dir / f"{spec.stem}.png"
        pdf = args.output_dir / f"{spec.stem}.pdf"
        _plot_one(series, spec.ticker, statement_day, png, pdf, args.dpi)
        stacked_payload.append((series, spec.ticker, statement_day))
        print(f"Saved: {png}")
        print(f"Saved: {pdf}")

    stacked_png = args.output_dir / "fxi_tsla_node_windows_minimal_stacked.png"
    stacked_pdf = args.output_dir / "fxi_tsla_node_windows_minimal_stacked.pdf"
    _plot_stacked(stacked_payload, stacked_png, stacked_pdf, args.dpi)
    print(f"Saved: {stacked_png}")
    print(f"Saved: {stacked_pdf}")

    # Variant A: future segment (post-signal) is gray dashed.
    future_png = args.output_dir / "fxi_tsla_node_windows_minimal_stacked_future_dashed.png"
    future_pdf = args.output_dir / "fxi_tsla_node_windows_minimal_stacked_future_dashed.pdf"
    _plot_stacked(stacked_payload, future_png, future_pdf, args.dpi, mode="future_dashed")
    print(f"Saved: {future_png}")
    print(f"Saved: {future_pdf}")

    # Variant B: history segment (pre-signal) is gray dashed.
    history_png = args.output_dir / "fxi_tsla_node_windows_minimal_stacked_history_dashed.png"
    history_pdf = args.output_dir / "fxi_tsla_node_windows_minimal_stacked_history_dashed.pdf"
    _plot_stacked(stacked_payload, history_png, history_pdf, args.dpi, mode="history_dashed")
    print(f"Saved: {history_png}")
    print(f"Saved: {history_pdf}")


if __name__ == "__main__":
    main()
