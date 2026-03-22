#!/usr/bin/env python3
"""Plot main trade-off figure for selected-20 benchmark results.

Input:
- benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed.csv

Output:
- benchmarks/compare/figures/selected20_main_tradeoff_event_vs_intent.png
- benchmarks/compare/figures/selected20_main_tradeoff_event_vs_intent.pdf
- benchmarks/compare/tables/selected20_main_tradeoff_points.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


METHOD_ORDER = ["KICL", "AWAC", "IQL", "BC", "CQL", "TD3BC"]
METHOD_COLORS = {
    "KICL": "#F39C12",
    "AWAC": "#17BECF",
    "IQL": "#2CA02C",
    "BC": "#8C564B",
    "CQL": "#D62728",
    "TD3BC": "#9467BD",
}

METHOD_POINT_MARKERS = {
    "KICL": "X",
    "AWAC": "D",
    "IQL": "^",
    "BC": "o",
    "CQL": "v",
    "TD3BC": "s",
}

METHOD_MEAN_MARKERS = {
    "KICL": "X",
    "AWAC": "D",
    "IQL": "^",
    "BC": "o",
    "CQL": "v",
    "TD3BC": "s",
}

LABEL_OFFSETS = {
    "KICL": (6, 6),
    "AWAC": (6, 4),
    "IQL": (6, 2),
    "BC": (6, -8),
    "CQL": (6, 5),
    "TD3BC": (6, -6),
}

# Optional source-specific label offsets to avoid overlap in dense panels.
SOURCE_LABEL_OFFSETS = {
    "youtube": {
        "AWAC": (8, 12),
        "IQL": (8, -10),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        default="benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed.csv",
        help="Detailed selected20 results CSV.",
    )
    p.add_argument(
        "--output-prefix",
        default="benchmarks/compare/figures/selected20_main_tradeoff_event_vs_intent",
        help="Output figure prefix (without extension).",
    )
    p.add_argument(
        "--output-points-csv",
        default="benchmarks/compare/tables/selected20_main_tradeoff_points.csv",
        help="Where to save method-level points used in the figure.",
    )
    p.add_argument(
        "--return-metric",
        choices=["event_cumulative_return", "daily_trained_cumulative_return"],
        default="event_cumulative_return",
        help="Y-axis metric.",
    )
    p.add_argument(
        "--intent-metric",
        choices=["intent_consistency_bd", "sign_agreement_rate", "baseline_policy_corr"],
        default="intent_consistency_bd",
        help="X-axis metric.",
    )
    p.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include BASELINE points if present in input.",
    )
    p.add_argument(
        "--show-errorbars",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show std error bars on method means.",
    )
    p.add_argument(
        "--show-kol-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show faint per-KOL points as background texture.",
    )
    p.add_argument(
        "--kol-point-size",
        type=float,
        default=14.0,
        help="Marker size for per-KOL points.",
    )
    p.add_argument(
        "--kol-point-alpha",
        type=float,
        default=0.12,
        help="Alpha for per-KOL points.",
    )
    p.add_argument(
        "--kol-point-jitter-x",
        type=float,
        default=0.0012,
        help="Deterministic display jitter for per-KOL x-points (0 disables).",
    )
    p.add_argument(
        "--kol-point-jitter-y",
        type=float,
        default=0.0018,
        help="Deterministic display jitter for per-KOL y-points (0 disables).",
    )
    p.add_argument(
        "--show-pareto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw Pareto frontier over method mean points in each panel.",
    )
    p.add_argument(
        "--pareto-color",
        default="#303030",
        help="Pareto frontier line color.",
    )
    p.add_argument(
        "--pareto-linestyle",
        default="--",
        help="Pareto frontier line style.",
    )
    p.add_argument(
        "--pareto-linewidth",
        type=float,
        default=1.15,
        help="Pareto frontier line width.",
    )
    p.add_argument(
        "--show-desirable-region",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shade top-right desirable trade-off region in each panel.",
    )
    p.add_argument(
        "--desirable-quantile",
        type=float,
        default=0.70,
        help="Quantile of method means used to define desirable region thresholds.",
    )
    p.add_argument(
        "--desirable-color",
        default="#DFF3E3",
        help="Fill color for desirable region.",
    )
    p.add_argument(
        "--desirable-alpha",
        type=float,
        default=0.42,
        help="Alpha for desirable region fill.",
    )
    p.add_argument(
        "--show-better-arrow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Annotate top-right direction as better trade-off.",
    )
    p.add_argument(
        "--shared-x-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one centered x-axis label for all panels.",
    )
    p.add_argument(
        "--shared-y-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one y-axis label on the left panel only.",
    )
    p.add_argument(
        "--hide-y-tick-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide numeric y-axis tick labels (relative position focus).",
    )
    p.add_argument(
        "--hide-y-ticks-on-right",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide y ticks entirely on non-left panels when shared y-label is used.",
    )
    p.add_argument(
        "--minimal-x-ticks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use minimal x-axis ticks (xmin and 1.0) for cleaner paper style.",
    )
    p.add_argument(
        "--label-mode",
        choices=["none", "kicl", "all"],
        default="all",
        help="Which points to annotate with method names.",
    )
    p.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show global method legend at bottom.",
    )
    p.add_argument(
        "--show-suptitle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show global figure title at top.",
    )
    p.add_argument(
        "--focus-means",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-zoom each panel around method mean points for readability.",
    )
    p.add_argument(
        "--focus-pad-ratio",
        type=float,
        default=0.20,
        help="Padding ratio used when --focus-means is enabled.",
    )
    p.add_argument(
        "--fig-width",
        type=float,
        default=7.2,
        help="Figure width in inches.",
    )
    p.add_argument(
        "--fig-height",
        type=float,
        default=3.4,
        help="Figure height in inches.",
    )
    p.add_argument("--title-fontsize", type=float, default=10.5)
    p.add_argument("--label-fontsize", type=float, default=9.0)
    p.add_argument("--tick-fontsize", type=float, default=8.0)
    p.add_argument("--anno-fontsize", type=float, default=8.0)
    p.add_argument("--dpi", type=int, default=320)
    return p.parse_args()


def _build_points(df: pd.DataFrame, return_metric: str, intent_metric: str) -> pd.DataFrame:
    d = df.copy()
    if intent_metric == "intent_consistency_bd":
        d["intent_consistency"] = 1.0 / (1.0 + d["mean_abs_deviation"].astype(float))
    elif intent_metric == "sign_agreement_rate":
        d["intent_consistency"] = d["sign_agreement_rate"].astype(float)
    else:
        d["intent_consistency"] = d["baseline_policy_corr"].astype(float)

    d["ret"] = d[return_metric].astype(float)
    out = (
        d.groupby(["source", "method"], as_index=False)
        .agg(
            intent_mean=("intent_consistency", "mean"),
            intent_std=("intent_consistency", "std"),
            ret_mean=("ret", "mean"),
            ret_std=("ret", "std"),
            n_kols=("kol", "nunique"),
        )
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["intent_mean", "ret_mean"])
    return out


def _build_detailed_points(df: pd.DataFrame, return_metric: str, intent_metric: str) -> pd.DataFrame:
    d = df.copy()
    if intent_metric == "intent_consistency_bd":
        d["intent_consistency"] = 1.0 / (1.0 + d["mean_abs_deviation"].astype(float))
    elif intent_metric == "sign_agreement_rate":
        d["intent_consistency"] = d["sign_agreement_rate"].astype(float)
    else:
        d["intent_consistency"] = d["baseline_policy_corr"].astype(float)
    d["ret"] = d[return_metric].astype(float)
    keep = ["source", "method", "kol", "intent_consistency", "ret"]
    d = d[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=["intent_consistency", "ret"])
    return d


def _deterministic_jitter(n: int, scale: float) -> np.ndarray:
    if n <= 1 or scale <= 0:
        return np.zeros(n, dtype=float)
    # Reproducible spread that separates overlaps without changing semantics.
    return np.linspace(-scale, scale, n, dtype=float)


def _label_for_intent(mode: str) -> str:
    if mode == "intent_consistency_bd":
        return "Intent Consistency = 1 / (1 + mean_abs_deviation) (higher is better)"
    if mode == "sign_agreement_rate":
        return "Sign Agreement Rate (higher is better)"
    return "Baseline-Policy Correlation (higher is better)"


def _sort_methods(methods: list[str]) -> list[str]:
    present = set(methods)
    ordered = [m for m in METHOD_ORDER if m in present]
    extras = sorted(m for m in present if m not in METHOD_ORDER)
    return ordered + extras


def _pareto_frontier(points: pd.DataFrame) -> pd.DataFrame:
    """Max-max Pareto frontier on (intent_mean, ret_mean)."""
    if points.empty:
        return points.copy()

    arr = points[["intent_mean", "ret_mean"]].to_numpy(dtype=float)
    keep = np.ones(len(arr), dtype=bool)
    for i in range(len(arr)):
        if not keep[i]:
            continue
        xi, yi = arr[i]
        dominated = (
            (arr[:, 0] >= xi)
            & (arr[:, 1] >= yi)
            & ((arr[:, 0] > xi) | (arr[:, 1] > yi))
        )
        dominated[i] = False
        if dominated.any():
            keep[i] = False

    frontier = points.loc[keep, ["method", "intent_mean", "ret_mean"]].copy()
    frontier = frontier.sort_values("intent_mean", kind="mergesort").reset_index(drop=True)
    return frontier


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    needed = {
        "source",
        "method",
        "kol",
        "mean_abs_deviation",
        "sign_agreement_rate",
        "baseline_policy_corr",
        args.return_metric,
    }
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in input CSV: {sorted(miss)}")

    points = _build_points(df, args.return_metric, args.intent_metric)
    detailed = _build_detailed_points(df, args.return_metric, args.intent_metric)
    if not args.include_baseline:
        points = points[points["method"].str.upper() != "BASELINE"].copy()
        detailed = detailed[detailed["method"].str.upper() != "BASELINE"].copy()

    # Save exact plotting points for reproducibility.
    out_points = Path(args.output_points_csv)
    out_points.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(out_points, index=False)

    sources = [s for s in ["x", "youtube"] if s in set(points["source"])]
    if not sources:
        raise RuntimeError("No supported sources found in points table.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(sources), figsize=(args.fig_width, args.fig_height), squeeze=False)
    axes = axes[0]

    for i, (ax, source) in enumerate(zip(axes, sources)):
        sdf = points[points["source"] == source].copy()
        ddf = detailed[detailed["source"] == source].copy()
        methods = _sort_methods(sdf["method"].tolist())
        for method in methods:
            row = sdf[sdf["method"] == method]
            if row.empty:
                continue
            color = METHOD_COLORS.get(method, "#333333")
            point_marker = METHOD_POINT_MARKERS.get(method, "o")

            if args.show_kol_points:
                mdf = ddf[ddf["method"] == method]
                if not mdf.empty:
                    mdf = mdf.sort_values("kol").reset_index(drop=True)
                    n = len(mdf)
                    xvals = mdf["intent_consistency"].to_numpy(dtype=float) + _deterministic_jitter(
                        n, args.kol_point_jitter_x
                    )
                    yvals = mdf["ret"].to_numpy(dtype=float) + _deterministic_jitter(
                        n, args.kol_point_jitter_y
                    )
                    alpha = args.kol_point_alpha * (1.35 if method == "KICL" else 1.0)
                    size = args.kol_point_size * (1.25 if method == "KICL" else 1.0)
                    ax.scatter(
                        xvals,
                        yvals,
                        s=size,
                        marker=point_marker,
                        c=[color],
                        alpha=min(alpha, 0.55),
                        linewidths=0.45 if point_marker in {"X", "x", "+", "P"} else 0,
                        zorder=1,
                    )

            x = float(row["intent_mean"].iloc[0])
            y = float(row["ret_mean"].iloc[0])
            xerr = float(row["intent_std"].iloc[0])
            yerr = float(row["ret_std"].iloc[0])
            marker = METHOD_MEAN_MARKERS.get(method, "o")
            size = 168 if method == "KICL" else 86
            lw = 1.45 if method == "KICL" else 1.0

            if method == "KICL":
                # Halo ring to make KICL visually dominant without clutter.
                ax.scatter(
                    [x],
                    [y],
                    s=size * 1.9,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.1,
                    alpha=0.45,
                    zorder=3.6,
                )

            ax.scatter(
                [x],
                [y],
                s=size,
                c=[color],
                marker=marker,
                edgecolors="black",
                linewidths=lw,
                zorder=4 if method == "KICL" else 3,
                label=method,
            )
            if args.show_errorbars:
                ax.errorbar(
                    [x],
                    [y],
                    xerr=[xerr],
                    yerr=[yerr],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.0,
                    alpha=0.45,
                    capsize=2.0,
                    zorder=2,
                )
            show_label = (
                (args.label_mode == "all")
                or (args.label_mode == "kicl" and method == "KICL")
            )
            if show_label:
                dx, dy = SOURCE_LABEL_OFFSETS.get(source, {}).get(
                    method, LABEL_OFFSETS.get(method, (4, 4))
                )
                ax.annotate(
                    method,
                    (x, y),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=args.anno_fontsize,
                    weight="bold" if method == "KICL" else "normal",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                    ),
                )

        if args.show_pareto and not sdf.empty:
            frontier = _pareto_frontier(sdf)
            if len(frontier) >= 2:
                ax.plot(
                    frontier["intent_mean"].values,
                    frontier["ret_mean"].values,
                    linestyle=args.pareto_linestyle,
                    linewidth=args.pareto_linewidth,
                    color=args.pareto_color,
                    alpha=0.9,
                    zorder=2.2,
                )
            elif len(frontier) == 1:
                # one-point frontier: subtle ring hint
                fx = float(frontier["intent_mean"].iloc[0])
                fy = float(frontier["ret_mean"].iloc[0])
                ax.scatter(
                    [fx],
                    [fy],
                    s=220,
                    facecolors="none",
                    edgecolors=args.pareto_color,
                    linewidths=1.0,
                    alpha=0.65,
                    zorder=2.1,
                )

        if args.focus_means and not sdf.empty:
            x_min, x_max = float(sdf["intent_mean"].min()), float(sdf["intent_mean"].max())
            y_min, y_max = float(sdf["ret_mean"].min()), float(sdf["ret_mean"].max())
            x_span = max(1e-6, x_max - x_min)
            y_span = max(1e-6, y_max - y_min)
            x_pad = max(0.012, x_span * args.focus_pad_ratio)
            y_pad = max(0.01, y_span * args.focus_pad_ratio)
            ax.set_xlim(max(0.0, x_min - x_pad), min(1.0, x_max + x_pad))
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

        if args.show_desirable_region and not sdf.empty:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            x_thr = float(sdf["intent_mean"].quantile(args.desirable_quantile))
            y_thr = float(sdf["ret_mean"].quantile(args.desirable_quantile))
            x_thr = min(max(x_thr, x0), x1)
            y_thr = min(max(y_thr, y0), y1)
            w = max(0.0, x1 - x_thr)
            h = max(0.0, y1 - y_thr)
            if w > 0 and h > 0:
                ax.add_patch(
                    Rectangle(
                        (x_thr, y_thr),
                        w,
                        h,
                        facecolor=args.desirable_color,
                        edgecolor="none",
                        alpha=args.desirable_alpha,
                        zorder=0.2,
                    )
                )
                ax.plot([x_thr, x1], [y_thr, y_thr], linestyle=":", linewidth=0.8, color="#4B4B4B", alpha=0.45, zorder=0.9)
                ax.plot([x_thr, x_thr], [y_thr, y1], linestyle=":", linewidth=0.8, color="#4B4B4B", alpha=0.45, zorder=0.9)

        if args.show_better_arrow:
            ax.annotate(
                "better (top-right)",
                xy=(0.985, 0.98),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=args.anno_fontsize,
                color="#2F4F2F",
                weight="bold",
                zorder=5,
            )

        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.28)
        ax.set_title("X" if source == "x" else "YouTube", fontsize=args.title_fontsize)
        if not args.shared_x_label:
            ax.set_xlabel("Intent consistency (higher is better)", fontsize=args.label_fontsize)
        if args.shared_y_label:
            ax.set_ylabel(
                (
                    "Event cumulative return"
                    if args.return_metric == "event_cumulative_return"
                    else "Daily cumulative return"
                )
                if i == 0
                else "",
                fontsize=args.label_fontsize,
            )
        else:
            ax.set_ylabel(
                "Event cumulative return"
                if args.return_metric == "event_cumulative_return"
                else "Daily cumulative return",
                fontsize=args.label_fontsize,
            )
        ax.tick_params(labelsize=args.tick_fontsize)
        if args.minimal_x_ticks:
            x0, x1 = ax.get_xlim()
            xticks = [x0]
            if x0 < 0.999:
                xticks.append(1.0)
            ax.set_xticks(xticks)
            ax.set_xticklabels(
                [f"{v:.2f}".rstrip("0").rstrip(".") for v in xticks],
                fontsize=args.tick_fontsize,
            )
        if args.hide_y_tick_labels:
            ax.tick_params(labelleft=False)
        if args.shared_y_label and args.hide_y_ticks_on_right and i > 0:
            ax.tick_params(left=False, labelleft=False)

    if args.show_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        fig.legend(
            list(uniq.values()),
            list(uniq.keys()),
            ncol=min(6, len(uniq)),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            fontsize=8.8,
            frameon=True,
        )
    if args.show_suptitle:
        fig.suptitle(
            "Main Trade-off: Return vs Intent Consistency (Selected-20, method means)",
            fontsize=12.4,
            y=0.995,
        )
    if args.shared_x_label:
        fig.supxlabel("Intent consistency (higher is better)", fontsize=args.label_fontsize, y=0.01)
    bottom_pad = 0.06 if args.show_legend else 0.02
    if args.shared_x_label and bottom_pad < 0.06:
        bottom_pad = 0.06
    top_pad = 0.94 if args.show_suptitle else 0.99
    fig.tight_layout(rect=(0, bottom_pad, 1, top_pad), w_pad=1.2)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png = prefix.with_suffix(".png")
    pdf = prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"Saved figure: {png}")
    print(f"Saved figure: {pdf}")
    print(f"Saved points: {out_points}")


if __name__ == "__main__":
    main()
