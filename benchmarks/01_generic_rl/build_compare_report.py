"""Build per-KOL comparison tables/plots across methods.

Default comparison:
- KICL (mainline method)
- BC benchmark
- IQL benchmark

Outputs are written under benchmarks/01_generic_rl/compare by default.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import pandas as pd


EVENT_KEYS = ["cumulative_return", "sharpe", "max_drawdown"]
BETRAYAL_KEYS = [
    "reversal_rate",
    "entry_violation_rate",
    "mean_abs_deviation",
    "mean_normalized_deviation",
    "sign_agreement_rate",
    "baseline_policy_corr",
]


@dataclass
class RunInfo:
    source: str
    kol: str
    run_name: str
    run_dir: Path
    timestamp: str
    event_metrics_path: Optional[Path]
    event_positions_path: Optional[Path]
    daily_metrics_path: Optional[Path]
    daily_equity_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build benchmark comparison report.")
    p.add_argument(
        "--ours-root",
        default="outputs/multisource_test_mainline",
        help="Root path containing mainline method test outputs.",
    )
    p.add_argument(
        "--ours-name",
        default="KICL",
        help="Display name for mainline method (instead of 'ours').",
    )
    p.add_argument(
        "--method",
        action="append",
        default=[],
        help="Additional method in NAME=PATH format. Can repeat.",
    )
    p.add_argument(
        "--output-dir",
        default="benchmarks/compare",
        help="Directory to save comparison outputs.",
    )
    p.add_argument(
        "--mode",
        choices=["anchor_ours", "intersection", "union"],
        default="anchor_ours",
        help=(
            "How to choose KOLs: anchor_ours=all KOLs in ours-name; "
            "intersection=only KOLs shared by all methods; union=all KOLs across methods."
        ),
    )
    p.add_argument(
        "--plot-format",
        choices=["png", "pdf"],
        default="png",
        help="Output format for per-KOL equity plots.",
    )
    p.add_argument(
        "--event-curve-mode",
        choices=["daily_mtm", "signal_step"],
        default="daily_mtm",
        help=(
            "How to build event_equity_compare: "
            "daily_mtm=use daily mark-to-market trained equity (non-flat between signals); "
            "signal_step=accumulate signal-step returns from positions_test.csv."
        ),
    )
    p.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include baseline curve in per-KOL daily equity plots/csv.",
    )
    p.add_argument(
        "--highlight-method",
        default=None,
        help="Method name to highlight in per-KOL equity curves. Default: --ours-name.",
    )
    p.add_argument(
        "--highlight-color",
        default="#1f77b4",
        help="Color for highlighted method curve.",
    )
    p.add_argument(
        "--highlight-linewidth",
        type=float,
        default=3.2,
        help="Line width for highlighted method curve.",
    )
    p.add_argument(
        "--other-linewidth",
        type=float,
        default=1.8,
        help="Line width for non-highlighted curves.",
    )
    p.add_argument(
        "--other-alpha",
        type=float,
        default=1.0,
        help="Alpha for non-highlighted curves.",
    )
    return p.parse_args()


def parse_method_args(extra_methods: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for raw in extra_methods:
        if "=" not in raw:
            raise ValueError(f"--method expects NAME=PATH, got: {raw}")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --method value: {raw}")
        out[name] = Path(path)
    return out


def split_run_name(run_name: str) -> Tuple[str, str]:
    m = re.match(r"(.+)_([0-9]{8}_[0-9]{6})$", run_name)
    if not m:
        return run_name, ""
    return m.group(1), m.group(2)


def discover_latest_runs(root: Path) -> Dict[Tuple[str, str], RunInfo]:
    if not root.exists():
        raise FileNotFoundError(f"Method root not found: {root}")

    latest: Dict[Tuple[str, str], RunInfo] = {}
    for source_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        source = source_dir.name
        for run_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
            event_metrics = run_dir / "event" / "metrics_test.json"
            event_positions = run_dir / "event" / "positions_test.csv"
            daily_metrics = run_dir / "daily" / "metrics_daily.json"
            daily_equity = run_dir / "daily" / "equity_daily.csv"
            if not event_metrics.exists() and not daily_metrics.exists():
                continue

            kol, ts = split_run_name(run_dir.name)
            key = (source, kol)
            info = RunInfo(
                source=source,
                kol=kol,
                run_name=run_dir.name,
                run_dir=run_dir,
                timestamp=ts,
                event_metrics_path=event_metrics if event_metrics.exists() else None,
                event_positions_path=event_positions if event_positions.exists() else None,
                daily_metrics_path=daily_metrics if daily_metrics.exists() else None,
                daily_equity_path=daily_equity if daily_equity.exists() else None,
            )

            if key not in latest:
                latest[key] = info
                continue

            prev = latest[key]
            if info.timestamp and prev.timestamp:
                if info.timestamp > prev.timestamp:
                    latest[key] = info
            elif info.timestamp and not prev.timestamp:
                latest[key] = info
            elif not info.timestamp and not prev.timestamp:
                if info.run_dir.stat().st_mtime > prev.run_dir.stat().st_mtime:
                    latest[key] = info
    return latest


def read_json(path: Optional[Path]) -> Dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_equity(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" not in df.columns:
        return pd.DataFrame()
    keep_cols = ["date"]
    if "equity_trained" in df.columns:
        keep_cols.append("equity_trained")
    if "equity_baseline" in df.columns:
        keep_cols.append("equity_baseline")
    if len(keep_cols) <= 1:
        return pd.DataFrame()
    out = df[keep_cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def read_event_equity_from_positions(path: Optional[Path]) -> pd.DataFrame:
    """Rebuild event-equity curve from positions_test.csv with evaluate_run.py logic."""
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"date", "weight", "reward"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    # `positions_test.csv` is written after merging with action logs in evaluate_run.py,
    # which can introduce duplicate rows for the same portfolio transition.
    # Deduplicate to recover the original transition-level rows.
    dedup_cols = [
        "date",
        "ticker",
        "reward",
        "raw_score",
        "prev_weight",
        "weight",
        "weight_delta",
        "allocation",
        "allocation_delta",
        "action",
    ]
    present = [c for c in dedup_cols if c in df.columns]
    if present:
        df = df[present].drop_duplicates()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["weighted_return"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0) * pd.to_numeric(
        df["reward"], errors="coerce"
    ).fillna(0.0)
    agg = df.groupby("date", as_index=False)["weighted_return"].sum().sort_values("date")
    agg["equity"] = (1.0 + agg["weighted_return"]).cumprod()
    return agg[["date", "equity"]]


def safe_col(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def method_styles(
    method_order: List[str],
    highlight_method: str,
    highlight_color: str,
    highlight_linewidth: float,
    other_linewidth: float,
    other_alpha: float,
) -> Dict[str, Dict]:
    base_palette = {
        "BC": "#8c564b",
        "IQL": "#2ca02c",
        "CQL": "#d62728",
        "TD3BC": "#9467bd",
        "AWAC": "#17becf",
    }
    cmap = plt.get_cmap("tab10")
    out: Dict[str, Dict] = {}
    for i, method in enumerate(method_order):
        out[method] = {
            "color": base_palette.get(method, cmap(i % 10)),
            "linewidth": other_linewidth,
            "alpha": other_alpha,
            "zorder": 2,
            "label": method,
            "highlight": False,
        }
    if highlight_method in out:
        out[highlight_method].update(
            {
                "color": highlight_color,
                "linewidth": highlight_linewidth,
                "alpha": 1.0,
                "zorder": 6,
                "label": f"{highlight_method} (Ours)",
                "highlight": True,
            }
        )
    return out


def beautify_equity_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, pad=10, fontweight="semibold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Equity", fontsize=10)

    # Light, readable grid: major + minor on y; major on x.
    ax.set_facecolor("#fbfcff")
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.28)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.22)
    ax.minorticks_on()
    # Reference line for initial equity.
    ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1.0, alpha=0.55, zorder=0)

    # Improve date readability with explicit month-year ticks.
    xmin, xmax = ax.get_xlim()
    left = mdates.num2date(xmin)
    right = mdates.num2date(xmax)
    span_days = max(1, (right - left).days)
    if span_days <= 220:
        month_interval = 1
    elif span_days <= 500:
        month_interval = 2
    else:
        month_interval = 3
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")

    # Subtle frame styling.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)
    ax.tick_params(axis="both", labelsize=9, length=4, width=0.8)


def draw_method_line(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    style: Dict,
) -> None:
    line = ax.plot(
        x,
        y,
        label=style.get("label", ""),
        linewidth=style.get("linewidth", 1.8),
        color=style.get("color"),
        alpha=style.get("alpha", 1.0),
        zorder=style.get("zorder", 2),
    )[0]
    if style.get("highlight", False):
        line.set_path_effects(
            [
                pe.Stroke(linewidth=style.get("linewidth", 1.8) + 2.0, foreground="white"),
                pe.Normal(),
            ]
        )
        y_clean = pd.to_numeric(y, errors="coerce")
        valid = y_clean.notna()
        if valid.any():
            x_last = x[valid].iloc[-1]
            y_last = y_clean[valid].iloc[-1]
            ax.scatter(
                [x_last],
                [y_last],
                s=34,
                color=style.get("color"),
                edgecolors="white",
                linewidths=0.9,
                zorder=8,
            )


def build_key_set(
    mode: str,
    method_runs: Dict[str, Dict[Tuple[str, str], RunInfo]],
    anchor_method: str,
) -> List[Tuple[str, str]]:
    sets = {k: set(v.keys()) for k, v in method_runs.items()}
    if mode == "anchor_ours":
        keys = sets.get(anchor_method, set())
    elif mode == "intersection":
        keys = set.intersection(*sets.values()) if sets else set()
    else:
        keys = set.union(*sets.values()) if sets else set()
    return sorted(keys, key=lambda x: (x[0], x[1].lower()))


def write_per_kol_outputs(
    key: Tuple[str, str],
    method_order: List[str],
    method_runs: Dict[str, Dict[Tuple[str, str], RunInfo]],
    output_root: Path,
    plot_format: str,
    event_curve_mode: str,
    styles: Dict[str, Dict],
    include_baseline: bool,
) -> Dict:
    source, kol = key
    kol_dir = output_root / source / kol
    kol_dir.mkdir(parents=True, exist_ok=True)

    event_rows = []
    daily_rows = []
    betrayal_rows = []
    curves: List[pd.DataFrame] = []
    baseline_curve: Optional[pd.DataFrame] = None
    event_curves: List[pd.DataFrame] = []

    summary_row = {"source": source, "kol": kol}

    for method in method_order:
        info = method_runs.get(method, {}).get(key)
        event = read_json(info.event_metrics_path if info else None)
        daily = read_json(info.daily_metrics_path if info else None)
        betrayal = event.get("betrayal_metrics", {}) if event else {}

        er = {"method": method, "run_name": info.run_name if info else ""}
        for k in EVENT_KEYS:
            er[k] = event.get(k)
            summary_row[f"{safe_col(method)}_event_{k}"] = event.get(k)
        event_rows.append(er)

        br = {"method": method, "run_name": info.run_name if info else ""}
        for k in BETRAYAL_KEYS:
            br[k] = betrayal.get(k)
            summary_row[f"{safe_col(method)}_betrayal_{k}"] = betrayal.get(k)
        betrayal_rows.append(br)

        trained = daily.get("trained", {}) if daily else {}
        baseline = daily.get("baseline", {}) if daily else {}
        dr = {"method": method, "run_name": info.run_name if info else ""}
        for k in EVENT_KEYS:
            dr[f"trained_{k}"] = trained.get(k)
            dr[f"baseline_{k}"] = baseline.get(k)
            summary_row[f"{safe_col(method)}_daily_trained_{k}"] = trained.get(k)
            summary_row[f"{safe_col(method)}_daily_baseline_{k}"] = baseline.get(k)
        daily_rows.append(dr)

        eq = read_equity(info.daily_equity_path if info else None)
        if not eq.empty and "equity_trained" in eq.columns:
            curve = eq[["date", "equity_trained"]].rename(columns={"equity_trained": method})
            curves.append(curve)
        if (
            include_baseline
            and baseline_curve is None
            and not eq.empty
            and "equity_baseline" in eq.columns
        ):
            baseline_curve = eq[["date", "equity_baseline"]].rename(
                columns={"equity_baseline": "Baseline"}
            )

        if event_curve_mode == "daily_mtm":
            ev_curve = pd.DataFrame()
            if info:
                ev_daily = read_equity(info.daily_equity_path)
                if not ev_daily.empty and "equity_trained" in ev_daily.columns:
                    ev_curve = ev_daily[["date", "equity_trained"]].rename(
                        columns={"equity_trained": "equity"}
                    )
        else:
            ev_curve = read_event_equity_from_positions(info.event_positions_path if info else None)
        if not ev_curve.empty:
            event_curves.append(ev_curve.rename(columns={"equity": method}))

    pd.DataFrame(event_rows).to_csv(kol_dir / "event_metrics_compare.csv", index=False)
    pd.DataFrame(betrayal_rows).to_csv(kol_dir / "betrayal_metrics_compare.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(kol_dir / "daily_metrics_compare.csv", index=False)

    merged: Optional[pd.DataFrame] = None
    for c in curves:
        if merged is None:
            merged = c.copy()
        else:
            merged = merged.merge(c, on="date", how="outer")
    if merged is not None:
        if include_baseline and baseline_curve is not None:
            merged = merged.merge(baseline_curve, on="date", how="left")
        merged = merged.sort_values("date")
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        merged.to_csv(kol_dir / "equity_daily_compare.csv", index=False)

        plot_df = pd.read_csv(kol_dir / "equity_daily_compare.csv")
        plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
        plot_df = plot_df.dropna(subset=["date"]).sort_values("date")

        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        for method in method_order:
            if method in plot_df.columns:
                style = styles.get(method, {})
                draw_method_line(ax=ax, x=plot_df["date"], y=plot_df[method], style=style)
        if include_baseline and "Baseline" in plot_df.columns:
            ax.plot(
                plot_df["date"],
                plot_df["Baseline"],
                label="Baseline",
                linestyle="--",
                linewidth=1.3,
                color="#444444",
                alpha=0.62,
                zorder=1,
            )
        beautify_equity_axis(ax, f"{source}/{kol} Daily Equity Comparison")
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9, ncol=1)
        fig.tight_layout()
        fig.savefig(kol_dir / f"equity_daily_compare.{plot_format}", dpi=200)
        plt.close(fig)

    merged_event: Optional[pd.DataFrame] = None
    for c in event_curves:
        if merged_event is None:
            merged_event = c.copy()
        else:
            merged_event = merged_event.merge(c, on="date", how="outer")
    if merged_event is not None:
        merged_event = merged_event.sort_values("date")
        merged_event_out = merged_event.copy()
        merged_event_out["date"] = merged_event_out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        merged_event_out.to_csv(kol_dir / "event_equity_compare.csv", index=False)

        plot_df = merged_event.sort_values("date")
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        for method in method_order:
            if method in plot_df.columns:
                style = styles.get(method, {})
                draw_method_line(ax=ax, x=plot_df["date"], y=plot_df[method], style=style)
        beautify_equity_axis(ax, f"{source}/{kol} Event Equity Comparison")
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9, ncol=1)
        fig.tight_layout()
        fig.savefig(kol_dir / f"event_equity_compare.{plot_format}", dpi=200)
        plt.close(fig)

    return summary_row


def summarize_by_method(summary_df: pd.DataFrame, method_order: List[str]) -> pd.DataFrame:
    rows = []
    for method in method_order:
        prefix = safe_col(method)
        row = {"method": method, "n_kols": len(summary_df)}
        for k in EVENT_KEYS:
            row[f"event_mean_{k}"] = summary_df[f"{prefix}_event_{k}"].mean()
        for k in BETRAYAL_KEYS:
            row[f"betrayal_mean_{k}"] = summary_df[f"{prefix}_betrayal_{k}"].mean()
        for k in EVENT_KEYS:
            row[f"daily_trained_mean_{k}"] = summary_df[f"{prefix}_daily_trained_{k}"].mean()
            row[f"daily_baseline_mean_{k}"] = summary_df[f"{prefix}_daily_baseline_{k}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def write_overview_plots(
    summary_df: pd.DataFrame, method_order: List[str], output_dir: Path
) -> None:
    # Event metrics: boxplots
    for metric in EVENT_KEYS:
        cols = [f"{safe_col(m)}_event_{metric}" for m in method_order]
        data = [summary_df[c].dropna().tolist() for c in cols]
        if not any(len(x) > 0 for x in data):
            continue
        plt.figure(figsize=(8.5, 5.2))
        plt.boxplot(data, tick_labels=method_order, showmeans=True)
        plt.title(f"Event {metric} Distribution by Method")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_dir / f"overview_event_{metric}_boxplot.png", dpi=180)
        plt.close()

    # Mean betrayal metrics as grouped bars
    selected_betrayal = [
        "reversal_rate",
        "entry_violation_rate",
        "mean_abs_deviation",
        "baseline_policy_corr",
    ]
    rows = []
    for method in method_order:
        prefix = safe_col(method)
        row = {"method": method}
        for m in selected_betrayal:
            row[m] = summary_df[f"{prefix}_betrayal_{m}"].mean()
        rows.append(row)
    bdf = pd.DataFrame(rows)
    bdf.to_csv(output_dir / "overview_betrayal_means.csv", index=False)
    if not bdf.empty:
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5))
        axes = axes.ravel()
        for i, m in enumerate(selected_betrayal):
            ax = axes[i]
            ax.bar(bdf["method"], bdf[m])
            ax.set_title(f"Mean {m}")
            ax.tick_params(axis="x", rotation=15)
        fig.suptitle("Betrayal Metric Means by Method")
        fig.tight_layout()
        fig.savefig(output_dir / "overview_betrayal_means.png", dpi=180)
        plt.close(fig)

    # Event metric means (bar)
    rows = []
    for method in method_order:
        prefix = safe_col(method)
        row = {"method": method}
        for m in EVENT_KEYS:
            row[m] = summary_df[f"{prefix}_event_{m}"].mean()
        rows.append(row)
    edf = pd.DataFrame(rows)
    edf.to_csv(output_dir / "overview_event_means.csv", index=False)
    if not edf.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
        for i, m in enumerate(EVENT_KEYS):
            ax = axes[i]
            ax.bar(edf["method"], edf[m])
            ax.set_title(f"Mean {m}")
            ax.tick_params(axis="x", rotation=15)
        fig.suptitle("Event Metric Means by Method")
        fig.tight_layout()
        fig.savefig(output_dir / "overview_event_means.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()

    methods = {args.ours_name: Path(args.ours_root)}
    default_methods = {
        "BC": Path("benchmarks/01_generic_rl/test results/bc_event_test"),
        "IQL": Path("benchmarks/01_generic_rl/test results/iql_event_test"),
    }
    for name, path in default_methods.items():
        if path.exists():
            methods[name] = path
    if args.method:
        methods.update(parse_method_args(args.method))
    method_order = list(methods.keys())
    anchor_method = args.ours_name
    highlight_method = args.highlight_method or args.ours_name
    styles = method_styles(
        method_order=method_order,
        highlight_method=highlight_method,
        highlight_color=args.highlight_color,
        highlight_linewidth=args.highlight_linewidth,
        other_linewidth=args.other_linewidth,
        other_alpha=args.other_alpha,
    )

    method_runs = {name: discover_latest_runs(root) for name, root in methods.items()}
    keys = build_key_set(args.mode, method_runs, anchor_method=anchor_method)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for key in keys:
        summary_rows.append(
            write_per_kol_outputs(
                key=key,
                method_order=method_order,
                method_runs=method_runs,
                output_root=out_dir,
                plot_format=args.plot_format,
                event_curve_mode=args.event_curve_mode,
                styles=styles,
                include_baseline=args.include_baseline,
            )
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["source", "kol"])
    summary_df.to_csv(out_dir / "summary_by_kol.csv", index=False)

    method_summary = summarize_by_method(summary_df, method_order)
    method_summary.to_csv(out_dir / "summary_by_method_mean.csv", index=False)
    write_overview_plots(summary_df, method_order, out_dir)

    meta = {
        "mode": args.mode,
        "event_curve_mode": args.event_curve_mode,
        "include_baseline": args.include_baseline,
        "highlight_method": highlight_method,
        "methods": {k: str(v) for k, v in methods.items()},
        "n_kols_total": int(len(summary_df)),
        "output_dir": str(out_dir),
    }
    with (out_dir / "compare_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved comparison to: {out_dir}")
    print(f"Total KOL entries: {len(summary_df)}")
    print("Methods:", ", ".join(method_order))


if __name__ == "__main__":
    main()
