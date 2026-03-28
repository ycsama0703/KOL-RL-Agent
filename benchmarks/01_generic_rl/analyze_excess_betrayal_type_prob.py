#!/usr/bin/env python3
"""Experiment B (reframed): betrayal-type probabilities under excess-gain events.

Core question:
For each method, when excess gain occurs, what are the probabilities of
hard betrayal and soft betrayal?
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["KICL", "RMB", "SUP_DELTA", "HAP", "BC", "IQL", "AWAC", "CQL", "TD3BC"]
METHOD_COLORS = {
    "KICL": "#F39C12",
    "AWAC": "#17BECF",
    "IQL": "#2CA02C",
    "BC": "#8C564B",
    "CQL": "#D62728",
    "TD3BC": "#9467BD",
}
METHOD_DISPLAY_LABEL = {
    "SUP_DELTA": "SDELTA",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest-json",
        default="benchmarks/compare/meta/compare_manifest_benchtest.json",
        help="Manifest with method roots.",
    )
    p.add_argument(
        "--detailed-csv",
        default="benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed_benchtest.csv",
        help="Selected universe CSV (source, kol used as universe).",
    )
    p.add_argument(
        "--output-dir",
        default="benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20",
        help="Output directory.",
    )
    p.add_argument(
        "--method-order",
        nargs="*",
        default=METHOD_ORDER,
        help="Method order.",
    )
    p.add_argument(
        "--pair-mode",
        choices=["intersection", "union"],
        default="intersection",
        help="KOL pair mode across methods.",
    )
    p.add_argument("--entry-threshold", type=float, default=0.02)
    p.add_argument("--action-threshold", type=float, default=0.02)
    p.add_argument("--dev-threshold", type=float, default=0.20)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument(
        "--condition-mode",
        choices=["excess_vs_baseline_proxy", "profit_event"],
        default="excess_vs_baseline_proxy",
        help=(
            "excess_vs_baseline_proxy: cond = (policy_action-baseline_action)*reward > eps; "
            "profit_event: cond = event_return > eps."
        ),
    )
    p.add_argument("--dpi", type=int, default=320)
    p.add_argument("--fig-width", type=float, default=8.8)
    p.add_argument("--fig-height", type=float, default=3.5)
    p.add_argument("--font-size", type=float, default=11.0)
    p.add_argument("--x-color", default="#4C72B0", help="Bar color for X source.")
    p.add_argument("--youtube-color", default="#DD8452", help="Bar color for YouTube source.")
    return p.parse_args()


def split_run_name(run_name: str) -> Tuple[str, str]:
    m = re.match(r"(.+)_([0-9]{8}_[0-9]{6})$", run_name)
    if not m:
        return run_name, ""
    return m.group(1), m.group(2)


def discover_latest_runs(root: Path) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Tuple[str, Path]] = {}
    if not root.exists():
        return {}
    for source_dir in [p for p in root.iterdir() if p.is_dir()]:
        source = source_dir.name
        for run_dir in [p for p in source_dir.iterdir() if p.is_dir()]:
            kol, ts = split_run_name(run_dir.name)
            key = (source, kol)
            prev = out.get(key)
            if prev is None or ts > prev[0]:
                out[key] = (ts, run_dir)
    return {k: v[1] for k, v in out.items()}


def event_flags(
    df: pd.DataFrame,
    entry_threshold: float,
    action_threshold: float,
    dev_threshold: float,
    eps: float,
    condition_mode: str,
) -> pd.DataFrame:
    req = {"reward", "baseline_action", "policy_action"}
    if not req.issubset(df.columns):
        return pd.DataFrame()

    b = pd.to_numeric(df["baseline_action"], errors="coerce").fillna(0.0).to_numpy()
    p = pd.to_numeric(df["policy_action"], errors="coerce").fillna(0.0).to_numpy()
    r = pd.to_numeric(df["reward"], errors="coerce").fillna(0.0).to_numpy()

    if "weight" in df.columns:
        w = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy()
        event_return = w * r
    else:
        event_return = p * r

    has_signal = np.abs(b) >= entry_threshold
    no_signal = ~has_signal

    reversal = has_signal & ((b * p) < 0.0)
    entry_violation = no_signal & (np.abs(p) > action_threshold)

    dev = np.zeros(len(df), dtype=float)
    idx = has_signal
    dev[idx] = np.abs(p[idx] - b[idx]) / (np.abs(b[idx]) + 1e-8)
    dev_flag = dev >= dev_threshold

    hard_flag = reversal | entry_violation
    soft_flag = dev_flag
    any_flag = hard_flag | soft_flag

    excess_proxy = (p - b) * r
    if condition_mode == "profit_event":
        cond = event_return > eps
    else:
        cond = excess_proxy > eps

    return pd.DataFrame(
        {
            "cond_excess_pos": cond.astype(int),
            "hard_flag": hard_flag.astype(int),
            "soft_flag": soft_flag.astype(int),
            "any_flag": any_flag.astype(int),
            "reversal_flag": reversal.astype(int),
            "entry_flag": entry_violation.astype(int),
            "dev_value": dev.astype(float),
            "excess_proxy": excess_proxy.astype(float),
            "event_return": event_return.astype(float),
        }
    )


def _rate(v: np.ndarray) -> float:
    if v.size == 0:
        return float("nan")
    return float(v.mean())


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    methods = [m for m in args.method_order if m in manifest["methods"]]
    method_roots = {m: Path(manifest["methods"][m]) for m in methods}
    if not methods:
        raise RuntimeError("No methods available from manifest and method-order intersection.")

    detailed = pd.read_csv(args.detailed_csv)
    universe = set(map(tuple, detailed[["source", "kol"]].drop_duplicates().itertuples(index=False, name=None)))
    if not universe:
        raise RuntimeError("No source/kol pairs in detailed CSV.")

    latest_by_method = {m: discover_latest_runs(root) for m, root in method_roots.items()}
    pairs_by_method = {m: set(latest.keys()) & universe for m, latest in latest_by_method.items()}

    if args.pair_mode == "intersection":
        common_pairs = set.intersection(*(pairs_by_method[m] for m in methods))
    else:
        common_pairs = set.union(*(pairs_by_method[m] for m in methods))
    if not common_pairs:
        raise RuntimeError("No common pairs under current pair-mode.")

    coverage_rows: List[dict] = []
    rows: List[dict] = []

    for m in methods:
        latest = latest_by_method[m]
        method_pairs = sorted(common_pairs if args.pair_mode == "intersection" else pairs_by_method[m])
        found = 0
        n_events_total = 0
        for src, kol in method_pairs:
            run = latest.get((src, kol))
            if run is None:
                continue
            pos = run / "event" / "positions_test.csv"
            if not pos.exists():
                continue
            df = pd.read_csv(pos)
            fl = event_flags(
                df,
                entry_threshold=args.entry_threshold,
                action_threshold=args.action_threshold,
                dev_threshold=args.dev_threshold,
                eps=args.eps,
                condition_mode=args.condition_mode,
            )
            if fl.empty:
                continue

            cond = fl["cond_excess_pos"].to_numpy(dtype=int) == 1
            non = ~cond
            hard = fl["hard_flag"].to_numpy(dtype=int)
            soft = fl["soft_flag"].to_numpy(dtype=int)
            anyf = fl["any_flag"].to_numpy(dtype=int)
            rev = fl["reversal_flag"].to_numpy(dtype=int)
            ent = fl["entry_flag"].to_numpy(dtype=int)

            row = {
                "source": src,
                "kol": kol,
                "method": m,
                "n_events": int(len(fl)),
                "n_excess_pos": int(cond.sum()),
                "n_excess_nonpos": int(non.sum()),
                "p_hard_given_excess_pos": _rate(hard[cond]),
                "p_soft_given_excess_pos": _rate(soft[cond]),
                "p_any_given_excess_pos": _rate(anyf[cond]),
                "p_reversal_given_excess_pos": _rate(rev[cond]),
                "p_entry_given_excess_pos": _rate(ent[cond]),
                "p_hard_given_excess_nonpos": _rate(hard[non]),
                "p_soft_given_excess_nonpos": _rate(soft[non]),
                "p_any_given_excess_nonpos": _rate(anyf[non]),
                "uplift_hard": _rate(hard[cond]) - _rate(hard[non]),
                "uplift_soft": _rate(soft[cond]) - _rate(soft[non]),
                "uplift_any": _rate(anyf[cond]) - _rate(anyf[non]),
            }
            rows.append(row)
            found += 1
            n_events_total += len(fl)

        coverage_rows.append(
            {
                "method": m,
                "pair_mode": args.pair_mode,
                "pairs_target": len(method_pairs),
                "pairs_found": found,
                "event_rows": n_events_total,
            }
        )

    if not rows:
        raise RuntimeError("No event rows aggregated.")

    pd.DataFrame(coverage_rows).to_csv(out_dir / "excess_betrayal_type_coverage.csv", index=False)
    by_kol = pd.DataFrame(rows).sort_values(["source", "method", "kol"]).reset_index(drop=True)
    by_kol.to_csv(out_dir / "excess_betrayal_type_by_kol.csv", index=False)

    by_method_source = (
        by_kol.groupby(["source", "method"], as_index=False)
        .agg(
            n_kols=("kol", "count"),
            n_events=("n_events", "sum"),
            n_excess_pos=("n_excess_pos", "sum"),
            p_hard_given_excess_pos=("p_hard_given_excess_pos", "mean"),
            p_soft_given_excess_pos=("p_soft_given_excess_pos", "mean"),
            p_any_given_excess_pos=("p_any_given_excess_pos", "mean"),
            p_reversal_given_excess_pos=("p_reversal_given_excess_pos", "mean"),
            p_entry_given_excess_pos=("p_entry_given_excess_pos", "mean"),
            p_hard_given_excess_nonpos=("p_hard_given_excess_nonpos", "mean"),
            p_soft_given_excess_nonpos=("p_soft_given_excess_nonpos", "mean"),
            p_any_given_excess_nonpos=("p_any_given_excess_nonpos", "mean"),
            uplift_hard=("uplift_hard", "mean"),
            uplift_soft=("uplift_soft", "mean"),
            uplift_any=("uplift_any", "mean"),
        )
        .sort_values(["source", "method"])
        .reset_index(drop=True)
    )
    by_method_source.to_csv(out_dir / "excess_betrayal_type_by_method_source.csv", index=False)

    # Plot: grouped bars of hard betrayal probability by source (X vs YouTube)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": args.font_size,
            "axes.titlesize": args.font_size + 2,
            "axes.labelsize": args.font_size + 1,
            "xtick.labelsize": args.font_size - 0.5,
            "ytick.labelsize": args.font_size - 0.5,
            "legend.fontsize": args.font_size - 0.5,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(args.fig_width, args.fig_height))
    order = [m for m in methods if m in set(by_method_source["method"])]
    xs = np.arange(len(order))
    w = 0.36

    src_x = (
        by_method_source[by_method_source["source"] == "x"][["method", "p_hard_given_excess_pos"]]
        .set_index("method")["p_hard_given_excess_pos"]
        .to_dict()
    )
    src_y = (
        by_method_source[by_method_source["source"] == "youtube"][["method", "p_hard_given_excess_pos"]]
        .set_index("method")["p_hard_given_excess_pos"]
        .to_dict()
    )

    vals_x = np.array([src_x.get(m, np.nan) for m in order], dtype=float)
    vals_y = np.array([src_y.get(m, np.nan) for m in order], dtype=float)

    bars_x = ax.bar(
        xs - w / 2,
        vals_x,
        width=w,
        label="X",
        color=args.x_color,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    bars_y = ax.bar(
        xs + w / 2,
        vals_y,
        width=w,
        label="YouTube",
        color=args.youtube_color,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    global_max = float(np.nanmax(np.concatenate([vals_x, vals_y]))) if (vals_x.size + vals_y.size) > 0 else 1.0
    # Tight headroom to avoid large empty top area in paper layout.
    y_upper = min(0.72, max(0.62, global_max + 0.035))
    label_offset = max(0.008, 0.012 * y_upper)

    for j, m in enumerate(order):
        if m == "KICL":
            bars_x[j].set_edgecolor("#111111")
            bars_x[j].set_linewidth(2.2)
            bars_y[j].set_edgecolor("#111111")
            bars_y[j].set_linewidth(2.2)

        ymax = np.nanmax([vals_x[j], vals_y[j]])
        method_y = None
        if np.isfinite(ymax):
            method_y = min(
                ymax + label_offset + (0.02 if m == "KICL" else 0.0),
                y_upper - 0.004,
            )
            if m == "KICL":
                # KICL bars are often near zero; force label higher to avoid overlap.
                method_y = max(method_y, 0.075)
            disp = METHOD_DISPLAY_LABEL.get(m, m)
            ax.text(
                xs[j],
                method_y,
                disp,
                ha="center",
                va="bottom",
                fontsize=args.font_size + 1.0,
                fontweight="bold",
            )
        # Numeric labels:
        # - KICL: above bars (below method label) for readability
        # - others: inside bars to avoid crowding
        vx = vals_x[j]
        vy = vals_y[j]
        if np.isfinite(vx):
            if m == "KICL" and method_y is not None:
                yx = min(vx + 0.018, method_y - 0.02)
                va_x = "bottom"
                color_x = "#1f2937"
            else:
                yx = max(0.01, (vx - 0.03) if vx >= 0.10 else (vx * 0.55))
                va_x = "top" if vx >= 0.10 else "center"
                color_x = "white" if vx >= 0.16 else "#1f2937"
            ax.text(
                xs[j] - w / 2,
                yx,
                f"{vx:.2f}",
                ha="center",
                va=va_x,
                fontsize=args.font_size - 2.0,
                color=color_x,
            )
        if np.isfinite(vy):
            if m == "KICL" and method_y is not None:
                yy = min(vy + 0.018, method_y - 0.02)
                va_y = "bottom"
                color_y = "#1f2937"
            else:
                yy = max(0.01, (vy - 0.03) if vy >= 0.10 else (vy * 0.55))
                va_y = "top" if vy >= 0.10 else "center"
                color_y = "white" if vy >= 0.16 else "#1f2937"
            ax.text(
                xs[j] + w / 2,
                yy,
                f"{vy:.2f}",
                ha="center",
                va=va_y,
                fontsize=args.font_size - 2.0,
                color=color_y,
            )

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylim(0.0, y_upper)
    # Dynamic ticks to avoid expanding the visual top margin.
    tick_step = 0.1 if y_upper <= 0.75 else 0.2
    ax.set_yticks(np.arange(0.0, y_upper + 1e-9, tick_step))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.30, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Hard betrayal prob.")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=2,
        frameon=True,
        prop={"size": args.font_size + 1.0, "weight": "bold"},
    )

    fig.tight_layout(rect=(0.02, 0.04, 1.0, 0.995))
    fig.savefig(out_dir / "excess_betrayal_type_probability.png", dpi=args.dpi)
    fig.savefig(out_dir / "excess_betrayal_type_probability.pdf")
    plt.close(fig)

    md_lines = []
    md_lines.append("# Excess-Condition Betrayal Type Probability")
    md_lines.append("")
    md_lines.append(f"- Pair mode: `{args.pair_mode}`")
    md_lines.append(f"- Condition mode: `{args.condition_mode}`")
    md_lines.append(f"- Universe pairs: `{len(universe)}`")
    md_lines.append(f"- Common pairs used: `{len(common_pairs)}`")
    md_lines.append("")
    md_lines.append("Outputs:")
    md_lines.append("- `excess_betrayal_type_coverage.csv`")
    md_lines.append("- `excess_betrayal_type_by_kol.csv`")
    md_lines.append("- `excess_betrayal_type_by_method_source.csv`")
    md_lines.append("- `excess_betrayal_type_probability.png`")
    (out_dir / "README_excess_betrayal_type_prob.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {out_dir / 'excess_betrayal_type_coverage.csv'}")
    print(f"Saved: {out_dir / 'excess_betrayal_type_by_kol.csv'}")
    print(f"Saved: {out_dir / 'excess_betrayal_type_by_method_source.csv'}")
    print(f"Saved: {out_dir / 'excess_betrayal_type_probability.png'}")
    print(f"Saved: {out_dir / 'excess_betrayal_type_probability.pdf'}")
    print(f"Saved: {out_dir / 'README_excess_betrayal_type_prob.md'}")


if __name__ == "__main__":
    main()
