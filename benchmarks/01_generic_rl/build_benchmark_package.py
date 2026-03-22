#!/usr/bin/env python3
"""Build a clean benchmark package from canonical compare outputs.

Input:
  benchmarks/compare/canonical_all/tables/summary_by_kol.csv
  benchmarks/compare/canonical_all/tables/summary_by_method_mean_by_source.csv

Output:
  benchmarks/benchmark_package/mainline/
    - tables/*.csv
    - figures/*.png
    - README.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["KICL", "BC", "IQL", "TD3BC", "CQL", "AWAC"]
METHOD_COLORS: Dict[str, str] = {
    "KICL": "#F39C12",
    "BC": "#8c564b",
    "IQL": "#2ca02c",
    "TD3BC": "#9467bd",
    "CQL": "#d62728",
    "AWAC": "#17becf",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("benchmarks/compare/canonical_all"),
        help="Canonical merged compare root.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/benchmark_package/mainline"),
        help="Benchmark package output root.",
    )
    return p.parse_args()


def _sort_methods(df: pd.DataFrame) -> pd.DataFrame:
    if "method" not in df.columns:
        return df
    out = df.copy()
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["source", "method"]).reset_index(drop=True)


def save_tables(by_source: pd.DataFrame, out_tables: Path) -> None:
    out_tables.mkdir(parents=True, exist_ok=True)

    event_cols = [
        "source",
        "method",
        "n_kols",
        "event_mean_cumulative_return",
        "event_mean_sharpe",
        "event_mean_max_drawdown",
    ]
    betrayal_cols = [
        "source",
        "method",
        "n_kols",
        "betrayal_mean_reversal_rate",
        "betrayal_mean_entry_violation_rate",
        "betrayal_mean_mean_abs_deviation",
        "betrayal_mean_mean_normalized_deviation",
        "betrayal_mean_sign_agreement_rate",
        "betrayal_mean_baseline_policy_corr",
    ]
    daily_cols = [
        "source",
        "method",
        "n_kols",
        "daily_trained_mean_cumulative_return",
        "daily_baseline_mean_cumulative_return",
        "daily_trained_mean_sharpe",
        "daily_baseline_mean_sharpe",
        "daily_trained_mean_max_drawdown",
        "daily_baseline_mean_max_drawdown",
    ]

    event_df = by_source[event_cols].copy()
    betrayal_df = by_source[betrayal_cols].copy()
    daily_df = by_source[daily_cols].copy()

    event_df.to_csv(out_tables / "benchmark_event_by_source.csv", index=False)
    betrayal_df.to_csv(out_tables / "benchmark_betrayal_by_source.csv", index=False)
    daily_df.to_csv(out_tables / "benchmark_daily_by_source.csv", index=False)

    # Ranking tables
    rank_event = (
        event_df.sort_values(["source", "event_mean_cumulative_return"], ascending=[True, False])
        .groupby("source", group_keys=False)
        .apply(lambda x: x.assign(rank_event_return=range(1, len(x) + 1)))
        .reset_index(drop=True)
    )
    rank_event.to_csv(out_tables / "benchmark_ranking_event_return.csv", index=False)

    rank_betrayal = (
        betrayal_df.sort_values(
            ["source", "betrayal_mean_mean_abs_deviation"], ascending=[True, True]
        )
        .groupby("source", group_keys=False)
        .apply(lambda x: x.assign(rank_betrayal_absdev=range(1, len(x) + 1)))
        .reset_index(drop=True)
    )
    rank_betrayal.to_csv(out_tables / "benchmark_ranking_betrayal.csv", index=False)


def _bar_colors(methods: List[str]) -> List[str]:
    return [METHOD_COLORS.get(m, "#4c72b0") for m in methods]


def save_figures(by_source: pd.DataFrame, out_fig: Path) -> None:
    out_fig.mkdir(parents=True, exist_ok=True)
    sources = sorted(by_source["source"].unique().tolist())

    # 1) Event return bars by source
    fig, axes = plt.subplots(1, len(sources), figsize=(6.2 * len(sources), 4.6), squeeze=False)
    for i, src in enumerate(sources):
        ax = axes[0, i]
        sdf = by_source[by_source["source"] == src].copy()
        sdf["method"] = pd.Categorical(sdf["method"], categories=METHOD_ORDER, ordered=True)
        sdf = sdf.sort_values("method")
        ax.bar(
            sdf["method"].astype(str),
            sdf["event_mean_cumulative_return"],
            color=_bar_colors(sdf["method"].astype(str).tolist()),
            alpha=0.9,
        )
        ax.set_title(f"{src}: Event Mean Cumulative Return")
        ax.set_ylabel("cumulative_return")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_fig / "event_return_by_source.png", dpi=200)
    plt.close(fig)

    # 2) Betrayal bars (mean abs deviation) by source
    fig, axes = plt.subplots(1, len(sources), figsize=(6.2 * len(sources), 4.6), squeeze=False)
    for i, src in enumerate(sources):
        ax = axes[0, i]
        sdf = by_source[by_source["source"] == src].copy()
        sdf["method"] = pd.Categorical(sdf["method"], categories=METHOD_ORDER, ordered=True)
        sdf = sdf.sort_values("method")
        ax.bar(
            sdf["method"].astype(str),
            sdf["betrayal_mean_mean_abs_deviation"],
            color=_bar_colors(sdf["method"].astype(str).tolist()),
            alpha=0.9,
        )
        ax.set_title(f"{src}: Betrayal Mean Abs Deviation")
        ax.set_ylabel("mean_abs_deviation (lower better)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_fig / "betrayal_absdev_by_source.png", dpi=200)
    plt.close(fig)

    # 3) Tradeoff scatter by source (x: intent consistency proxy, y: event return)
    fig, axes = plt.subplots(1, len(sources), figsize=(6.2 * len(sources), 4.8), squeeze=False)
    for i, src in enumerate(sources):
        ax = axes[0, i]
        sdf = by_source[by_source["source"] == src].copy()
        # Higher better on x
        sdf["intent_consistency"] = 1.0 / (1.0 + sdf["betrayal_mean_mean_abs_deviation"].clip(lower=0.0))
        for _, r in sdf.iterrows():
            m = str(r["method"])
            ax.scatter(
                r["intent_consistency"],
                r["event_mean_cumulative_return"],
                s=170 if m == "KICL" else 110,
                marker="X" if m == "KICL" else "o",
                color=METHOD_COLORS.get(m, "#4c72b0"),
                edgecolors="white",
                linewidths=1.0,
                alpha=0.95,
                zorder=5 if m == "KICL" else 3,
            )
            ax.text(
                r["intent_consistency"] + 0.002,
                r["event_mean_cumulative_return"] + 0.002,
                m,
                fontsize=9,
            )
        ax.set_title(f"{src}: Event Return vs Intent Consistency")
        ax.set_xlabel("intent_consistency = 1/(1 + abs_deviation)")
        ax.set_ylabel("event_mean_cumulative_return")
        ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_fig / "tradeoff_return_vs_intent_by_source.png", dpi=220)
    plt.close(fig)


def save_readme(out_root: Path, by_source: pd.DataFrame) -> None:
    # Small textual summary for quick copy into notes/paper draft.
    lines: List[str] = []
    lines.append("# Benchmark Package (Mainline)")
    lines.append("")
    lines.append("Generated from `benchmarks/compare/canonical_all`.")
    lines.append("")
    lines.append("## Included")
    lines.append("- Tables: event / betrayal / daily (by source)")
    lines.append("- Rankings: by event return, by betrayal abs deviation")
    lines.append("- Figures: event bars, betrayal bars, return-vs-intent tradeoff")
    lines.append("")
    lines.append("## Quick Summary")
    for src in sorted(by_source["source"].unique().tolist()):
        sdf = by_source[by_source["source"] == src].copy()
        best_ret = sdf.sort_values("event_mean_cumulative_return", ascending=False).iloc[0]
        best_betr = sdf.sort_values("betrayal_mean_mean_abs_deviation", ascending=True).iloc[0]
        lines.append(
            f"- {src}: best event return = {best_ret['method']} ({best_ret['event_mean_cumulative_return']:.4f})"
        )
        lines.append(
            f"- {src}: best betrayal(abs dev) = {best_betr['method']} ({best_betr['betrayal_mean_mean_abs_deviation']:.4f})"
        )
    lines.append("")
    lines.append("## Paths")
    lines.append("- `tables/benchmark_event_by_source.csv`")
    lines.append("- `tables/benchmark_betrayal_by_source.csv`")
    lines.append("- `tables/benchmark_daily_by_source.csv`")
    lines.append("- `figures/event_return_by_source.png`")
    lines.append("- `figures/betrayal_absdev_by_source.png`")
    lines.append("- `figures/tradeoff_return_vs_intent_by_source.png`")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    canonical_tables = args.canonical_root / "tables"
    by_source_path = canonical_tables / "summary_by_method_mean_by_source.csv"
    if not by_source_path.exists():
        raise FileNotFoundError(f"Missing: {by_source_path}")

    by_source = pd.read_csv(by_source_path)
    by_source = _sort_methods(by_source)

    out_root = args.output_root
    out_tables = out_root / "tables"
    out_fig = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)

    save_tables(by_source, out_tables)
    save_figures(by_source, out_fig)
    save_readme(out_root, by_source)

    print(f"Saved benchmark package: {out_root}")
    print(f"Rows (source+method): {len(by_source)}")


if __name__ == "__main__":
    main()

