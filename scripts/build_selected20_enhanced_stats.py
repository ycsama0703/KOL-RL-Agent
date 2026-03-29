#!/usr/bin/env python3
"""Build an enhanced statistics pack for selected-20 discourse analysis.

This script aggregates existing outputs from:
1) company-level discourse stats
2) replay-buffer signal/silence stats

It produces:
- per-KOL merged metric table (by source)
- per-source distribution summary (mean/median/p25/p75/min/max)
- bootstrap CI (mean metric over KOLs)
- source separability summary (Cohen's d + Cliff's delta)
- replay stats summary by source
- markdown brief for quick paper selection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build enhanced selected-20 stats pack.")
    parser.add_argument(
        "--company-stats-root",
        default="outputs/analysis/company_stats_selected20_20260321",
        help="Root folder that contains x/ and youtube/ company stats outputs.",
    )
    parser.add_argument(
        "--signal-stats-root",
        default="outputs/signal_silence_stats_selected20_by_source",
        help="Root folder that contains x/ and youtube/ signal-silence stats outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/company_stats_selected20_20260321/enhanced_pack",
        help="Output directory.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=5000,
        help="Bootstrap iterations for CI on per-source mean metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for bootstrap.",
    )
    return parser.parse_args()


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    v1 = np.var(a, ddof=1)
    v2 = np.var(b, ddof=1)
    n1, n2 = len(a), len(b)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled <= 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    n = len(a) * len(b)
    if n == 0:
        return np.nan
    return float((gt - lt) / n)


def bootstrap_mean_ci(x: np.ndarray, iters: int, rng: np.random.Generator) -> Dict[str, float]:
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    means = []
    n = len(x)
    for _ in range(iters):
        s = rng.choice(x, size=n, replace=True)
        means.append(float(np.mean(s)))
    means = np.asarray(means, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def load_per_kol_metrics(src_root: Path, source: str) -> pd.DataFrame:
    base = src_root / source
    freq = pd.read_csv(base / "per_kol_frequency_stats.csv")[
        ["kol", "mentioned_once_pct"]
    ].copy()
    fol = pd.read_csv(base / "per_kol_followup_summary.csv")
    fol = fol[fol["window_days"] == 90][["kol", "no_followup_within_window_pct"]].copy()
    sil = pd.read_csv(base / "per_kol_silence_stats.csv")
    sil = sil[sil["scope"] == "all"][["kol", "gap_p90_days"]].copy()
    rev = pd.read_csv(base / "per_kol_reversal_summary.csv")[
        ["kol", "reversal_rate", "ttf_median_days"]
    ].copy()
    imb = pd.read_csv(base / "per_kol_imbalance_summary.csv")[
        ["kol", "positive_ratio", "negative_ratio", "neutral_ratio"]
    ].copy()

    out = freq.merge(fol, on="kol", how="inner")
    out = out.merge(sil, on="kol", how="inner")
    out = out.merge(rev, on="kol", how="inner")
    out = out.merge(imb, on="kol", how="inner")
    out.insert(0, "source", source)
    return out


def build_distribution_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for src, sub in df.groupby("source", sort=True):
        for m in metrics:
            x = pd.to_numeric(sub[m], errors="coerce").dropna().to_numpy(dtype=float)
            if len(x) == 0:
                continue
            rows.append(
                {
                    "source": src,
                    "metric": m,
                    "count": int(len(x)),
                    "mean": float(np.mean(x)),
                    "median": float(np.median(x)),
                    "p25": float(np.quantile(x, 0.25)),
                    "p75": float(np.quantile(x, 0.75)),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                }
            )
    return pd.DataFrame(rows)


def build_ci_table(
    df: pd.DataFrame,
    metrics: List[str],
    iters: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for src, sub in df.groupby("source", sort=True):
        for m in metrics:
            x = pd.to_numeric(sub[m], errors="coerce").dropna().to_numpy(dtype=float)
            ci = bootstrap_mean_ci(x, iters=iters, rng=rng)
            rows.append(
                {
                    "source": src,
                    "metric": m,
                    "mean": ci["mean"],
                    "ci95_low": ci["ci_low"],
                    "ci95_high": ci["ci_high"],
                }
            )
    return pd.DataFrame(rows)


def build_separability_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    y = df[df["source"] == "youtube"]
    x = df[df["source"] == "x"]
    rows = []
    for m in metrics:
        ya = pd.to_numeric(y[m], errors="coerce").dropna().to_numpy(dtype=float)
        xb = pd.to_numeric(x[m], errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            {
                "metric": m,
                "youtube_mean": float(np.mean(ya)) if len(ya) else np.nan,
                "x_mean": float(np.mean(xb)) if len(xb) else np.nan,
                "mean_gap_x_minus_youtube": (
                    float(np.mean(xb) - np.mean(ya)) if len(ya) and len(xb) else np.nan
                ),
                "cohen_d_x_vs_youtube": cohen_d(xb, ya),
                "cliffs_delta_x_vs_youtube": cliffs_delta(xb, ya),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_cohen_d"] = out["cohen_d_x_vs_youtube"].abs()
    out = out.sort_values("abs_cohen_d", ascending=False)
    return out


def load_replay_summary(sig_root: Path) -> pd.DataFrame:
    rows = []
    for src in ["youtube", "x"]:
        summary_path = sig_root / src / "distribution_summary.json"
        overall_path = sig_root / src / "overall_counts.csv"
        if not summary_path.exists() or not overall_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        overall = pd.read_csv(overall_path)
        overall = overall[overall["scope"] == "ALL"].iloc[0]
        rows.append(
            {
                "source": src,
                "d_sig": int(overall["d_sig"]),
                "d_sil": int(overall["d_sil"]),
                "d_total": int(overall["d_total"]),
                "rho_sig": float(overall["rho_sig"]),
                "rho_sil": float(overall["rho_sil"]),
                "sil_sig_ratio": float(overall["sil_sig_ratio"]),
                "num_tickers": int(summary["num_tickers"]),
                "pct_tickers_sil_gt_sig": float(summary["pct_tickers_sil_gt_sig"]),
                "ratio_median_finite": float(summary["ratio_median_finite"]),
                "ratio_p25_finite": float(summary["ratio_p25_finite"]),
                "ratio_p75_finite": float(summary["ratio_p75_finite"]),
            }
        )
    return pd.DataFrame(rows).sort_values("source")


def build_markdown(
    out_dir: Path,
    per_kol: pd.DataFrame,
    dist: pd.DataFrame,
    ci: pd.DataFrame,
    sep: pd.DataFrame,
    replay: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Selected-20 Enhanced Statistics Pack")
    lines.append("")
    lines.append("This file summarizes candidate statistics for paper inclusion.")
    lines.append("")

    lines.append("## 1) Coverage")
    lines.append("")
    lines.append(f"- Sources: {', '.join(sorted(per_kol['source'].unique().tolist()))}")
    for src, sub in per_kol.groupby("source"):
        lines.append(f"- `{src}` KOL count: {sub['kol'].nunique()}")
    lines.append("")

    lines.append("## 2) Per-KOL Distribution Summary (Median [P25, P75])")
    lines.append("")
    key_metrics = [
        "mentioned_once_pct",
        "no_followup_within_window_pct",
        "gap_p90_days",
        "reversal_rate",
        "ttf_median_days",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
    ]
    metric_name = {
        "mentioned_once_pct": "single_mention_ratio",
        "no_followup_within_window_pct": "no_followup_within_90d",
        "gap_p90_days": "silence_p90_days",
        "reversal_rate": "sentiment_reversal_rate",
        "ttf_median_days": "median_time_to_first_reversal_days",
        "positive_ratio": "positive_ratio",
        "negative_ratio": "negative_ratio",
        "neutral_ratio": "neutral_ratio",
    }
    lines.append("| Metric | YouTube median [p25,p75] | X median [p25,p75] |")
    lines.append("|---|---:|---:|")
    for m in key_metrics:
        ry = dist[(dist["source"] == "youtube") & (dist["metric"] == m)]
        rx = dist[(dist["source"] == "x") & (dist["metric"] == m)]
        if ry.empty or rx.empty:
            continue
        y = ry.iloc[0]
        x = rx.iloc[0]
        lines.append(
            f"| {metric_name[m]} | {y['median']:.4f} [{y['p25']:.4f}, {y['p75']:.4f}] | "
            f"{x['median']:.4f} [{x['p25']:.4f}, {x['p75']:.4f}] |"
        )
    lines.append("")

    lines.append("## 3) Mean Metric 95% Bootstrap CI (over KOLs)")
    lines.append("")
    lines.append("| Metric | YouTube mean [95% CI] | X mean [95% CI] |")
    lines.append("|---|---:|---:|")
    for m in key_metrics:
        cy = ci[(ci["source"] == "youtube") & (ci["metric"] == m)]
        cx = ci[(ci["source"] == "x") & (ci["metric"] == m)]
        if cy.empty or cx.empty:
            continue
        y = cy.iloc[0]
        x = cx.iloc[0]
        lines.append(
            f"| {metric_name[m]} | {y['mean']:.4f} [{y['ci95_low']:.4f}, {y['ci95_high']:.4f}] | "
            f"{x['mean']:.4f} [{x['ci95_low']:.4f}, {x['ci95_high']:.4f}] |"
        )
    lines.append("")

    lines.append("## 4) Source Separability Ranking")
    lines.append("")
    lines.append("| Metric | mean_gap (X - YouTube) | Cohen's d | Cliff's delta |")
    lines.append("|---|---:|---:|---:|")
    for _, r in sep.iterrows():
        lines.append(
            f"| {metric_name.get(r['metric'], r['metric'])} | {r['mean_gap_x_minus_youtube']:.4f} | "
            f"{r['cohen_d_x_vs_youtube']:.4f} | {r['cliffs_delta_x_vs_youtube']:.4f} |"
        )
    lines.append("")

    lines.append("## 5) Replay Signal/Silence by Source")
    lines.append("")
    lines.append("| Source | \\|D_sig\\| | \\|D_sil\\| | \\|D_sil\\|/\\|D_sig\\| | pct_tickers(\\|D_sil\\|>\\|D_sig\\|) |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in replay.iterrows():
        lines.append(
            f"| {r['source']} | {int(r['d_sig'])} | {int(r['d_sil'])} | {r['sil_sig_ratio']:.4f} | "
            f"{r['pct_tickers_sil_gt_sig']:.4f} |"
        )
    lines.append("")

    lines.append("## 6) Candidate High-Value Metrics For Main Text")
    lines.append("")
    lines.append("- Structural sparsity: `single_mention_ratio`, `no_followup_within_90d`, `silence_p90_days`")
    lines.append("- Direction dynamics: `sentiment_reversal_rate`, `median_time_to_first_reversal_days`")
    lines.append("- Replay dominance linkage: `|D_sil|/|D_sig|` and `pct_tickers(|D_sil|>|D_sig|)`")
    lines.append("")

    (out_dir / "enhanced_selected20_stats.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    company_root = Path(args.company_stats_root)
    signal_root = Path(args.signal_stats_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_kol = pd.concat(
        [
            load_per_kol_metrics(company_root, "youtube"),
            load_per_kol_metrics(company_root, "x"),
        ],
        ignore_index=True,
    ).sort_values(["source", "kol"])
    per_kol.to_csv(out_dir / "per_kol_metrics_merged.csv", index=False)

    metrics = [
        "mentioned_once_pct",
        "no_followup_within_window_pct",
        "gap_p90_days",
        "reversal_rate",
        "ttf_median_days",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
    ]
    dist = build_distribution_table(per_kol, metrics)
    dist.to_csv(out_dir / "distribution_summary_per_kol.csv", index=False)

    ci = build_ci_table(per_kol, metrics, iters=int(args.bootstrap_iters), seed=int(args.seed))
    ci.to_csv(out_dir / "bootstrap_ci_per_source.csv", index=False)

    sep = build_separability_table(per_kol, metrics)
    sep.to_csv(out_dir / "source_separability_ranked.csv", index=False)

    replay = load_replay_summary(signal_root)
    replay.to_csv(out_dir / "replay_signal_silence_by_source.csv", index=False)

    build_markdown(out_dir, per_kol, dist, ci, sep, replay)

    meta = {
        "company_stats_root": str(company_root),
        "signal_stats_root": str(signal_root),
        "output_dir": str(out_dir),
        "bootstrap_iters": int(args.bootstrap_iters),
        "seed": int(args.seed),
        "rows_per_kol_merged": int(len(per_kol)),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

