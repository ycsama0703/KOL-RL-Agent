#!/usr/bin/env python3
"""Assemble one canonical compare folder for all YouTube + X KOL test results.

Inputs (default):
  - benchmarks/compare/youtube
  - benchmarks/compare/xrefresh/x

Output (default):
  - benchmarks/compare/canonical_all
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


METHOD_ORDER = ["KICL", "BC", "IQL", "TD3BC", "CQL", "AWAC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--youtube-root",
        type=Path,
        default=Path("benchmarks/compare/youtube"),
        help="Per-KOL compare results for YouTube source",
    )
    parser.add_argument(
        "--x-root",
        type=Path,
        default=Path("benchmarks/compare/xrefresh/x"),
        help="Per-KOL compare results for X source",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/compare/canonical_all"),
        help="Canonical merged output root",
    )
    return parser.parse_args()


def method_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def ordered_methods(methods: Iterable[str]) -> List[str]:
    present = set(methods)
    ordered = [m for m in METHOD_ORDER if m in present]
    extras = sorted(m for m in present if m not in METHOD_ORDER)
    return ordered + extras


def copy_source_tree(src_root: Path, dst_root: Path) -> List[str]:
    dst_root.mkdir(parents=True, exist_ok=True)
    kols: List[str] = []
    if not src_root.exists():
        return kols
    for kol_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        kols.append(kol_dir.name)
        target = dst_root / kol_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for fp in kol_dir.iterdir():
            if fp.is_file():
                shutil.copy2(fp, target / fp.name)
    return kols


def read_metrics(kol_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_df = pd.read_csv(kol_dir / "event_metrics_compare.csv")
    daily_df = pd.read_csv(kol_dir / "daily_metrics_compare.csv")
    betrayal_df = pd.read_csv(kol_dir / "betrayal_metrics_compare.csv")
    return event_df, daily_df, betrayal_df


def build_summary_by_kol(canonical_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for source in ["youtube", "x"]:
        source_root = canonical_root / source
        if not source_root.exists():
            continue
        for kol_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
            event_df, daily_df, betrayal_df = read_metrics(kol_dir)
            row: Dict[str, float] = {"source": source, "kol": kol_dir.name}

            for _, r in event_df.iterrows():
                m = method_key(str(r["method"]))
                row[f"{m}_event_cumulative_return"] = float(r["cumulative_return"])
                row[f"{m}_event_sharpe"] = float(r["sharpe"])
                row[f"{m}_event_max_drawdown"] = float(r["max_drawdown"])

            for _, r in betrayal_df.iterrows():
                m = method_key(str(r["method"]))
                row[f"{m}_betrayal_reversal_rate"] = float(r["reversal_rate"])
                row[f"{m}_betrayal_entry_violation_rate"] = float(r["entry_violation_rate"])
                row[f"{m}_betrayal_mean_abs_deviation"] = float(r["mean_abs_deviation"])
                row[f"{m}_betrayal_mean_normalized_deviation"] = float(
                    r["mean_normalized_deviation"]
                )
                row[f"{m}_betrayal_sign_agreement_rate"] = float(r["sign_agreement_rate"])
                row[f"{m}_betrayal_baseline_policy_corr"] = float(r["baseline_policy_corr"])

            for _, r in daily_df.iterrows():
                m = method_key(str(r["method"]))
                row[f"{m}_daily_trained_cumulative_return"] = float(
                    r["trained_cumulative_return"]
                )
                row[f"{m}_daily_baseline_cumulative_return"] = float(
                    r["baseline_cumulative_return"]
                )
                row[f"{m}_daily_trained_sharpe"] = float(r["trained_sharpe"])
                row[f"{m}_daily_baseline_sharpe"] = float(r["baseline_sharpe"])
                row[f"{m}_daily_trained_max_drawdown"] = float(r["trained_max_drawdown"])
                row[f"{m}_daily_baseline_max_drawdown"] = float(r["baseline_max_drawdown"])

            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["source", "kol"])
    return pd.DataFrame(rows).sort_values(["source", "kol"]).reset_index(drop=True)


def mean_or_nan(df: pd.DataFrame, cols: List[str]) -> float:
    present = [c for c in cols if c in df.columns]
    if not present:
        return float("nan")
    return float(df[present].mean(axis=0).mean())


def aggregate_method_table(summary_by_kol: pd.DataFrame) -> pd.DataFrame:
    if summary_by_kol.empty:
        return pd.DataFrame()

    method_names = {
        "kicl": "KICL",
        "bc": "BC",
        "iql": "IQL",
        "td3bc": "TD3BC",
        "cql": "CQL",
        "awac": "AWAC",
    }
    rows: List[Dict[str, float]] = []

    for source in sorted(summary_by_kol["source"].unique()):
        sdf = summary_by_kol[summary_by_kol["source"] == source]
        for key, method in method_names.items():
            row = {
                "source": source,
                "method": method,
                "n_kols": int(len(sdf)),
                "event_mean_cumulative_return": mean_or_nan(
                    sdf, [f"{key}_event_cumulative_return"]
                ),
                "event_mean_sharpe": mean_or_nan(sdf, [f"{key}_event_sharpe"]),
                "event_mean_max_drawdown": mean_or_nan(
                    sdf, [f"{key}_event_max_drawdown"]
                ),
                "betrayal_mean_reversal_rate": mean_or_nan(
                    sdf, [f"{key}_betrayal_reversal_rate"]
                ),
                "betrayal_mean_entry_violation_rate": mean_or_nan(
                    sdf, [f"{key}_betrayal_entry_violation_rate"]
                ),
                "betrayal_mean_mean_abs_deviation": mean_or_nan(
                    sdf, [f"{key}_betrayal_mean_abs_deviation"]
                ),
                "betrayal_mean_mean_normalized_deviation": mean_or_nan(
                    sdf, [f"{key}_betrayal_mean_normalized_deviation"]
                ),
                "betrayal_mean_sign_agreement_rate": mean_or_nan(
                    sdf, [f"{key}_betrayal_sign_agreement_rate"]
                ),
                "betrayal_mean_baseline_policy_corr": mean_or_nan(
                    sdf, [f"{key}_betrayal_baseline_policy_corr"]
                ),
                "daily_trained_mean_cumulative_return": mean_or_nan(
                    sdf, [f"{key}_daily_trained_cumulative_return"]
                ),
                "daily_baseline_mean_cumulative_return": mean_or_nan(
                    sdf, [f"{key}_daily_baseline_cumulative_return"]
                ),
                "daily_trained_mean_sharpe": mean_or_nan(
                    sdf, [f"{key}_daily_trained_sharpe"]
                ),
                "daily_baseline_mean_sharpe": mean_or_nan(
                    sdf, [f"{key}_daily_baseline_sharpe"]
                ),
                "daily_trained_mean_max_drawdown": mean_or_nan(
                    sdf, [f"{key}_daily_trained_max_drawdown"]
                ),
                "daily_baseline_mean_max_drawdown": mean_or_nan(
                    sdf, [f"{key}_daily_baseline_max_drawdown"]
                ),
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    out = out.sort_values(["source", "method"]).reset_index(drop=True)
    return out


def aggregate_overall(by_source_df: pd.DataFrame) -> pd.DataFrame:
    if by_source_df.empty:
        return pd.DataFrame()
    num_cols = [c for c in by_source_df.columns if c not in {"source", "method", "n_kols"}]
    rows: List[Dict[str, float]] = []
    for method in ordered_methods(by_source_df["method"].astype(str).tolist()):
        mdf = by_source_df[by_source_df["method"].astype(str) == method]
        row: Dict[str, float] = {
            "method": method,
            "n_kols": int(mdf["n_kols"].sum()),
        }
        for c in num_cols:
            # weighted by n_kols across sources
            if mdf[c].isna().all():
                row[c] = float("nan")
            else:
                w = mdf["n_kols"].astype(float)
                row[c] = float((mdf[c] * w).sum() / w.sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values("method").reset_index(drop=True)


def write_readme(path: Path, youtube_count: int, x_count: int, total: int) -> None:
    text = "\n".join(
        [
            "# Canonical Test Results (All KOLs)",
            "",
            "This folder is the single source of truth for plotting and tables.",
            "",
            "## Scope",
            f"- youtube KOLs: {youtube_count}",
            f"- x KOLs: {x_count}",
            f"- total KOLs: {total}",
            "",
            "## Structure",
            "- `youtube/<KOL>/...` per-KOL compare files",
            "- `x/<KOL>/...` per-KOL compare files",
            "- `tables/summary_by_kol.csv` merged per-KOL metrics",
            "- `tables/summary_by_method_mean_by_source.csv` method means per source",
            "- `tables/summary_by_method_mean.csv` method means across all sources",
            "- `compare_manifest.json` assembly metadata",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = args.output_root
    (out_root / "tables").mkdir(parents=True, exist_ok=True)

    yt_kols = copy_source_tree(args.youtube_root, out_root / "youtube")
    x_kols = copy_source_tree(args.x_root, out_root / "x")

    by_kol = build_summary_by_kol(out_root)
    by_source = aggregate_method_table(by_kol)
    by_method = aggregate_overall(by_source)

    by_kol.to_csv(out_root / "tables" / "summary_by_kol.csv", index=False)
    by_source.to_csv(out_root / "tables" / "summary_by_method_mean_by_source.csv", index=False)
    by_method.to_csv(out_root / "tables" / "summary_by_method_mean.csv", index=False)

    manifest = {
        "youtube_root": str(args.youtube_root),
        "x_root": str(args.x_root),
        "output_root": str(out_root),
        "n_youtube_kols": len(yt_kols),
        "n_x_kols": len(x_kols),
        "n_total_kols": len(yt_kols) + len(x_kols),
        "youtube_kols": yt_kols,
        "x_kols": x_kols,
    }
    (out_root / "compare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(out_root / "README.md", len(yt_kols), len(x_kols), len(yt_kols) + len(x_kols))

    print(f"Saved canonical folder: {out_root}")
    print(f"YouTube KOLs: {len(yt_kols)}")
    print(f"X KOLs: {len(x_kols)}")
    print(f"Total KOLs: {len(yt_kols) + len(x_kols)}")


if __name__ == "__main__":
    main()

