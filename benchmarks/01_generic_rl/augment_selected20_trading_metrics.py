"""Add event-level trading behavior metrics to selected20 benchmark tables.

Adds three metrics (from event positions):
1) turnover
2) rebalance frequency
3) active exposure ratio

And baseline-relative improvements:
- turnover_improve_vs_baseline = baseline_turnover - method_turnover
- rebalance_freq_improve_vs_baseline = baseline_rebalance_freq - method_rebalance_freq
- active_exposure_improve_vs_baseline = baseline_active_exposure - method_active_exposure

Input:
- detailed csv (selected20 all-methods table)
- manifest json with method->root mapping

Output:
- detailed csv with appended columns
- summary csv by source/method with appended mean columns
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


DEDUP_COLS = [
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment selected20 benchmark tables with trading-behavior metrics.")
    p.add_argument(
        "--detailed-csv",
        default="benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed_benchtest.csv",
    )
    p.add_argument(
        "--manifest-json",
        default="benchmarks/compare/meta/compare_manifest_benchtest.json",
        help="Manifest with methods->root mapping.",
    )
    p.add_argument(
        "--detailed-out",
        default="benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed_benchtest_plus_trading.csv",
    )
    p.add_argument(
        "--summary-out",
        default="benchmarks/compare/meta/selected20_method_summary_vs_baseline_benchtest_plus_trading.csv",
    )
    p.add_argument(
        "--rebalance-eps",
        type=float,
        default=1e-6,
        help="Turnover threshold for counting a day as rebalance.",
    )
    p.add_argument(
        "--active-eps",
        type=float,
        default=1e-2,
        help="Gross exposure threshold for counting a day as active.",
    )
    return p.parse_args()


def read_manifest(path: Path) -> Dict[str, Path]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    methods = obj.get("methods", {})
    return {k: Path(v) for k, v in methods.items()}


def find_run_dir(method_root: Path, source: str, run_name: str) -> Optional[Path]:
    candidates = [
        method_root / source / run_name,
        method_root / source / source / run_name,
        method_root / run_name,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def positions_path_for_row(method_roots: Dict[str, Path], source: str, method: str, run_name: str) -> Optional[Path]:
    if method == "BASELINE":
        return None
    root = method_roots.get(method)
    if root is None:
        return None
    run_dir = find_run_dir(root, source, run_name)
    if run_dir is None:
        return None
    p = run_dir / "event" / "positions_test.csv"
    if p.exists():
        return p
    return None


def read_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [c for c in DEDUP_COLS if c in df.columns]
    extra = [c for c in ["baseline_action", "policy_action", "weight"] if c in df.columns and c not in keep]
    cols = keep + extra
    if cols:
        df = df[cols].drop_duplicates()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def _compute_behavior_from_column(
    df: pd.DataFrame,
    action_col: str,
    rebalance_eps: float,
    active_eps: float,
) -> Dict[str, float]:
    if action_col not in df.columns:
        return {
            "turnover": float("nan"),
            "rebalance_freq": float("nan"),
            "active_exposure_ratio": float("nan"),
        }
    tmp = df[["date", "ticker", action_col]].copy()
    tmp[action_col] = pd.to_numeric(tmp[action_col], errors="coerce").fillna(0.0)

    w = (
        tmp.pivot_table(index="date", columns="ticker", values=action_col, aggfunc="last")
        .sort_index()
        .fillna(0.0)
    )
    if w.empty:
        return {
            "turnover": float("nan"),
            "rebalance_freq": float("nan"),
            "active_exposure_ratio": float("nan"),
        }

    gross = w.abs().sum(axis=1)
    diff = w.diff().abs().sum(axis=1)
    if len(diff) > 1:
        trans = diff.iloc[1:]
        turnover = float(trans.mean())
        rebalance_freq = float((trans > rebalance_eps).mean())
    else:
        turnover = 0.0
        rebalance_freq = 0.0
    aer = float((gross > active_eps).mean())

    return {
        "turnover": turnover,
        "rebalance_freq": rebalance_freq,
        "active_exposure_ratio": aer,
    }


def main() -> None:
    args = parse_args()
    detailed_csv = Path(args.detailed_csv)
    manifest_json = Path(args.manifest_json)
    detailed_out = Path(args.detailed_out)
    summary_out = Path(args.summary_out)

    if not detailed_csv.exists():
        raise FileNotFoundError(f"detailed csv not found: {detailed_csv}")

    df = pd.read_csv(detailed_csv)
    required = {"source", "kol", "method", "run_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"detailed csv missing columns: {missing}")

    method_roots = read_manifest(manifest_json)

    # Cache baseline behavior per (source, kol), extracted from any available positions file
    baseline_cache: Dict[Tuple[str, str], Dict[str, float]] = {}

    for _, row in df[df["method"] != "BASELINE"].iterrows():
        key = (str(row["source"]), str(row["kol"]))
        if key in baseline_cache:
            continue
        p = positions_path_for_row(
            method_roots=method_roots,
            source=str(row["source"]),
            method=str(row["method"]),
            run_name=str(row["run_name"]),
        )
        if p is None:
            continue
        pos = read_positions(p)
        baseline_cache[key] = _compute_behavior_from_column(
            pos,
            action_col="baseline_action",
            rebalance_eps=args.rebalance_eps,
            active_eps=args.active_eps,
        )

    # Compute method behavior + baseline references
    out_rows = []
    missing_positions = 0
    for _, row in df.iterrows():
        row_out = row.to_dict()
        source = str(row["source"])
        kol = str(row["kol"])
        method = str(row["method"])
        run_name = str(row["run_name"])
        key = (source, kol)

        if method == "BASELINE":
            b = baseline_cache.get(
                key,
                {"turnover": float("nan"), "rebalance_freq": float("nan"), "active_exposure_ratio": float("nan")},
            )
            row_out["event_turnover"] = b["turnover"]
            row_out["event_rebalance_freq"] = b["rebalance_freq"]
            row_out["event_active_exposure_ratio"] = b["active_exposure_ratio"]
            row_out["event_baseline_turnover"] = b["turnover"]
            row_out["event_baseline_rebalance_freq"] = b["rebalance_freq"]
            row_out["event_baseline_active_exposure_ratio"] = b["active_exposure_ratio"]
            row_out["turnover_improve_vs_baseline"] = 0.0
            row_out["rebalance_freq_improve_vs_baseline"] = 0.0
            row_out["active_exposure_improve_vs_baseline"] = 0.0
            out_rows.append(row_out)
            continue

        p = positions_path_for_row(
            method_roots=method_roots,
            source=source,
            method=method,
            run_name=run_name,
        )
        if p is None:
            missing_positions += 1
            m = {"turnover": float("nan"), "rebalance_freq": float("nan"), "active_exposure_ratio": float("nan")}
            b = baseline_cache.get(
                key,
                {"turnover": float("nan"), "rebalance_freq": float("nan"), "active_exposure_ratio": float("nan")},
            )
        else:
            pos = read_positions(p)
            m = _compute_behavior_from_column(
                pos,
                action_col="weight",
                rebalance_eps=args.rebalance_eps,
                active_eps=args.active_eps,
            )
            b = _compute_behavior_from_column(
                pos,
                action_col="baseline_action",
                rebalance_eps=args.rebalance_eps,
                active_eps=args.active_eps,
            )
            # Prefer stable per-kol baseline cache once observed
            if key not in baseline_cache:
                baseline_cache[key] = b
            else:
                b = baseline_cache[key]

        row_out["event_turnover"] = m["turnover"]
        row_out["event_rebalance_freq"] = m["rebalance_freq"]
        row_out["event_active_exposure_ratio"] = m["active_exposure_ratio"]
        row_out["event_baseline_turnover"] = b["turnover"]
        row_out["event_baseline_rebalance_freq"] = b["rebalance_freq"]
        row_out["event_baseline_active_exposure_ratio"] = b["active_exposure_ratio"]
        row_out["turnover_improve_vs_baseline"] = b["turnover"] - m["turnover"]
        row_out["rebalance_freq_improve_vs_baseline"] = b["rebalance_freq"] - m["rebalance_freq"]
        row_out["active_exposure_improve_vs_baseline"] = b["active_exposure_ratio"] - m["active_exposure_ratio"]
        out_rows.append(row_out)

    out_df = pd.DataFrame(out_rows)
    detailed_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(detailed_out, index=False)

    # Rebuild source/method summary (aligned with existing table style + new columns)
    grp = out_df.groupby(["source", "method"], as_index=False)
    summary = grp.agg(
        n_kols=("kol", "nunique"),
        mean_event_return=("event_cumulative_return", "mean"),
        mean_event_sharpe=("event_sharpe", "mean"),
        mean_event_mdd=("event_max_drawdown", "mean"),
        mean_daily_return=("daily_trained_cumulative_return", "mean"),
        mean_daily_sharpe=("daily_trained_sharpe", "mean"),
        mean_daily_mdd=("daily_trained_max_drawdown", "mean"),
        mean_uplift_return=("cumret_uplift_vs_baseline", "mean"),
        mean_uplift_sharpe=("sharpe_uplift_vs_baseline", "mean"),
        mean_mdd_improve=("mdd_improvement_vs_baseline", "mean"),
        win_rate_vs_baseline=("cumret_uplift_vs_baseline", lambda x: float((x > 0).mean())),
        mean_event_turnover=("event_turnover", "mean"),
        mean_event_rebalance_freq=("event_rebalance_freq", "mean"),
        mean_event_active_exposure_ratio=("event_active_exposure_ratio", "mean"),
        mean_turnover_improve=("turnover_improve_vs_baseline", "mean"),
        mean_rebalance_freq_improve=("rebalance_freq_improve_vs_baseline", "mean"),
        mean_active_exposure_improve=("active_exposure_improve_vs_baseline", "mean"),
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)

    print(f"Saved detailed: {detailed_out}")
    print(f"Saved summary : {summary_out}")
    print(f"Rows: {len(out_df)} | Missing positions rows: {missing_positions}")


if __name__ == "__main__":
    main()
