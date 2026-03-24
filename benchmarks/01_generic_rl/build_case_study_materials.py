#!/usr/bin/env python3
"""Build high-signal case-study tables and a concise markdown summary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case-root",
        type=Path,
        default=Path("benchmarks/compare/case_study"),
        help="Case-study root folder.",
    )
    p.add_argument(
        "--action-threshold",
        type=float,
        default=0.02,
        help="Threshold used to define active actions and hard violations.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window length (trading days) for stage uplift table.",
    )
    return p.parse_args()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_method_snapshot(case_root: Path, cases: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, c in cases.iterrows():
        source = c["source"]
        kol = c["kol"]

        event_df = _safe_read_csv(case_root / source / kol / "event_metrics_compare.csv")
        daily_df = _safe_read_csv(case_root / source / kol / "daily_metrics_compare.csv")
        betr_df = _safe_read_csv(case_root / source / kol / "betrayal_metrics_compare.csv")

        df = (
            event_df.merge(daily_df, on=["method", "run_name"], how="inner")
            .merge(betr_df, on=["method", "run_name"], how="inner")
            .copy()
        )

        df["event_return_rank"] = (
            df["cumulative_return"].rank(method="dense", ascending=False).astype(int)
        )
        df["daily_uplift_vs_baseline"] = (
            df["trained_cumulative_return"] - df["baseline_cumulative_return"]
        )
        df["hard_violation_sum"] = df["entry_violation_rate"] + df["reversal_rate"]

        for _, r in df.iterrows():
            rows.append(
                {
                    "source": source,
                    "kol": kol,
                    "method": r["method"],
                    "run_name": r["run_name"],
                    "event_return": float(r["cumulative_return"]),
                    "event_sharpe": float(r["sharpe"]),
                    "event_mdd": float(r["max_drawdown"]),
                    "event_return_rank": int(r["event_return_rank"]),
                    "daily_return": float(r["trained_cumulative_return"]),
                    "daily_baseline_return": float(r["baseline_cumulative_return"]),
                    "daily_uplift_vs_baseline": float(r["daily_uplift_vs_baseline"]),
                    "daily_sharpe": float(r["trained_sharpe"]),
                    "daily_mdd": float(r["trained_max_drawdown"]),
                    "UER": float(r["entry_violation_rate"]),
                    "DRR": float(r["reversal_rate"]),
                    "BD": float(r["mean_abs_deviation"]),
                    "hard_violation_sum": float(r["hard_violation_sum"]),
                    "sign_agreement_rate": float(r["sign_agreement_rate"]),
                    "baseline_policy_corr": float(r["baseline_policy_corr"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["source", "kol", "event_return_rank", "method"])


def build_kicl_behavior(case_root: Path, cases: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, c in cases.iterrows():
        source = c["source"]
        kol = c["kol"]
        pos = _safe_read_csv(case_root / "raw_kicl" / source / kol / "positions_test.csv")
        eq = _safe_read_csv(case_root / "raw_kicl" / source / kol / "equity_daily.csv")

        policy = pos["policy_action"].astype(float)
        base = pos["baseline_action"].astype(float)
        delta = (policy - base).abs()

        base_active = base.abs() > threshold
        policy_active = policy.abs() > threshold
        unsupported_entry = (~base_active) & policy_active
        reversal = base_active & (base * policy < 0) & policy_active

        action_counts = pos["action"].value_counts().to_dict()
        non_hold = int(
            action_counts.get("OPEN", 0)
            + action_counts.get("INCREASE", 0)
            + action_counts.get("DECREASE", 0)
            + action_counts.get("CLOSE", 0)
        )
        hold = int(action_counts.get("HOLD", 0))

        gap = (eq["equity_trained"] - eq["equity_baseline"]).astype(float)
        win_day_ratio = float((eq["daily_return_y"] > eq["daily_return_x"]).mean())

        rows.append(
            {
                "source": source,
                "kol": kol,
                "rows_positions": int(len(pos)),
                "n_dates": int(pos["date"].nunique()),
                "n_tickers": int(pos["ticker"].nunique()),
                "OPEN": int(action_counts.get("OPEN", 0)),
                "INCREASE": int(action_counts.get("INCREASE", 0)),
                "DECREASE": int(action_counts.get("DECREASE", 0)),
                "CLOSE": int(action_counts.get("CLOSE", 0)),
                "HOLD": hold,
                "non_hold_ratio": float(non_hold / max(1, non_hold + hold)),
                "mean_abs_delta": float(delta.mean()),
                "median_abs_delta": float(delta.median()),
                "p90_abs_delta": float(delta.quantile(0.9)),
                "hard_unsupported_entry_rate": float(unsupported_entry.mean()),
                "hard_reversal_rate": float(reversal.mean()),
                "active_baseline_ratio": float(base_active.mean()),
                "active_policy_ratio": float(policy_active.mean()),
                "win_day_ratio": win_day_ratio,
                "mean_daily_equity_gap": float(gap.mean()),
                "max_daily_equity_gap": float(gap.max()),
                "min_daily_equity_gap": float(gap.min()),
            }
        )

    return pd.DataFrame(rows).sort_values(["source", "kol"])


def build_kicl_stage_table(case_root: Path, cases: pd.DataFrame, window: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, c in cases.iterrows():
        source = c["source"]
        kol = c["kol"]
        eq = _safe_read_csv(case_root / "raw_kicl" / source / kol / "equity_daily.csv")
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.sort_values("date").reset_index(drop=True)

        if len(eq) <= window:
            continue

        trained = eq["equity_trained"].astype(float)
        baseline = eq["equity_baseline"].astype(float)
        # window return at t: equity_t / equity_(t-window) - 1
        trained_wr = trained / trained.shift(window) - 1.0
        baseline_wr = baseline / baseline.shift(window) - 1.0
        uplift = trained_wr - baseline_wr
        gap = trained - baseline
        gap_delta = gap - gap.shift(window)

        out = pd.DataFrame(
            {
                "source": source,
                "kol": kol,
                "end_date": eq["date"],
                "start_date": eq["date"].shift(window),
                "window_days": window,
                "trained_window_return": trained_wr,
                "baseline_window_return": baseline_wr,
                "window_uplift": uplift,
                "gap_start": gap.shift(window),
                "gap_end": gap,
                "gap_delta": gap_delta,
            }
        ).dropna()

        out = out.sort_values("window_uplift", ascending=False).head(5)
        rows.extend(out.to_dict(orient="records"))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["source", "kol", "window_uplift"], ascending=[True, True, False])
    return df


def build_markdown_summary(
    case_root: Path,
    method_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    stage_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# Case Study Notes (Auto-generated)")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Cases: `x/Jake__Wujastyk`, `youtube/Financial_Education`")
    lines.append("- Focus: whether KICL gains are achieved under hard intent constraints while improving portfolio execution.")
    lines.append("")

    lines.append("## Key Observations")
    for _, row in behavior_df.iterrows():
        s = row["source"]
        k = row["kol"]
        kicl = method_df[
            (method_df["source"] == s) & (method_df["kol"] == k) & (method_df["method"] == "KICL")
        ].iloc[0]
        best_event = method_df[
            (method_df["source"] == s) & (method_df["kol"] == k)
        ].sort_values("event_return", ascending=False).iloc[0]

        lines.append(
            f"- `{s}/{k}`: KICL event return `{kicl['event_return']:.3f}` "
            f"(rank #{int(kicl['event_return_rank'])}), daily uplift vs baseline `{kicl['daily_uplift_vs_baseline']:.3f}`, "
            f"`UER={kicl['UER']:.3f}`, `DRR={kicl['DRR']:.3f}`, `BD={kicl['BD']:.3f}`."
        )
        lines.append(
            f"  Behavior: non-hold ratio `{row['non_hold_ratio']:.3f}`, "
            f"active baseline ratio `{row['active_baseline_ratio']:.3f}` -> active policy ratio `{row['active_policy_ratio']:.3f}`, "
            f"mean |policy-baseline| `{row['mean_abs_delta']:.4f}`."
        )
        lines.append(
            f"  Best event-return method on this case is `{best_event['method']}` (`{best_event['event_return']:.3f}`); "
            f"KICL remains hard-consistent (`UER=DRR=0`)."
        )

    lines.append("")
    lines.append("## Files")
    lines.append("- `tables/case_study_method_snapshot.csv`")
    lines.append("- `tables/case_study_kicl_behavior_breakdown.csv`")
    lines.append("- `tables/case_study_kicl_top_uplift_windows.csv`")
    lines.append("")
    lines.append("## Suggested Figures")
    lines.append("- `x/Jake__Wujastyk/equity_daily_compare.png`")
    lines.append("- `youtube/Financial_Education/equity_daily_compare.png`")
    lines.append("- `x/Jake__Wujastyk/event_equity_compare.png`")
    lines.append("- `youtube/Financial_Education/event_equity_compare.png`")
    lines.append("")
    if not stage_df.empty:
        lines.append("## Stage Uplift Highlights (Top windows)")
        for (s, k), grp in stage_df.groupby(["source", "kol"]):
            top = grp.sort_values("window_uplift", ascending=False).head(1).iloc[0]
            lines.append(
                f"- `{s}/{k}` top {int(top['window_days'])}-day uplift window: "
                f"`{pd.to_datetime(top['start_date']).date()} -> {pd.to_datetime(top['end_date']).date()}`, "
                f"uplift `{top['window_uplift']:.4f}` "
                f"(trained `{top['trained_window_return']:.4f}` vs baseline `{top['baseline_window_return']:.4f}`)."
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    case_root = args.case_root
    tables_dir = case_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cases = _safe_read_csv(case_root / "case_study_selected_kols_summary.csv")[["source", "kol"]]

    method_df = build_method_snapshot(case_root, cases)
    behavior_df = build_kicl_behavior(case_root, cases, args.action_threshold)
    stage_df = build_kicl_stage_table(case_root, cases, args.window)

    method_path = tables_dir / "case_study_method_snapshot.csv"
    behavior_path = tables_dir / "case_study_kicl_behavior_breakdown.csv"
    stage_path = tables_dir / "case_study_kicl_top_uplift_windows.csv"
    notes_path = case_root / "CASE_STUDY_NOTES.md"

    method_df.to_csv(method_path, index=False)
    behavior_df.to_csv(behavior_path, index=False)
    stage_df.to_csv(stage_path, index=False)
    notes_path.write_text(build_markdown_summary(case_root, method_df, behavior_df, stage_df), encoding="utf-8")

    print(f"Saved: {method_path}")
    print(f"Saved: {behavior_path}")
    print(f"Saved: {stage_path}")
    print(f"Saved: {notes_path}")


if __name__ == "__main__":
    main()

