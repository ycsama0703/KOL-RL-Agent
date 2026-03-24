#!/usr/bin/env python3
"""Build trace-level evidence tables for case study.

Evidence chain per case:
KOL discourse row (title/text/ticker/sentiment) -> policy action -> ticker contribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case-root",
        type=Path,
        default=Path("benchmarks/compare/case_study"),
    )
    p.add_argument(
        "--x-reward-root",
        type=Path,
        default=Path("data/multisource_ready_22-25_xrefresh_20260320_144701/07_reward/x"),
    )
    p.add_argument(
        "--youtube-reward-root",
        type=Path,
        default=Path("data/multisource_ready_22-25/07_reward/youtube"),
    )
    p.add_argument("--top-days", type=int, default=8)
    p.add_argument("--top-tickers-per-day", type=int, default=3)
    return p.parse_args()


def _to_day(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date.astype(str)


def _load_reward_df(source: str, kol: str, x_root: Path, yt_root: Path) -> pd.DataFrame:
    base = x_root if source == "x" else yt_root
    path = base / kol / "test.csv"
    if not path.exists():
        raise FileNotFoundError(f"reward csv not found: {path}")
    df = pd.read_csv(path)
    keep = [c for c in ["trading_day", "ticker", "title", "text", "sentiment", "confidence", "event_id", "published_at"] if c in df.columns]
    out = df[keep].copy()
    out["day"] = _to_day(out["trading_day"])
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def _build_case_trace(
    case_root: Path,
    source: str,
    kol: str,
    reward_df: pd.DataFrame,
    top_days: int,
    top_tickers_per_day: int,
) -> Dict[str, pd.DataFrame]:
    pos_path = case_root / "raw_kicl" / source / kol / "positions_test.csv"
    if not pos_path.exists():
        raise FileNotFoundError(f"positions not found: {pos_path}")
    pos = pd.read_csv(pos_path)
    pos["day"] = _to_day(pos["date"])
    pos["ticker"] = pos["ticker"].astype(str).str.upper()
    pos["contribution"] = pos["weight"].astype(float) * pos["reward"].astype(float)

    daily = (
        pos.groupby("day", as_index=False)
        .agg(
            daily_contribution=("contribution", "sum"),
            n_rows=("ticker", "size"),
            n_active_tickers=("weight", lambda s: int((s > 0).sum())),
        )
        .sort_values("daily_contribution", ascending=False)
        .head(top_days)
        .reset_index(drop=True)
    )
    daily.insert(0, "source", source)
    daily.insert(1, "kol", kol)

    # top ticker contributions on each top day
    top_rows: List[pd.DataFrame] = []
    for d in daily["day"].tolist():
        sub = pos[pos["day"] == d].copy()
        sub = sub.sort_values("contribution", ascending=False).head(top_tickers_per_day)
        if sub.empty:
            continue
        sub.insert(0, "source", source)
        sub.insert(1, "kol", kol)
        sub.insert(2, "rank_in_day", range(1, len(sub) + 1))
        top_rows.append(
            sub[
                [
                    "source",
                    "kol",
                    "day",
                    "rank_in_day",
                    "ticker",
                    "action",
                    "baseline_action",
                    "policy_action",
                    "weight",
                    "reward",
                    "contribution",
                ]
            ]
        )

    ticker_top = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()

    # join discourse evidence
    if not ticker_top.empty:
        merged = ticker_top.merge(
            reward_df,
            left_on=["day", "ticker"],
            right_on=["day", "ticker"],
            how="left",
        )
        merged["title"] = merged.get("title", pd.Series([""] * len(merged))).fillna("")
        merged["text"] = merged.get("text", pd.Series([""] * len(merged))).fillna("")
        merged["text_preview"] = merged["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 220)
        merged = merged.sort_values(["day", "rank_in_day", "confidence"], ascending=[False, True, False])
    else:
        merged = pd.DataFrame()

    return {"daily_top": daily, "ticker_top": ticker_top, "evidence": merged}


def _case_md(source: str, kol: str, daily: pd.DataFrame, evidence: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append(f"# Case Trace: {source}/{kol}")
    lines.append("")
    if daily.empty:
        lines.append("No daily contribution rows found.")
        return "\n".join(lines) + "\n"

    lines.append("## Top Contribution Days (KICL)")
    for _, r in daily.head(5).iterrows():
        lines.append(
            f"- {r['day']}: daily_contribution={r['daily_contribution']:.4f}, "
            f"active_tickers={int(r['n_active_tickers'])}, rows={int(r['n_rows'])}"
        )
    lines.append("")

    lines.append("## Evidence Snippets (day + ticker)")
    if evidence.empty:
        lines.append("- No matched discourse rows found from reward csv.")
    else:
        used = set()
        for _, r in evidence.iterrows():
            key = (r["day"], r["ticker"], r["rank_in_day"])
            if key in used:
                continue
            used.add(key)
            lines.append(
                f"- {r['day']} | rank#{int(r['rank_in_day'])} | {r['ticker']} | "
                f"action={r.get('action','')} | contrib={float(r['contribution']):.4f} | "
                f"baseline={float(r['baseline_action']):.4f} -> policy={float(r['policy_action']):.4f}"
            )
            t = str(r.get("title", "")).strip()
            if t:
                lines.append(f"  title: {t}")
            prev = str(r.get("text_preview", "")).strip()
            if prev:
                lines.append(f"  text: {prev}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    case_root = args.case_root
    cases = pd.read_csv(case_root / "case_study_selected_kols_summary.csv")[["source", "kol"]]

    out_root = case_root / "trace"
    out_root.mkdir(parents=True, exist_ok=True)

    all_daily: List[pd.DataFrame] = []
    all_ticker: List[pd.DataFrame] = []
    all_evi: List[pd.DataFrame] = []

    for _, row in cases.iterrows():
        source = row["source"]
        kol = row["kol"]
        reward_df = _load_reward_df(source, kol, args.x_reward_root, args.youtube_reward_root)
        out = _build_case_trace(
            case_root=case_root,
            source=source,
            kol=kol,
            reward_df=reward_df,
            top_days=args.top_days,
            top_tickers_per_day=args.top_tickers_per_day,
        )
        case_dir = out_root / source / kol
        case_dir.mkdir(parents=True, exist_ok=True)

        out["daily_top"].to_csv(case_dir / "kicl_top_days_by_contribution.csv", index=False)
        out["ticker_top"].to_csv(case_dir / "kicl_top_tickers_on_top_days.csv", index=False)
        out["evidence"].to_csv(case_dir / "kicl_discourse_evidence.csv", index=False)
        (case_dir / "CASE_TRACE.md").write_text(
            _case_md(source, kol, out["daily_top"], out["evidence"]), encoding="utf-8"
        )

        all_daily.append(out["daily_top"])
        all_ticker.append(out["ticker_top"])
        all_evi.append(out["evidence"])

    pd.concat(all_daily, ignore_index=True).to_csv(out_root / "all_cases_top_days.csv", index=False)
    pd.concat(all_ticker, ignore_index=True).to_csv(out_root / "all_cases_top_tickers.csv", index=False)
    pd.concat(all_evi, ignore_index=True).to_csv(out_root / "all_cases_discourse_evidence.csv", index=False)

    print(f"Saved trace folder: {out_root}")


if __name__ == "__main__":
    main()

