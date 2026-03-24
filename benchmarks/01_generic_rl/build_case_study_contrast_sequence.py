#!/usr/bin/env python3
"""Build case-study contrast sequences:
KOL discourse -> sentiment -> baseline behavior -> KICL behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


DEFAULT_WINDOWS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("x", "Jake__Wujastyk"): ("2024-11-15", "2024-12-31"),
    ("youtube", "The_Maverick_of_Wall_Street"): ("2024-10-25", "2024-12-20"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case-root",
        type=Path,
        default=Path("benchmarks/compare/case_study"),
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("benchmarks/compare/case_study/case_study_selected_kols_summary.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/compare/case_study/sequence"),
    )
    p.add_argument("--top-days", type=int, default=4, help="Top d_gap days per case.")
    p.add_argument(
        "--top-tickers-per-day",
        type=int,
        default=5,
        help="Top positive-contribution tickers per focus day.",
    )
    p.add_argument(
        "--weight-eps",
        type=float,
        default=0.02,
        help="Near-zero threshold for behavior labels.",
    )
    return p.parse_args()


def _load_csv(path: Path, required_cols: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    miss = required_cols.difference(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")
    return df


def _baseline_behavior(b: float, eps: float) -> str:
    if abs(b) <= eps:
        return "No baseline position"
    if b > 0:
        return f"Baseline long ({b:.3f})"
    return f"Baseline short ({b:.3f})"


def _ours_behavior(action: str, p: float, eps: float) -> str:
    tag = action.upper() if isinstance(action, str) else "UNKNOWN"
    if abs(p) <= eps:
        return f"{tag} -> near flat ({p:.3f})"
    if p > 0:
        return f"{tag} -> long ({p:.3f})"
    return f"{tag} -> short ({p:.3f})"


def _relation_tag(b: float, p: float, eps: float) -> str:
    if abs(b) <= eps and abs(p) <= eps:
        return "Both near flat"
    if abs(b) <= eps and abs(p) > eps:
        return "New entry vs baseline"
    if abs(b) > eps and abs(p) <= eps:
        return "De-risk to flat vs baseline"
    if p > b + eps:
        return "Overweight vs baseline"
    if p < b - eps:
        return "Underweight vs baseline"
    return "Track baseline"


def _build_for_case(
    case_root: Path,
    source: str,
    kol: str,
    top_days: int,
    top_tickers: int,
    eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_root = case_root / "raw_kicl" / source / kol
    trace_root = case_root / "trace" / source / kol

    eq = _load_csv(
        raw_root / "equity_daily.csv",
        {"date", "equity_baseline", "equity_trained"},
    )
    eq["day"] = pd.to_datetime(eq["date"], errors="coerce").dt.floor("D")
    eq = eq.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    eq["gap"] = eq["equity_trained"] - eq["equity_baseline"]
    eq["d_gap"] = eq["gap"].diff().fillna(0.0)

    window = DEFAULT_WINDOWS.get((source, kol))
    if window is None:
        # fallback: use the middle region where positive d_gap appears
        cand = eq[eq["d_gap"] > 0]
        if cand.empty:
            start, end = eq["day"].min(), eq["day"].max()
        else:
            start = cand["day"].min()
            end = cand["day"].max()
    else:
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])

    eqw = eq[(eq["day"] >= start) & (eq["day"] <= end)].copy()
    if eqw.empty:
        return pd.DataFrame(), pd.DataFrame()

    focus_days = (
        eqw[eqw["d_gap"] > 0]
        .sort_values("d_gap", ascending=False)
        .head(top_days)[["day", "gap", "d_gap"]]
        .copy()
    )
    if focus_days.empty:
        return pd.DataFrame(), pd.DataFrame()

    pos = _load_csv(
        raw_root / "positions_test.csv",
        {"date", "ticker", "reward", "action", "baseline_action", "policy_action"},
    )
    pos["day"] = pd.to_datetime(pos["date"], errors="coerce").dt.floor("D")
    pos = pos.dropna(subset=["day"]).copy()
    pos["delta_action"] = pos["policy_action"] - pos["baseline_action"]
    pos["contribution"] = pos["reward"] * pos["delta_action"]

    # discourse evidence table already links ticker/day/sentiment/text.
    disc_path = trace_root / "kicl_discourse_evidence.csv"
    disc = None
    if disc_path.exists():
        disc = pd.read_csv(disc_path)
        if "day" in disc.columns:
            disc["day"] = pd.to_datetime(disc["day"], errors="coerce").dt.floor("D")

    event_days = sorted(pos["day"].dropna().unique().tolist())

    def _map_to_event_day(day: pd.Timestamp) -> pd.Timestamp | None:
        if not event_days:
            return None
        # map to the latest event day <= focus day
        cands = [d for d in event_days if d <= day]
        if not cands:
            return None
        return pd.Timestamp(cands[-1])

    rows = []
    day_rows = []
    used_mapped_days: set[pd.Timestamp] = set()
    for _, drow in focus_days.iterrows():
        day = pd.Timestamp(drow["day"]).floor("D")
        mapped_day = _map_to_event_day(day)
        if mapped_day is None:
            continue
        if mapped_day in used_mapped_days:
            continue
        used_mapped_days.add(mapped_day)

        q = pos[pos["day"] == mapped_day].copy().sort_values("contribution", ascending=False)
        q = q[q["contribution"] > 0].head(top_tickers).copy()
        if q.empty:
            continue

        for _, r in q.iterrows():
            sentiment = None
            confidence = None
            text_preview = None
            title = None
            event_id = None
            published_at = None

            if disc is not None and not disc.empty:
                hit = disc[(disc["day"] == mapped_day) & (disc["ticker"] == r["ticker"])]
                if hit.empty:
                    hit = disc[(disc["day"] == mapped_day)]
                if hit.empty:
                    # fallback 1: same ticker, nearest previous discourse day
                    ticker_hist = disc[(disc["ticker"] == r["ticker"]) & (disc["day"] <= mapped_day)]
                    if not ticker_hist.empty:
                        hit = ticker_hist.sort_values("day", ascending=False).head(1)
                if hit.empty:
                    # fallback 2: any ticker, nearest previous discourse day
                    hist = disc[disc["day"] <= mapped_day]
                    if not hist.empty:
                        hit = hist.sort_values("day", ascending=False).head(1)
                if not hit.empty:
                    h = hit.iloc[0]
                    sentiment = h.get("sentiment")
                    confidence = h.get("confidence")
                    text_preview = h.get("text_preview")
                    title = h.get("title")
                    event_id = h.get("event_id")
                    published_at = h.get("published_at")

            rows.append(
                {
                    "source": source,
                    "kol": kol,
                    "focus_day": day.date().isoformat(),
                    "mapped_event_day": mapped_day.date().isoformat(),
                    "gap": float(drow["gap"]),
                    "d_gap": float(drow["d_gap"]),
                    "ticker": r["ticker"],
                    "reward": float(r["reward"]),
                    "baseline_action": float(r["baseline_action"]),
                    "policy_action": float(r["policy_action"]),
                    "delta_action": float(r["delta_action"]),
                    "contribution": float(r["contribution"]),
                    "baseline_behavior": _baseline_behavior(float(r["baseline_action"]), eps),
                    "ours_behavior": _ours_behavior(str(r["action"]), float(r["policy_action"]), eps),
                    "relation": _relation_tag(float(r["baseline_action"]), float(r["policy_action"]), eps),
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "title": title,
                    "text_preview": text_preview,
                    "event_id": event_id,
                    "published_at": published_at,
                }
            )

        day_contrib = float((pos[pos["day"] == mapped_day]["contribution"]).sum())
        if day_contrib <= 0:
            continue
        day_rows.append(
            {
                "source": source,
                "kol": kol,
                "focus_day": day.date().isoformat(),
                "mapped_event_day": mapped_day.date().isoformat(),
                "gap": float(drow["gap"]),
                "d_gap": float(drow["d_gap"]),
                "day_total_contribution": day_contrib,
                "window_start": str(start.date()),
                "window_end": str(end.date()),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(day_rows)


def _write_markdown(path: Path, seq_df: pd.DataFrame, day_df: pd.DataFrame) -> None:
    lines = ["# Case Study Contrast Sequence", ""]
    if seq_df.empty:
        lines.append("No rows generated.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    src = seq_df["source"].iloc[0]
    kol = seq_df["kol"].iloc[0]
    lines.append(f"- Source: `{src}`")
    lines.append(f"- KOL: `{kol}`")
    if not day_df.empty:
        lines.append(
            f"- Focus window: `{day_df['window_start'].iloc[0]} -> {day_df['window_end'].iloc[0]}`"
        )
    lines.append("")
    lines.append("## Focus Days")
    lines.append("")
    lines.append("```text")
    lines.append(day_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Sequence (Discourse -> Sentiment -> Baseline -> Ours)")
    lines.append("")
    show_cols = [
        "focus_day",
        "ticker",
        "sentiment",
        "baseline_behavior",
        "ours_behavior",
        "relation",
        "contribution",
        "text_preview",
    ]
    lines.append("```text")
    lines.append(seq_df[show_cols].to_string(index=False))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cases = pd.read_csv(args.summary_csv)[["source", "kol"]].drop_duplicates()

    all_seq = []
    all_days = []

    for _, r in cases.iterrows():
        source, kol = str(r["source"]), str(r["kol"])
        seq, days = _build_for_case(
            case_root=args.case_root,
            source=source,
            kol=kol,
            top_days=args.top_days,
            top_tickers=args.top_tickers_per_day,
            eps=args.weight_eps,
        )
        if seq.empty:
            print(f"Skip (no sequence rows): {source}/{kol}")
            continue

        out_dir = args.output_dir / source / kol
        out_dir.mkdir(parents=True, exist_ok=True)
        seq_csv = out_dir / "contrast_sequence.csv"
        day_csv = out_dir / "focus_days.csv"
        md = out_dir / "CONTRAST_SEQUENCE.md"
        seq.to_csv(seq_csv, index=False)
        days.to_csv(day_csv, index=False)
        _write_markdown(md, seq, days)
        print(f"Saved: {seq_csv}")
        print(f"Saved: {day_csv}")
        print(f"Saved: {md}")

        all_seq.append(seq)
        all_days.append(days)

    if all_seq:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        all_seq_df = pd.concat(all_seq, ignore_index=True)
        all_days_df = pd.concat(all_days, ignore_index=True)
        all_seq_csv = args.output_dir / "all_cases_contrast_sequence.csv"
        all_days_csv = args.output_dir / "all_cases_focus_days.csv"
        all_seq_df.to_csv(all_seq_csv, index=False)
        all_days_df.to_csv(all_days_csv, index=False)
        print(f"Saved: {all_seq_csv}")
        print(f"Saved: {all_days_csv}")


if __name__ == "__main__":
    main()
