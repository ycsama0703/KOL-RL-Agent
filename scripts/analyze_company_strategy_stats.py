#!/usr/bin/env python3
"""Compute strategy-completeness statistics on KOL company-level signals.

Input expectation:
- Directory of per-KOL CSV files (e.g. data/22-25_youtube/*.csv)
- Required columns: company, sentiment, publishedAt (or published_at)
- Optional columns: confidence, channel_name

Outputs:
- Multiple CSV tables under output dir, at both global and per-KOL levels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze company-level discourse statistics for KOL datasets.")
    p.add_argument("--input-dir", default="data/22-25_youtube", help="Directory containing per-KOL CSV files.")
    p.add_argument(
        "--output-dir",
        default="outputs/analysis/youtube_company_stats",
        help="Directory to write analysis tables.",
    )
    p.add_argument(
        "--min-directional-mentions-for-reversal",
        type=int,
        default=2,
        help="Minimum number of non-neutral mentions for reversal denominator.",
    )
    p.add_argument(
        "--normalize-company",
        action="store_true",
        default=True,
        help="Normalize company names before grouping (default: enabled).",
    )
    return p.parse_args()


def normalize_company_name(name: str) -> str:
    lowered = str(name).strip().lower()
    lowered = re.sub(r"\.com\b", "", lowered)
    lowered = re.sub(r"[^a-z0-9&\-\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def sentiment_class(v: float) -> str:
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return "neutral"


def summarize_series(values: pd.Series) -> Dict[str, float]:
    if values.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
    }


def load_input(input_dir: Path, normalize_company: bool) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {input_dir}")

    rows: List[pd.DataFrame] = []
    for p in files:
        df = pd.read_csv(p)
        date_col = "publishedAt" if "publishedAt" in df.columns else "published_at"
        required = {"company", "sentiment", date_col}
        if not required.issubset(df.columns):
            continue
        out = df.copy()
        out["kol"] = p.stem.replace("_companies_cleaned", "")
        out["date"] = pd.to_datetime(out[date_col], errors="coerce", utc=True)
        out = out.dropna(subset=["date", "company", "sentiment"]).copy()
        out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce")
        out = out.dropna(subset=["sentiment"]).copy()
        if normalize_company:
            out["company"] = out["company"].astype(str).map(normalize_company_name)
            out = out[out["company"] != ""]
        out["sentiment_cls"] = out["sentiment"].map(sentiment_class)
        rows.append(out[["kol", "date", "company", "sentiment", "sentiment_cls"]])

    if not rows:
        raise ValueError("No valid rows found after schema filtering.")
    return pd.concat(rows, ignore_index=True)


def ticker_mention_frequency(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df["company"].value_counts().rename_axis("company").reset_index(name="mentions")
    s = counts["mentions"]
    stats = summarize_series(s)
    total_companies = len(counts)
    once = int((s == 1).sum())
    ge2 = int((s >= 2).sum())
    row = {
        "companies_total": total_companies,
        "mentioned_once_companies": once,
        "mentioned_once_pct": once / total_companies if total_companies else np.nan,
        "mentioned_ge2_companies": ge2,
        "mentioned_ge2_pct": ge2 / total_companies if total_companies else np.nan,
        **{f"mentions_{k}": v for k, v in stats.items()},
    }
    return pd.DataFrame([row]), counts


def silence_duration(df: pd.DataFrame) -> pd.DataFrame:
    base = df.sort_values(["company", "date"]).copy()
    eligible = base["company"].value_counts()
    eligible = set(eligible[eligible >= 2].index.tolist())
    base = base[base["company"].isin(eligible)]

    def gaps_for(sub: pd.DataFrame) -> pd.Series:
        values: List[float] = []
        for _, g in sub.groupby("company", sort=False):
            d = g["date"].sort_values()
            if len(d) < 2:
                continue
            gap_days = d.diff().dropna().dt.total_seconds() / 86400.0
            values.extend(gap_days.tolist())
        return pd.Series(values, dtype=float)

    all_gaps = gaps_for(base)
    pos_gaps = gaps_for(base[base["sentiment"] > 0])
    neg_gaps = gaps_for(base[base["sentiment"] < 0])

    rows = []
    for label, series in [
        ("all", all_gaps),
        ("positive_only", pos_gaps),
        ("negative_only", neg_gaps),
    ]:
        stat = summarize_series(series)
        rows.append(
            {
                "scope": label,
                "gaps_count": stat["count"],
                "gap_median_days": stat["median"],
                "gap_p90_days": stat["p90"],
                "gap_max_days": stat["max"],
            }
        )
    return pd.DataFrame(rows)


def sentiment_reversal(
    df: pd.DataFrame,
    min_directional_mentions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_rows: List[Dict[str, float]] = []
    for company, g in df.sort_values("date").groupby("company", sort=False):
        gd = g[g["sentiment"] != 0].copy()
        gd = gd.sort_values("date")
        directional_n = len(gd)
        if directional_n < min_directional_mentions:
            out_rows.append(
                {
                    "company": company,
                    "directional_mentions": directional_n,
                    "ever_reversal": False,
                    "time_to_first_reversal_days": np.nan,
                }
            )
            continue

        signs = np.sign(gd["sentiment"].to_numpy())
        dates = gd["date"].to_numpy()
        first_sign = signs[0]
        first_date = pd.Timestamp(dates[0])

        rev_idx = None
        for i in range(1, len(signs)):
            if signs[i] != first_sign:
                rev_idx = i
                break

        if rev_idx is None:
            out_rows.append(
                {
                    "company": company,
                    "directional_mentions": directional_n,
                    "ever_reversal": False,
                    "time_to_first_reversal_days": np.nan,
                }
            )
        else:
            rev_date = pd.Timestamp(dates[rev_idx])
            ttf = (rev_date - first_date).total_seconds() / 86400.0
            out_rows.append(
                {
                    "company": company,
                    "directional_mentions": directional_n,
                    "ever_reversal": True,
                    "time_to_first_reversal_days": float(ttf),
                }
            )

    company_rev = pd.DataFrame(out_rows)
    denom = int((company_rev["directional_mentions"] >= min_directional_mentions).sum())
    rev_n = int(company_rev["ever_reversal"].sum())
    ttf = company_rev.loc[company_rev["ever_reversal"], "time_to_first_reversal_days"].dropna()
    ttf_stats = summarize_series(ttf)

    summary = pd.DataFrame(
        [
            {
                "companies_with_min_directional_mentions": denom,
                "companies_with_reversal": rev_n,
                "reversal_rate": rev_n / denom if denom else np.nan,
                "ttf_count": ttf_stats["count"],
                "ttf_median_days": ttf_stats["median"],
                "ttf_p90_days": ttf_stats["p90"],
                "ttf_max_days": ttf_stats["max"],
            }
        ]
    )
    return summary, company_rev


def signal_imbalance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = len(df)
    pos = int((df["sentiment"] > 0).sum())
    neg = int((df["sentiment"] < 0).sum())
    neu = int((df["sentiment"] == 0).sum())

    transitions: Dict[tuple[str, str], int] = {}
    for _, g in df.sort_values("date").groupby("company", sort=False):
        seq = g["sentiment_cls"].tolist()
        for i in range(1, len(seq)):
            key = (seq[i - 1], seq[i])
            transitions[key] = transitions.get(key, 0) + 1

    matrix_rows = []
    states = ["positive", "negative", "neutral"]
    for a in states:
        row_total = sum(transitions.get((a, b), 0) for b in states)
        for b in states:
            c = transitions.get((a, b), 0)
            matrix_rows.append(
                {
                    "from": a,
                    "to": b,
                    "count": c,
                    "row_prob": c / row_total if row_total else np.nan,
                }
            )
    matrix = pd.DataFrame(matrix_rows)

    pos_to_neg = transitions.get(("positive", "negative"), 0)
    pos_row_total = sum(transitions.get(("positive", b), 0) for b in states)
    neg_to_pos = transitions.get(("negative", "positive"), 0)
    neg_row_total = sum(transitions.get(("negative", b), 0) for b in states)

    summary = pd.DataFrame(
        [
            {
                "posts_total": total,
                "positive_count": pos,
                "negative_count": neg,
                "neutral_count": neu,
                "positive_ratio": pos / total if total else np.nan,
                "negative_ratio": neg / total if total else np.nan,
                "neutral_ratio": neu / total if total else np.nan,
                "pos_to_neg_count": pos_to_neg,
                "p_next_neg_given_pos": pos_to_neg / pos_row_total if pos_row_total else np.nan,
                "neg_to_pos_count": neg_to_pos,
                "p_next_pos_given_neg": neg_to_pos / neg_row_total if neg_row_total else np.nan,
            }
        ]
    )
    return summary, matrix


def long_silence_after_first(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for company, g in df.sort_values("date").groupby("company", sort=False):
        d = g["date"].sort_values().tolist()
        first = d[0]
        if len(d) >= 2:
            days_to_next = (d[1] - first).total_seconds() / 86400.0
        else:
            days_to_next = np.nan
        rows.append({"company": company, "days_to_first_followup": days_to_next})
    per_company = pd.DataFrame(rows)

    out = []
    for w in (30, 60, 90):
        no_follow = int(
            per_company["days_to_first_followup"].isna().sum()
            + (per_company["days_to_first_followup"] > w).sum()
        )
        total = len(per_company)
        out.append(
            {
                "window_days": w,
                "companies_total": total,
                "no_followup_within_window_count": no_follow,
                "no_followup_within_window_pct": no_follow / total if total else np.nan,
            }
        )
    return pd.DataFrame(out), per_company


def run_level(df: pd.DataFrame, out_dir: Path, prefix: str, min_directional_mentions: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_stats, freq_counts = ticker_mention_frequency(df)
    silence = silence_duration(df)
    rev_summary, rev_company = sentiment_reversal(df, min_directional_mentions=min_directional_mentions)
    imbalance_summary, transition_matrix = signal_imbalance(df)
    followup_summary, followup_company = long_silence_after_first(df)

    freq_stats.to_csv(out_dir / f"{prefix}_frequency_stats.csv", index=False)
    freq_counts.to_csv(out_dir / f"{prefix}_frequency_counts.csv", index=False)
    silence.to_csv(out_dir / f"{prefix}_silence_stats.csv", index=False)
    rev_summary.to_csv(out_dir / f"{prefix}_reversal_summary.csv", index=False)
    rev_company.to_csv(out_dir / f"{prefix}_reversal_by_company.csv", index=False)
    imbalance_summary.to_csv(out_dir / f"{prefix}_imbalance_summary.csv", index=False)
    transition_matrix.to_csv(out_dir / f"{prefix}_transition_matrix.csv", index=False)
    followup_summary.to_csv(out_dir / f"{prefix}_followup_summary.csv", index=False)
    followup_company.to_csv(out_dir / f"{prefix}_followup_by_company.csv", index=False)


def run_per_kol(df: pd.DataFrame, out_dir: Path, min_directional_mentions: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_rows = []
    silence_rows = []
    rev_rows = []
    imb_rows = []
    fol_rows = []

    for kol, sub in df.groupby("kol", sort=True):
        freq_stats, _ = ticker_mention_frequency(sub)
        freq_stats.insert(0, "kol", kol)
        freq_rows.append(freq_stats)

        sil = silence_duration(sub)
        sil.insert(0, "kol", kol)
        silence_rows.append(sil)

        rev_summary, _ = sentiment_reversal(sub, min_directional_mentions=min_directional_mentions)
        rev_summary.insert(0, "kol", kol)
        rev_rows.append(rev_summary)

        imb_summary, _ = signal_imbalance(sub)
        imb_summary.insert(0, "kol", kol)
        imb_rows.append(imb_summary)

        followup_summary, _ = long_silence_after_first(sub)
        followup_summary.insert(0, "kol", kol)
        fol_rows.append(followup_summary)

    pd.concat(freq_rows, ignore_index=True).to_csv(out_dir / "per_kol_frequency_stats.csv", index=False)
    pd.concat(silence_rows, ignore_index=True).to_csv(out_dir / "per_kol_silence_stats.csv", index=False)
    pd.concat(rev_rows, ignore_index=True).to_csv(out_dir / "per_kol_reversal_summary.csv", index=False)
    pd.concat(imb_rows, ignore_index=True).to_csv(out_dir / "per_kol_imbalance_summary.csv", index=False)
    pd.concat(fol_rows, ignore_index=True).to_csv(out_dir / "per_kol_followup_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    df = load_input(input_dir=input_dir, normalize_company=bool(args.normalize_company))
    df = df.sort_values(["kol", "company", "date"]).reset_index(drop=True)

    run_level(
        df=df,
        out_dir=output_dir,
        prefix="global",
        min_directional_mentions=int(args.min_directional_mentions_for_reversal),
    )
    run_per_kol(
        df=df,
        out_dir=output_dir,
        min_directional_mentions=int(args.min_directional_mentions_for_reversal),
    )

    summary = {
        "rows": int(len(df)),
        "kols": int(df["kol"].nunique()),
        "companies": int(df["company"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "output_dir": str(output_dir),
    }
    (output_dir / "run_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

