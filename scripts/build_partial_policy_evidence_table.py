#!/usr/bin/env python3
"""Build a partial-policy evidence table for selected KOL subsets.

Goal:
- Move beyond generic sentiment statistics.
- Quantify whether KOL discourse is:
  1) asset-specific and directional,
  2) under-specified on execution dimensions.

Inputs:
- input-root/source/*.csv   (must include text columns like excerpt/title)
- temporal-root/source/*.csv (existing company-stats outputs)

Outputs:
- per-row execution flags
- per-kol summary
- per-source summary
- merged table (CSV + MD)
- LaTeX table draft + metric definitions
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


TICKER_RE = re.compile(r"(?<!\w)\$[A-Z]{1,6}\b")
DIR_RE = re.compile(
    r"\b(?:"
    r"bullish|bearish|upside|downside|long|short|breakout|breakdown|"
    r"rally|selloff|bounce|dip|overvalued|undervalued|momentum|"
    r"accumulat(?:e|ing|ion)|weak|strong|outperform|underperform"
    r")\b",
    re.IGNORECASE,
)
COND_RE = re.compile(
    r"\b(?:"
    r"if|when|unless|until|once|as long as|wait for|before acting|"
    r"if it breaks|if it holds|needs to|only if|on pullback|above\b|below\b"
    r")\b",
    re.IGNORECASE,
)
ENTRY_RE = re.compile(
    r"\b(?:"
    r"buy|bought|add|added|accumulat(?:e|ing|ion)|enter|entry|"
    r"start(?:ed)? (?:a )?position|initiat(?:e|ed) (?:a )?position|"
    r"open(?:ed)? (?:a )?(?:long|short|position)"
    r")\b",
    re.IGNORECASE,
)
SIZING_RE = re.compile(
    r"\b(?:"
    r"\d+(?:\.\d+)?\s*(?:shares?|%|percent|pct)|"
    r"position size|allocation|weight|sized|sizing|"
    r"small position|full position|trim|reduc(?:e|ed|ing)|scale in|scale out"
    r")\b",
    re.IGNORECASE,
)
HORIZON_RE = re.compile(
    r"\b(?:"
    r"next (?:day|week|month|quarter|year)|"
    r"short[- ]term|long[- ]term|"
    r"end of (?:day|week|month|quarter|year)|"
    r"by (?:year[- ]end|month[- ]end|quarter[- ]end|\\d{4})|"
    r"for (?:\\d+\\s*(?:days?|weeks?|months?|years?))|"
    r"hold(?:ing)? (?:for|through|into)"
    r")\b",
    re.IGNORECASE,
)
EXIT_RE = re.compile(
    r"\b(?:"
    r"sell|sold|exit|close(?:d)? (?:a )?position|take profit|tp\b|"
    r"stop loss|sl\b|cut loss|de-?risk|"
    r"trim|reduce|lighten (?:up|position)"
    r")\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build partial-policy evidence table.")
    parser.add_argument(
        "--input-root",
        default="outputs/analysis/company_stats_selected20_20260321/input_subset",
        help="Root with source subdirs containing per-KOL csv files.",
    )
    parser.add_argument(
        "--temporal-root",
        default="outputs/analysis/company_stats_selected20_20260321",
        help="Root with source subdirs containing global_* temporal stats csv files.",
    )
    parser.add_argument(
        "--sources",
        default="youtube,x",
        help="Comma-separated sources to include.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/company_stats_selected20_20260321/partial_policy_evidence",
        help="Output directory.",
    )
    parser.add_argument(
        "--directional-sentiment-threshold",
        type=float,
        default=0.2,
        help="Absolute sentiment threshold used as a directional cue.",
    )
    return parser.parse_args()


def clean_kol_name(stem: str) -> str:
    return stem.replace("_companies_cleaned", "")


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def fnum(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{nd}f}"


def parse_text_columns(df: pd.DataFrame) -> pd.Series:
    excerpt = df["excerpt"].fillna("").astype(str) if "excerpt" in df.columns else pd.Series([""] * len(df))
    title = df["title"].fillna("").astype(str) if "title" in df.columns else pd.Series([""] * len(df))
    text = (excerpt + " " + title).str.replace(r"\s+", " ", regex=True).str.strip()
    return text


def add_execution_flags(df: pd.DataFrame, directional_sentiment_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["text"] = parse_text_columns(out)
    text_l = out["text"].str.lower()

    company = out["company"].fillna("").astype(str).str.lower() if "company" in out.columns else pd.Series([""] * len(out))
    company_in_text = pd.Series(
        [bool(c) and (c in t) for c, t in zip(company, text_l)],
        index=out.index,
    )
    has_ticker_token = text_l.str.contains(TICKER_RE, na=False)
    out["ticker_explicit"] = has_ticker_token | company_in_text

    sentiment = pd.to_numeric(out.get("sentiment", 0.0), errors="coerce").fillna(0.0)
    out["directional_expression"] = text_l.str.contains(DIR_RE, na=False) | (sentiment.abs() >= directional_sentiment_threshold)
    out["conditional_action"] = text_l.str.contains(COND_RE, na=False)

    out["explicit_entry"] = text_l.str.contains(ENTRY_RE, na=False)
    out["explicit_sizing"] = text_l.str.contains(SIZING_RE, na=False)
    out["explicit_horizon"] = text_l.str.contains(HORIZON_RE, na=False)
    out["explicit_exit"] = text_l.str.contains(EXIT_RE, na=False)
    out["full_execution_completeness"] = (
        out["explicit_entry"] & out["explicit_sizing"] & out["explicit_horizon"] & out["explicit_exit"]
    )
    return out


def build_source_flags(input_root: Path, source: str, directional_sentiment_threshold: float) -> pd.DataFrame:
    files = sorted((input_root / source).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {(input_root / source)}")
    rows = []
    for p in files:
        df = pd.read_csv(p)
        date_col = "publishedAt" if "publishedAt" in df.columns else ("published_at" if "published_at" in df.columns else None)
        required_cols = {"company", "sentiment"}
        if not required_cols.issubset(df.columns):
            continue

        # Align base filtering with company_stats analyzer:
        # - valid company
        # - numeric sentiment
        # - valid timestamp when available
        tmp = df.copy()
        tmp["company"] = tmp["company"].astype(str).str.strip()
        tmp = tmp[tmp["company"] != ""].copy()
        tmp["sentiment"] = pd.to_numeric(tmp["sentiment"], errors="coerce")
        tmp = tmp.dropna(subset=["sentiment"])
        if date_col is not None:
            tmp["__date__"] = pd.to_datetime(tmp[date_col], errors="coerce", utc=True)
            tmp = tmp.dropna(subset=["__date__"])

        if tmp.empty:
            continue

        tmp["source"] = source
        tmp["kol"] = clean_kol_name(p.stem)
        flagged = add_execution_flags(tmp, directional_sentiment_threshold=directional_sentiment_threshold)
        rows.append(flagged)
    if not rows:
        raise ValueError(f"No valid rows for source={source} under {input_root / source}")
    return pd.concat(rows, ignore_index=True)


def summarize_ratios(df: pd.DataFrame) -> Dict[str, float]:
    n = len(df)
    if n == 0:
        return {k: np.nan for k in [
            "rows",
            "ticker_explicit_ratio",
            "directional_expression_ratio",
            "conditional_action_ratio",
            "explicit_entry_ratio",
            "explicit_sizing_ratio",
            "explicit_horizon_ratio",
            "explicit_exit_ratio",
            "full_execution_completeness_ratio",
        ]}
    return {
        "rows": int(n),
        "ticker_explicit_ratio": float(df["ticker_explicit"].mean()),
        "directional_expression_ratio": float(df["directional_expression"].mean()),
        "conditional_action_ratio": float(df["conditional_action"].mean()),
        "explicit_entry_ratio": float(df["explicit_entry"].mean()),
        "explicit_sizing_ratio": float(df["explicit_sizing"].mean()),
        "explicit_horizon_ratio": float(df["explicit_horizon"].mean()),
        "explicit_exit_ratio": float(df["explicit_exit"].mean()),
        "full_execution_completeness_ratio": float(df["full_execution_completeness"].mean()),
    }


def load_temporal_metrics(temporal_root: Path, source: str) -> Dict[str, float]:
    src = temporal_root / source
    freq = pd.read_csv(src / "global_frequency_stats.csv").iloc[0]
    follow = pd.read_csv(src / "global_followup_summary.csv")
    sil = pd.read_csv(src / "global_silence_stats.csv")
    rev = pd.read_csv(src / "global_reversal_summary.csv").iloc[0]

    row30 = follow[follow["window_days"] == 30].iloc[0]
    row60 = follow[follow["window_days"] == 60].iloc[0]
    row90 = follow[follow["window_days"] == 90].iloc[0]
    sil_all = sil[sil["scope"] == "all"].iloc[0]

    return {
        "mentioned_companies": float(freq["companies_total"]),
        "single_mention_ratio": float(freq["mentioned_once_pct"]),
        "no_followup_within_30d": float(row30["no_followup_within_window_pct"]),
        "no_followup_within_60d": float(row60["no_followup_within_window_pct"]),
        "no_followup_within_90d": float(row90["no_followup_within_window_pct"]),
        "silence_duration_p90_days": float(sil_all["gap_p90_days"]),
        "sentiment_reversal_rate": float(rev["reversal_rate"]),
        "median_time_to_first_reversal_days": float(rev["ttf_median_days"]),
    }


def build_table(summary_by_source: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    order = ["youtube", "x"]
    rows = []
    def add(section: str, metric: str, key: str):
        y = summary_by_source["youtube"].get(key, np.nan)
        x = summary_by_source["x"].get(key, np.nan)
        rows.append({"section": section, "metric": metric, "YouTube": y, "X": x})

    add("Coverage / specificity", "Mentioned companies", "mentioned_companies")
    add("Coverage / specificity", "Ticker-explicit ratio", "ticker_explicit_ratio")
    add("Coverage / specificity", "Directional-expression ratio", "directional_expression_ratio")
    add("Coverage / specificity", "Conditional-action ratio", "conditional_action_ratio")
    add("Coverage / specificity", "Single-mention ratio", "single_mention_ratio")

    add("Execution under-specification", "Explicit entry ratio", "explicit_entry_ratio")
    add("Execution under-specification", "Explicit sizing ratio", "explicit_sizing_ratio")
    add("Execution under-specification", "Explicit holding-horizon ratio", "explicit_horizon_ratio")
    add("Execution under-specification", "Explicit exit ratio", "explicit_exit_ratio")
    add("Execution under-specification", "Full-execution completeness ratio", "full_execution_completeness_ratio")

    add("Temporal maintenance", "No follow-up within 30 days", "no_followup_within_30d")
    add("Temporal maintenance", "No follow-up within 60 days", "no_followup_within_60d")
    add("Temporal maintenance", "No follow-up within 90 days", "no_followup_within_90d")
    add("Temporal maintenance", "Silence duration (p90, days)", "silence_duration_p90_days")
    add("Temporal maintenance", "Sentiment reversal rate", "sentiment_reversal_rate")
    add("Temporal maintenance", "Median time to first reversal (days)", "median_time_to_first_reversal_days")

    return pd.DataFrame(rows)


def format_table_for_md(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype({"YouTube": "object", "X": "object"})
    for idx, row in out.iterrows():
        metric = row["metric"].lower()
        if "mentioned companies" in metric:
            out.at[idx, "YouTube"] = fnum(row["YouTube"], nd=0)
            out.at[idx, "X"] = fnum(row["X"], nd=0)
        elif "silence duration" in metric or "median time to first reversal" in metric:
            out.at[idx, "YouTube"] = fnum(row["YouTube"], nd=2)
            out.at[idx, "X"] = fnum(row["X"], nd=2)
        else:
            out.at[idx, "YouTube"] = pct(row["YouTube"])
            out.at[idx, "X"] = pct(row["X"])
    return out


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    widths = {c: max(len(str(c)), *(len(str(v)) for v in df[c].tolist())) for c in cols}
    header = "| " + " | ".join(f"{c:<{widths[c]}}" for c in cols) + " |"
    sep = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(f"{str(row[c]):<{widths[c]}}" for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def write_latex_skeleton(table_md: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Empirical evidence that KOL discourse is a partial trading policy.}")
    lines.append(r"\label{tab:partial_policy_evidence}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4.5pt}")
    lines.append(r"\begin{tabular}{l l c c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Section} & \textbf{Statistic} & \textbf{YouTube} & \textbf{X} \\")
    lines.append(r"\midrule")
    current = None
    for _, r in table_md.iterrows():
        sec = r["section"]
        metric = r["metric"]
        y = r["YouTube"]
        x = r["X"]
        left = sec if sec != current else ""
        current = sec
        lines.append(f"{left} & {metric} & {y} & {x} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_definitions(out_path: Path) -> None:
    definitions = {
        "ticker_explicit_ratio": "Share of asset-level discourse rows with explicit asset mention in text (ticker token like $AAPL or company-name mention in excerpt/title).",
        "directional_expression_ratio": "Share of rows with explicit directional expression (lexical bullish/bearish cues or |sentiment| >= threshold).",
        "conditional_action_ratio": "Share of rows containing conditional execution language (e.g., if/when/wait-for/above/below).",
        "explicit_entry_ratio": "Share of rows explicitly describing entry/opening action.",
        "explicit_sizing_ratio": "Share of rows explicitly describing size/allocation/scale in-out information.",
        "explicit_holding_horizon_ratio": "Share of rows explicitly specifying temporal holding horizon/window.",
        "explicit_exit_ratio": "Share of rows explicitly describing exit/profit-taking/stop/de-risk actions.",
        "full_execution_completeness_ratio": "Share of rows where entry + sizing + horizon + exit are all simultaneously explicit.",
        "single_mention_ratio": "Company-level ratio of firms mentioned only once in the observation window.",
        "no_followup_within_30d": "Company-level ratio with no second mention within 30 days from first mention.",
        "no_followup_within_60d": "Company-level ratio with no second mention within 60 days from first mention.",
        "no_followup_within_90d": "Company-level ratio with no second mention within 90 days from first mention.",
        "silence_duration_p90_days": "90th percentile of inter-mention gap (days) on company timeline.",
        "sentiment_reversal_rate": "Company-level proportion that ever flips directional sentiment (positive <-> negative).",
        "median_time_to_first_reversal_days": "Median time from first directional mention to first observed reversal.",
    }
    out_path.write_text(json.dumps(definitions, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    temporal_root = Path(args.temporal_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    all_source_summary: Dict[str, Dict[str, float]] = {}

    for source in sources:
        source_dir = out_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)

        flags = build_source_flags(
            input_root=input_root,
            source=source,
            directional_sentiment_threshold=float(args.directional_sentiment_threshold),
        )
        flags.to_csv(source_dir / "partial_policy_flags_per_row.csv", index=False)

        per_kol_rows = []
        for kol, g in flags.groupby("kol"):
            r = summarize_ratios(g)
            r["kol"] = kol
            per_kol_rows.append(r)
        per_kol = pd.DataFrame(per_kol_rows)
        per_kol.to_csv(source_dir / "partial_policy_summary_per_kol.csv", index=False)

        src_summary = summarize_ratios(flags)
        src_summary.update(load_temporal_metrics(temporal_root=temporal_root, source=source))
        all_source_summary[source] = src_summary

    table_raw = build_table(all_source_summary)
    table_raw.to_csv(out_dir / "table_partial_policy_evidence_raw.csv", index=False)

    table_fmt = format_table_for_md(table_raw)
    table_fmt.to_csv(out_dir / "table_partial_policy_evidence_formatted.csv", index=False)
    md_body = dataframe_to_markdown(table_fmt)
    (out_dir / "table_partial_policy_evidence.md").write_text(
        "# Partial Policy Evidence Table (Selected-20)\n\n" + md_body,
        encoding="utf-8",
    )
    write_latex_skeleton(table_fmt, out_dir / "table_partial_policy_evidence.tex")
    write_definitions(out_dir / "metric_definitions.json")

    meta = {
        "input_root": str(input_root),
        "temporal_root": str(temporal_root),
        "sources": sources,
        "output_dir": str(out_dir),
        "directional_sentiment_threshold": float(args.directional_sentiment_threshold),
        "rows_by_source": {k: int(v["rows"]) for k, v in all_source_summary.items()},
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
