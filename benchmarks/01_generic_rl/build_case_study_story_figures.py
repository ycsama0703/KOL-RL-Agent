#!/usr/bin/env python3
"""Build narrative case-study figures:
discourse -> action -> contribution -> equity impact.
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Case:
    source: str
    kol: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case-root",
        type=Path,
        default=Path("benchmarks/compare/case_study"),
    )
    p.add_argument(
        "--top-k-tickers",
        type=int,
        default=4,
        help="Top contributing tickers shown on anchor day.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/compare/case_study/figures"),
    )
    return p.parse_args()


def _clean_text(s: object, n: int = 220) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    txt = str(s).replace("\n", " ").strip()
    txt = " ".join(txt.split())
    return txt[:n]


def _load_case(case_root: Path, c: Case) -> Tuple[pd.DataFrame, pd.DataFrame]:
    eq = pd.read_csv(case_root / "raw_kicl" / c.source / c.kol / "equity_daily.csv")
    evi = pd.read_csv(case_root / "trace" / c.source / c.kol / "kicl_discourse_evidence.csv")
    eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
    evi["day_dt"] = pd.to_datetime(evi["day"], errors="coerce")
    return eq, evi


def _pick_anchor_row(evi: pd.DataFrame) -> pd.Series:
    base = evi[evi["contribution"].fillna(0) > 0].copy()
    if base.empty:
        base = evi.copy()
    base = base.sort_values("contribution", ascending=False)

    has_text = (
        base["text_preview"].fillna("").astype(str).str.strip().ne("")
        | base["text"].fillna("").astype(str).str.strip().ne("")
        | base["title"].fillna("").astype(str).str.strip().ne("")
    )
    if has_text.any():
        return base.loc[has_text].iloc[0]
    return base.iloc[0]


def _enrich_quote(day_rows: pd.DataFrame, all_evi: pd.DataFrame, anchor_day: pd.Timestamp) -> pd.DataFrame:
    out = day_rows.copy()
    out["quote_title"] = out["title"].apply(_clean_text)
    out["quote_text"] = out["text_preview"].apply(_clean_text)
    out["quote_text"] = np.where(out["quote_text"].eq(""), out["text"].apply(_clean_text), out["quote_text"])

    for i, r in out.iterrows():
        if out.at[i, "quote_title"] or out.at[i, "quote_text"]:
            continue
        t = str(r["ticker"])
        cand = all_evi[all_evi["ticker"].astype(str).eq(t)].copy()
        cand = cand[
            cand["title"].fillna("").astype(str).str.strip().ne("")
            | cand["text"].fillna("").astype(str).str.strip().ne("")
            | cand["text_preview"].fillna("").astype(str).str.strip().ne("")
        ]
        if cand.empty:
            continue
        cand["dist"] = (cand["day_dt"] - anchor_day).abs()
        pick = cand.sort_values("dist").iloc[0]
        out.at[i, "quote_title"] = _clean_text(pick.get("title", ""))
        out.at[i, "quote_text"] = _clean_text(pick.get("text_preview", "") or pick.get("text", ""))
    return out


def _format_quote_block(rows: pd.DataFrame, max_items: int = 2) -> str:
    chunks: List[str] = []
    for _, r in rows.head(max_items).iterrows():
        ticker = str(r["ticker"])
        action = str(r.get("action", ""))
        contrib = float(r.get("contribution", 0.0))
        title = _clean_text(r.get("quote_title", ""), 120)
        text = _clean_text(r.get("quote_text", ""), 220)
        lines = [f"[{ticker}] {action} | contrib={contrib:.4f}"]
        if title:
            lines.append(f"title: {title}")
        if text:
            lines.append(f"text: {text}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) if chunks else "No matched discourse text found."


def plot_case_story(case_root: Path, c: Case, out_dir: Path, top_k: int) -> Tuple[Path, Path]:
    eq, evi = _load_case(case_root, c)
    anchor = _pick_anchor_row(evi)
    anchor_day = pd.to_datetime(anchor["day_dt"])

    day_rows = (
        evi[evi["day_dt"].dt.date == anchor_day.date()]
        .copy()
        .sort_values("contribution", ascending=False)
    )
    day_rows = day_rows[day_rows["contribution"].fillna(0) > 0].head(top_k)
    day_rows = _enrich_quote(day_rows, evi, anchor_day)

    # --- Figure ---
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(13.8, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[1.0, 1.0], wspace=0.20, hspace=0.12)

    ax_eq = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, 1])

    # Panel A: Equity
    ax_eq.plot(eq["date"], eq["equity_baseline"], color="#7f8c8d", lw=2.0, label="Baseline")
    ax_eq.plot(eq["date"], eq["equity_trained"], color="#f39c12", lw=2.8, label="KICL")
    ax_eq.axvline(anchor_day, color="#2c3e50", lw=1.8, ls="--", alpha=0.9)
    ax_eq.scatter([anchor_day], [float(eq.loc[eq["date"] == anchor_day, "equity_trained"].iloc[0]) if (eq["date"] == anchor_day).any() else np.nan],
                  marker="o", s=40, color="#2c3e50", zorder=6)
    ax_eq.set_title(f"{c.source}/{c.kol}: equity trajectory", fontsize=13, pad=8)
    ax_eq.set_ylabel("Equity")
    ax_eq.legend(loc="upper left", frameon=True, fontsize=10)
    ax_eq.grid(True, ls="--", alpha=0.35)
    ax_eq.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax_eq.get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(9)

    # anchor annotation
    anchor_contrib = float(evi[evi["day_dt"].dt.date == anchor_day.date()]["contribution"].fillna(0).sum())
    ax_eq.text(
        0.02,
        0.04,
        f"Anchor day: {anchor_day.date()} | total contribution={anchor_contrib:.4f}",
        transform=ax_eq.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#aaaaaa"),
    )

    # Panel B: Top ticker contributions
    if day_rows.empty:
        ax_bar.text(0.5, 0.5, "No positive-contribution tickers.", ha="center", va="center")
        ax_bar.set_axis_off()
    else:
        y = np.arange(len(day_rows))
        vals = day_rows["contribution"].astype(float).values
        labels = day_rows["ticker"].astype(str).tolist()
        bars = ax_bar.barh(y, vals, color="#3498db", alpha=0.85)
        ax_bar.set_yticks(y, labels=labels)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Contribution (weight × reward)")
        ax_bar.set_title(f"Top tickers on {anchor_day.date()}", fontsize=12, pad=6)
        ax_bar.grid(True, axis="x", ls="--", alpha=0.3)
        for bar, (_, r) in zip(bars, day_rows.iterrows()):
            v = float(bar.get_width())
            act = str(r.get("action", ""))
            b = float(r.get("baseline_action", 0.0))
            p = float(r.get("policy_action", 0.0))
            ax_bar.text(v, bar.get_y() + bar.get_height() / 2, f"  {act} ({b:.2f}->{p:.2f})",
                        va="center", ha="left", fontsize=9, color="#2c3e50")

    # Panel C: discourse snippets
    ax_txt.set_axis_off()
    quote_block = _format_quote_block(day_rows, max_items=2)
    wrapped = "\n\n".join(
        textwrap.fill(chunk, width=62, break_long_words=False) for chunk in quote_block.split("\n\n")
    )
    ax_txt.text(
        0.01,
        0.98,
        "Discourse evidence (top contributors)",
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax_txt.text(
        0.01,
        0.90,
        wrapped,
        fontsize=9.5,
        va="top",
        ha="left",
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
    )

    fig.suptitle("Case-study evidence chain: discourse → action → contribution", fontsize=14, y=0.98)
    out_sub = out_dir / c.source / c.kol
    out_sub.mkdir(parents=True, exist_ok=True)
    png = out_sub / "case_story_chain.png"
    pdf = out_sub / "case_story_chain.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    case_root = args.case_root
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases_df = pd.read_csv(case_root / "case_study_selected_kols_summary.csv")
    cases = [Case(source=r["source"], kol=r["kol"]) for _, r in cases_df.iterrows()]

    generated: List[Tuple[Path, Path]] = []
    for c in cases:
        generated.append(plot_case_story(case_root, c, args.output_dir, args.top_k_tickers))

    print("Generated case-story figures:")
    for png, pdf in generated:
        print(f"- {png}")
        print(f"- {pdf}")


if __name__ == "__main__":
    main()

