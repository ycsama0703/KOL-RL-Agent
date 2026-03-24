#!/usr/bin/env python3
"""Build a focused case-study package:
1) side-by-side figure: Ours vs Baseline, Ours vs an ablation variant
2) node-level evidence table: discourse/sentiment/actions around key divergence nodes
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


@dataclass
class Node:
    node_id: int
    contrast: str
    focus_day: pd.Timestamp
    d_gap: float
    gap: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-root", type=Path, default=Path("benchmarks/compare/case_study"))
    p.add_argument(
        "--ab-root",
        type=Path,
        default=Path("ablation study/ab_test_results/w_no_hard"),
        help="Ablation test root containing <source>/<kol_run>/daily,event.",
    )
    p.add_argument("--source", type=str, default="youtube")
    p.add_argument("--kol", type=str, default="The_Maverick_of_Wall_Street")
    p.add_argument(
        "--kicl-root",
        type=Path,
        default=Path("benchmarks/bench_test_results/multisource_test_mainline_xrefresh"),
        help="KICL test results root: <root>/<source>/<kol_run>/{daily,event}.",
    )
    p.add_argument("--variant-label", type=str, default="WO_HARD")
    p.add_argument("--num-nodes", type=int, default=3)
    p.add_argument("--min-node-gap-days", type=int, default=14)
    p.add_argument("--top-tickers-per-node", type=int, default=4)
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
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/compare/case_study/focused_case"),
    )
    p.add_argument("--dpi", type=int, default=320)
    return p.parse_args()


def _platform(source: str) -> str:
    return "YouTube" if source.lower() == "youtube" else "X"


def _find_latest_kicl_run(kicl_root: Path, source: str, kol: str) -> Path:
    root = kicl_root / source
    runs = sorted(root.glob(f"{kol}_*"))
    if not runs:
        raise FileNotFoundError(f"No KICL run found: {root}/{kol}_*")
    return runs[-1]


def _load_kicl_equity(case_root: Path, kicl_root: Path, source: str, kol: str) -> tuple[pd.DataFrame, Path]:
    # Prefer benchmark test-root run.
    run = _find_latest_kicl_run(kicl_root, source, kol)
    p = run / "daily" / "equity_daily.csv"
    if not p.exists():
        # fallback to old case-root cache
        p = case_root / "raw_kicl" / source / kol / "equity_daily.csv"
        run = case_root / "raw_kicl" / source / kol
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    req = {"date", "equity_baseline", "equity_trained"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"{p} missing columns: {sorted(miss)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df, run


def _latest_run(ab_root: Path, source: str, kol: str) -> Path:
    root = ab_root / source
    runs = sorted(root.glob(f"{kol}_*"))
    if not runs:
        raise FileNotFoundError(f"No run found: {root}/{kol}_*")
    return runs[-1]


def _load_ab_equity(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "daily" / "equity_daily.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    req = {"date", "equity_trained"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"{p} missing columns: {sorted(miss)}")
    df = df[["date", "equity_trained"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df.rename(columns={"equity_trained": "equity_variant"})


def _load_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    req = {"date", "ticker", "reward", "baseline_action", "policy_action"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")
    df["event_day"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.dropna(subset=["event_day"]).copy()
    return df


def _load_discourse(trace_path: Path) -> pd.DataFrame:
    if not trace_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(trace_path)
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.floor("D")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def _load_reward_discourse(source: str, kol: str, x_root: Path, yt_root: Path) -> pd.DataFrame:
    root = x_root if source.lower() == "x" else yt_root
    p = root / kol / "test.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    keep = [c for c in ["trading_day", "ticker", "title", "text", "sentiment", "confidence", "event_id", "published_at"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    out = df[keep].copy()
    if "trading_day" in out.columns:
        out["day"] = pd.to_datetime(out["trading_day"], errors="coerce").dt.floor("D")
    else:
        out["day"] = pd.NaT
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.upper()
    if "text_preview" not in out.columns:
        if "text" in out.columns:
            out["text_preview"] = out["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 260)
        else:
            out["text_preview"] = ""
    return out.dropna(subset=["day"]).copy()


def _pick_nodes(df: pd.DataFrame, num_nodes: int, min_gap_days: int) -> list[Node]:
    picks: list[Node] = []

    # One anchor from each contrast first.
    anchors = []
    for contrast, dcol, gcol in [
        ("ours_vs_baseline", "d_gap_bl", "gap_bl"),
        ("ours_vs_variant", "d_gap_var", "gap_var"),
    ]:
        sub = df[df[dcol] > 0].sort_values(dcol, ascending=False)
        if not sub.empty:
            r = sub.iloc[0]
            anchors.append(Node(0, contrast, pd.Timestamp(r["date"]), float(r[dcol]), float(r[gcol])))

    for n in anchors:
        if not any(abs((n.focus_day - x.focus_day).days) < min_gap_days for x in picks):
            picks.append(n)

    # Fill remaining globally by positive d_gap strength.
    cand_rows = []
    for contrast, dcol, gcol in [
        ("ours_vs_baseline", "d_gap_bl", "gap_bl"),
        ("ours_vs_variant", "d_gap_var", "gap_var"),
    ]:
        sub = df[df[dcol] > 0].copy()
        sub["contrast"] = contrast
        sub["dcol"] = dcol
        sub["gcol"] = gcol
        cand_rows.append(sub)
    cand = pd.concat(cand_rows, ignore_index=True) if cand_rows else pd.DataFrame()
    if not cand.empty:
        cand = cand.sort_values("d_gap_bl" if "d_gap_bl" in cand.columns else "date", ascending=False)
        # robust sorting by per-row selected d value
        cand["_score"] = np.where(
            cand["contrast"].eq("ours_vs_baseline"),
            cand["d_gap_bl"],
            cand["d_gap_var"],
        )
        cand = cand.sort_values("_score", ascending=False)
        for _, r in cand.iterrows():
            if len(picks) >= num_nodes:
                break
            day = pd.Timestamp(r["date"])
            if any(abs((day - x.focus_day).days) < min_gap_days for x in picks):
                continue
            if r["contrast"] == "ours_vs_baseline":
                d, g = float(r["d_gap_bl"]), float(r["gap_bl"])
            else:
                d, g = float(r["d_gap_var"]), float(r["gap_var"])
            picks.append(Node(0, str(r["contrast"]), day, d, g))

    picks = picks[: max(1, num_nodes)]
    for i, p in enumerate(picks, start=1):
        p.node_id = i
    return picks


def _event_day_map(days: Iterable[pd.Timestamp], focus_day: pd.Timestamp) -> pd.Timestamp | None:
    d = sorted(set(pd.Timestamp(x).floor("D") for x in days))
    if not d:
        return None
    cands = [x for x in d if x <= focus_day]
    if not cands:
        return None
    return cands[-1]


def _find_discourse_row(disc: pd.DataFrame, event_day: pd.Timestamp, ticker: str) -> pd.Series | None:
    if disc.empty:
        return None
    hit = disc[(disc["day"] == event_day) & (disc["ticker"] == ticker)]
    if hit.empty:
        hit = disc[(disc["ticker"] == ticker) & (disc["day"] <= event_day)].sort_values("day", ascending=False).head(1)
    if hit.empty:
        hit = disc[disc["day"] <= event_day].sort_values("day", ascending=False).head(1)
    if hit.empty:
        return None
    return hit.iloc[0]


def _build_node_table(
    nodes: list[Node],
    kicl_pos: pd.DataFrame,
    var_pos: pd.DataFrame,
    disc: pd.DataFrame,
    source: str,
    kol: str,
    variant_label: str,
    top_tickers: int,
) -> pd.DataFrame:
    rows = []
    k_event_days = kicl_pos["event_day"].tolist()

    # align variant positions by day+ticker for comparator action
    var_small = (
        var_pos[["event_day", "ticker", "policy_action"]]
        .rename(columns={"policy_action": "variant_action"})
        .copy()
    )

    for n in nodes:
        event_day = _event_day_map(k_event_days, n.focus_day)
        if event_day is None:
            continue

        day_df = kicl_pos[kicl_pos["event_day"] == event_day].copy()
        if day_df.empty:
            continue
        day_df = day_df.merge(var_small[var_small["event_day"] == event_day], on=["event_day", "ticker"], how="left")
        day_df["variant_action"] = day_df["variant_action"].fillna(0.0)

        if n.contrast == "ours_vs_baseline":
            day_df["contrast_contrib"] = day_df["reward"] * (day_df["policy_action"] - day_df["baseline_action"])
            comp_col = "baseline_action"
            comp_name = "Baseline"
        else:
            day_df["contrast_contrib"] = day_df["reward"] * (day_df["policy_action"] - day_df["variant_action"])
            comp_col = "variant_action"
            comp_name = variant_label

        pool = day_df[day_df["contrast_contrib"] > 1e-8].sort_values("contrast_contrib", ascending=False).head(
            max(20, top_tickers * 4)
        ).copy()
        if pool.empty:
            pool = day_df.sort_values("contrast_contrib", ascending=False).head(max(20, top_tickers * 4)).copy()
        picked_rows = []
        fallback_rows = []
        for _, r in pool.iterrows():
            drow = _find_discourse_row(disc, event_day, str(r["ticker"]))
            if drow is not None and (
                str(drow.get("text_preview", "")).strip()
                or str(drow.get("text", "")).strip()
                or str(drow.get("title", "")).strip()
            ):
                picked_rows.append((r, drow))
            else:
                fallback_rows.append((r, drow))

        chosen = picked_rows[:top_tickers]
        if len(chosen) < top_tickers:
            chosen.extend(fallback_rows[: (top_tickers - len(chosen))])

        for r, drow in chosen:
            rows.append(
                {
                    "source": source,
                    "kol": kol,
                    "node_id": n.node_id,
                    "contrast": n.contrast,
                    "focus_day": n.focus_day.date().isoformat(),
                    "mapped_event_day": event_day.date().isoformat(),
                    "node_d_gap": n.d_gap,
                    "node_gap": n.gap,
                    "ticker": r["ticker"],
                    "reward": float(r["reward"]),
                    "baseline_action": float(r["baseline_action"]),
                    "ours_action": float(r["policy_action"]),
                    "variant_action": float(r["variant_action"]),
                    "comparator_name": comp_name,
                    "comparator_action": float(r[comp_col]),
                    "contrast_contribution": float(r["contrast_contrib"]),
                    "sentiment": (None if drow is None else drow.get("sentiment")),
                    "confidence": (None if drow is None else drow.get("confidence")),
                    "title": (None if drow is None else drow.get("title")),
                    "text_preview": (None if drow is None else drow.get("text_preview")),
                    "event_id": (None if drow is None else drow.get("event_id")),
                    "published_at": (None if drow is None else drow.get("published_at")),
                }
            )

    return pd.DataFrame(rows)


def _draw_circle(ax: plt.Axes, x_day: pd.Timestamp, y: float, x_days: int, y_span: float) -> None:
    x = mdates.date2num(x_day)
    patch = Ellipse((x, y), width=x_days, height=y_span, fill=False, edgecolor="red", linewidth=1.8, alpha=0.9, zorder=8)
    ax.add_patch(patch)


def _plot_dual(
    merged: pd.DataFrame,
    nodes: list[Node],
    source: str,
    kol: str,
    variant_label: str,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharex=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.22, color="#bdbdbd")
        ax.grid(True, which="minor", linestyle="-", linewidth=0.35, alpha=0.16, color="#d6d6d6")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelsize=8.5)

    # Left: baseline vs ours
    axes[0].plot(merged["date"], merged["equity_baseline"], color="#2E86DE", lw=1.2, ls="--", label="Baseline")
    axes[0].plot(merged["date"], merged["equity_trained"], color="#f39c12", lw=1.55, label="KICL (Ours)")
    axes[0].set_title("Ours vs Baseline", fontsize=12, pad=8)
    axes[0].set_ylabel("Equity")
    axes[0].legend(loc="upper left", fontsize=9.5, frameon=True)

    # Right: variant vs ours
    axes[1].plot(merged["date"], merged["equity_variant"], color="#8e44ad", lw=1.2, ls="--", label=variant_label)
    axes[1].plot(merged["date"], merged["equity_trained"], color="#f39c12", lw=1.55, label="KICL (Ours)")
    axes[1].set_title(f"Ours vs {variant_label}", fontsize=12, pad=8)
    axes[1].legend(loc="upper left", fontsize=9.5, frameon=True)
    axes[1].tick_params(axis="y", labelleft=False)

    # common y-lims
    y_all = pd.concat([merged["equity_baseline"], merged["equity_trained"], merged["equity_variant"]], axis=0).dropna()
    ymin, ymax = float(y_all.min()), float(y_all.max())
    ypad = max(0.01, 0.045 * (ymax - ymin))
    for ax in axes:
        ax.set_ylim(ymin - ypad, ymax + ypad)

    y_span = (ymax - ymin) * 0.35
    if y_span <= 0:
        y_span = 0.04

    # callouts
    for n in nodes:
        row = merged[merged["date"] == n.focus_day]
        if row.empty:
            continue
        y_left = float(row["equity_trained"].iloc[0])
        y_right = float(row["equity_trained"].iloc[0])
        if n.contrast == "ours_vs_baseline":
            _draw_circle(axes[0], n.focus_day, y_left, x_days=36, y_span=y_span)
            axes[0].text(n.focus_day, y_left + y_span * 0.55, f"#{n.node_id}", color="red", fontsize=10, ha="center", fontweight="bold")
        else:
            _draw_circle(axes[1], n.focus_day, y_right, x_days=36, y_span=y_span)
            axes[1].text(n.focus_day, y_right + y_span * 0.55, f"#{n.node_id}", color="red", fontsize=10, ha="center", fontweight="bold")

    sup = f"{_platform(source)} · {kol.replace('_', ' ')}"
    fig.suptitle(sup, fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    source = args.source
    kol = args.kol
    variant_label = args.variant_label

    kicl_eq, kicl_run_dir = _load_kicl_equity(args.case_root, args.kicl_root, source, kol)
    run_dir = _latest_run(args.ab_root, source, kol)
    var_eq = _load_ab_equity(run_dir)

    merged = (
        kicl_eq.merge(var_eq, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    merged["gap_bl"] = merged["equity_trained"] - merged["equity_baseline"]
    merged["d_gap_bl"] = merged["gap_bl"].diff().fillna(0.0)
    merged["gap_var"] = merged["equity_trained"] - merged["equity_variant"]
    merged["d_gap_var"] = merged["gap_var"].diff().fillna(0.0)

    nodes = _pick_nodes(merged, args.num_nodes, args.min_node_gap_days)
    if not nodes:
        raise RuntimeError("No divergence nodes found.")

    pos_path = kicl_run_dir / "event" / "positions_test.csv"
    if not pos_path.exists():
        pos_path = args.case_root / "raw_kicl" / source / kol / "positions_test.csv"
    kicl_pos = _load_positions(pos_path)
    var_pos = _load_positions(run_dir / "event" / "positions_test.csv")
    disc_trace = _load_discourse(args.case_root / "trace" / source / kol / "kicl_discourse_evidence.csv")
    disc_reward = _load_reward_discourse(
        source=source,
        kol=kol,
        x_root=args.x_reward_root,
        yt_root=args.youtube_reward_root,
    )
    if disc_reward.empty and disc_trace.empty:
        disc = pd.DataFrame()
    elif disc_reward.empty:
        disc = disc_trace
    elif disc_trace.empty:
        disc = disc_reward
    else:
        disc = pd.concat([disc_reward, disc_trace], ignore_index=True, sort=False)
        disc = disc.sort_values("day").reset_index(drop=True)

    table = _build_node_table(
        nodes=nodes,
        kicl_pos=kicl_pos,
        var_pos=var_pos,
        disc=disc,
        source=source,
        kol=kol,
        variant_label=variant_label,
        top_tickers=args.top_tickers_per_node,
    )

    out_root = args.output_dir / source / kol
    fig_png = out_root / "case_dual_contrast.png"
    fig_pdf = out_root / "case_dual_contrast.pdf"
    _plot_dual(
        merged=merged,
        nodes=nodes,
        source=source,
        kol=kol,
        variant_label=variant_label,
        out_png=fig_png,
        out_pdf=fig_pdf,
        dpi=args.dpi,
    )

    nodes_df = pd.DataFrame(
        [
            {
                "node_id": n.node_id,
                "contrast": n.contrast,
                "focus_day": n.focus_day.date().isoformat(),
                "d_gap": n.d_gap,
                "gap": n.gap,
            }
            for n in nodes
        ]
    )
    out_root.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_root / "case_dual_timeseries.csv", index=False)
    nodes_df.to_csv(out_root / "case_dual_nodes.csv", index=False)
    table.to_csv(out_root / "case_dual_node_evidence.csv", index=False)

    # concise md for paper writing
    md = []
    md.append(f"# Focused Case: {source}/{kol}")
    md.append("")
    md.append(f"- KICL run: `{kicl_run_dir.name}`")
    md.append(f"- Variant comparator: `{variant_label}` (run: `{run_dir.name}`)")
    md.append(f"- Figure: `{fig_png}`")
    md.append("")
    md.append("## Key Nodes")
    for _, r in nodes_df.iterrows():
        md.append(
            f"- Node #{int(r['node_id'])}: `{r['contrast']}` on `{r['focus_day']}`, "
            f"d_gap={float(r['d_gap']):.4f}, gap={float(r['gap']):.4f}"
        )
    md.append("")
    md.append("## Evidence Table")
    md.append("- File: `case_dual_node_evidence.csv`")
    md.append("- Columns include: sentiment, text_preview, baseline_action, ours_action, variant_action.")
    (out_root / "CASE_DUAL_NOTES.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Saved: {fig_png}")
    print(f"Saved: {fig_pdf}")
    print(f"Saved: {out_root / 'case_dual_nodes.csv'}")
    print(f"Saved: {out_root / 'case_dual_node_evidence.csv'}")
    print(f"Saved: {out_root / 'CASE_DUAL_NOTES.md'}")


if __name__ == "__main__":
    main()
