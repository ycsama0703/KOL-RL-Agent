#!/usr/bin/env python3
"""Build one focused case-study figure + evidence table for a single contrast.

Contrasts:
  - baseline: Ours vs Baseline
  - variant : Ours vs an ablation variant (e.g., WO_HARD)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TH = 0.02


@dataclass
class Node:
    node_id: int
    focus_day: pd.Timestamp
    d_gap: float
    gap: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-root", type=Path, default=Path("benchmarks/compare/case_study"))
    p.add_argument(
        "--kicl-root",
        type=Path,
        default=Path("benchmarks/bench_test_results/multisource_test_mainline_xrefresh"),
    )
    p.add_argument(
        "--ab-root",
        type=Path,
        default=Path("ablation study/ab_test_results/w_no_hard"),
        help="Needed for contrast=variant.",
    )
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--kol", type=str, required=True)
    p.add_argument("--contrast", type=str, choices=["baseline", "variant"], required=True)
    p.add_argument("--variant-label", type=str, default="WO_HARD")
    p.add_argument("--num-nodes", type=int, default=3)
    p.add_argument("--min-gap-days", type=int, default=14)
    p.add_argument(
        "--focus-days",
        type=str,
        default="",
        help="Optional comma-separated YYYY-MM-DD list. If set, use these days as nodes instead of auto-pick.",
    )
    p.add_argument("--top-tickers", type=int, default=5)
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
        default=Path("benchmarks/compare/case_study/focused_case_single"),
    )
    p.add_argument("--dpi", type=int, default=320)
    return p.parse_args()


def _platform(source: str) -> str:
    return "YouTube" if source.lower() == "youtube" else "X"


def _latest_run(root: Path, source: str, kol: str) -> Path:
    s = root / source
    runs = sorted(s.glob(f"{kol}_*"))
    if not runs:
        raise FileNotFoundError(f"No run found in {s} for {kol}")
    return runs[-1]


def _load_eq(path: Path, col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "equity_trained" not in df.columns:
        raise ValueError(f"{path} missing required columns")
    out = df[["date", "equity_trained"]].copy().rename(columns={"equity_trained": col_name})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.floor("D")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def _load_kicl_eq(case_root: Path, kicl_root: Path, source: str, kol: str) -> tuple[pd.DataFrame, Path]:
    run = _latest_run(kicl_root, source, kol)
    p = run / "daily" / "equity_daily.csv"
    if p.exists():
        df = pd.read_csv(p)
        req = {"date", "equity_baseline", "equity_trained"}
        if not req.issubset(df.columns):
            raise ValueError(f"{p} missing columns: {sorted(req.difference(df.columns))}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df, run

    # fallback
    p2 = case_root / "raw_kicl" / source / kol / "equity_daily.csv"
    if not p2.exists():
        raise FileNotFoundError(p2)
    df = pd.read_csv(p2)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df, (case_root / "raw_kicl" / source / kol)


def _load_pos(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"date", "ticker", "reward", "baseline_action", "policy_action"}
    if not req.issubset(df.columns):
        raise ValueError(f"{path} missing columns: {sorted(req.difference(df.columns))}")
    df["event_day"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["event_day"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def _map_event_day(days: Iterable[pd.Timestamp], focus_day: pd.Timestamp) -> pd.Timestamp | None:
    d = sorted(set(pd.Timestamp(x).floor("D") for x in days))
    cands = [x for x in d if x <= focus_day]
    return cands[-1] if cands else None


def _load_reward_discourse(source: str, kol: str, x_root: Path, yt_root: Path) -> pd.DataFrame:
    root = x_root if source.lower() == "x" else yt_root
    p = root / kol / "test.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "trading_day" not in df.columns:
        return pd.DataFrame()
    keep = [c for c in ["trading_day", "ticker", "title", "text", "sentiment", "confidence", "event_id", "published_at"] if c in df.columns]
    out = df[keep].copy()
    out["day"] = pd.to_datetime(out["trading_day"], errors="coerce").dt.floor("D")
    out["ticker"] = out["ticker"].astype(str).str.upper()
    if "text_preview" not in out.columns:
        if "text" in out.columns:
            out["text_preview"] = out["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 260)
        else:
            out["text_preview"] = ""
    return out.dropna(subset=["day"])


def _load_trace_discourse(case_root: Path, source: str, kol: str) -> pd.DataFrame:
    p = case_root / "trace" / source / kol / "kicl_discourse_evidence.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.floor("D")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def _pick_nodes(ts: pd.DataFrame, num_nodes: int, min_gap_days: int) -> list[Node]:
    sub = ts[ts["d_gap"] > 0].sort_values("d_gap", ascending=False)
    out: list[Node] = []
    for _, r in sub.iterrows():
        d = pd.Timestamp(r["date"])
        if any(abs((d - x.focus_day).days) < min_gap_days for x in out):
            continue
        out.append(Node(node_id=len(out) + 1, focus_day=d, d_gap=float(r["d_gap"]), gap=float(r["gap"])))
        if len(out) >= num_nodes:
            break
    return out


def _pick_nodes_manual(ts: pd.DataFrame, focus_days_raw: str) -> list[Node]:
    days_raw = [x.strip() for x in str(focus_days_raw).split(",") if x.strip()]
    if not days_raw:
        return []
    out: list[Node] = []
    dser = ts["date"]
    for i, d in enumerate(days_raw, start=1):
        target = pd.Timestamp(d).floor("D")
        hit = ts[ts["date"] == target]
        if hit.empty:
            # fallback to nearest day in timeseries
            idx = (dser - target).abs().idxmin()
            hit = ts.loc[[idx]]
        r = hit.iloc[0]
        out.append(
            Node(
                node_id=i,
                focus_day=pd.Timestamp(r["date"]).floor("D"),
                d_gap=float(r["d_gap"]),
                gap=float(r["gap"]),
            )
        )
    return out


def _find_discourse(disc: pd.DataFrame, day: pd.Timestamp, ticker: str) -> pd.Series | None:
    if disc.empty:
        return None
    h = disc[(disc["day"] == day) & (disc["ticker"] == ticker)]
    if h.empty:
        h = disc[(disc["ticker"] == ticker) & (disc["day"] <= day)].sort_values("day", ascending=False).head(1)
    if h.empty:
        h = disc[disc["day"] <= day].sort_values("day", ascending=False).head(1)
    if h.empty:
        return None
    return h.iloc[0]


def _alignment(sentiment: float | None, ours: float, comp: float, contrast: str) -> float | np.nan:
    if sentiment is None or (isinstance(sentiment, float) and np.isnan(sentiment)):
        return np.nan
    if contrast == "baseline":
        if sentiment >= 0.2:
            return 1.0 if ours >= comp - 1e-9 else 0.0
        if sentiment <= -0.2:
            return 1.0 if ours <= comp + 1e-9 else 0.0
        return 1.0
    # variant contrast: check ours action sign with sentiment
    if sentiment >= 0.2:
        return 1.0 if ours > TH else 0.0
    if sentiment <= -0.2:
        return 1.0 if abs(ours) <= TH else 0.0
    return 1.0


def _build_evidence(
    nodes: list[Node],
    kpos: pd.DataFrame,
    comp_pos: pd.DataFrame | None,
    disc: pd.DataFrame,
    source: str,
    kol: str,
    contrast: str,
    variant_label: str,
    top_tickers: int,
) -> pd.DataFrame:
    rows = []
    comp_small = None
    if comp_pos is not None:
        comp_small = (
            comp_pos[["event_day", "ticker", "policy_action"]]
            .rename(columns={"policy_action": "variant_action"})
            .copy()
        )

    for n in nodes:
        ed = _map_event_day(kpos["event_day"], n.focus_day)
        if ed is None:
            continue
        q = kpos[kpos["event_day"] == ed].copy()
        if contrast == "variant":
            if comp_small is None:
                continue
            q = q.merge(comp_small[comp_small["event_day"] == ed], on=["event_day", "ticker"], how="left")
            q["variant_action"] = q["variant_action"].fillna(0.0)
            q["comp_action"] = q["variant_action"]
            q["comp_name"] = variant_label
        else:
            q["variant_action"] = np.nan
            q["comp_action"] = q["baseline_action"]
            q["comp_name"] = "Baseline"

        q["contrast_contribution"] = q["reward"] * (q["policy_action"] - q["comp_action"])
        q = q[q["contrast_contribution"] > 1e-8].sort_values("contrast_contribution", ascending=False).head(max(top_tickers * 4, 20))
        if q.empty:
            continue

        # attach discourse + keep rows with text first
        picks = []
        for _, r in q.iterrows():
            drow = _find_discourse(disc, ed, str(r["ticker"]))
            sent = None if drow is None else drow.get("sentiment")
            ali = _alignment(sent, float(r["policy_action"]), float(r["comp_action"]), contrast)
            picks.append(
                {
                    "node_id": n.node_id,
                    "source": source,
                    "kol": kol,
                    "contrast": contrast,
                    "focus_day": n.focus_day.date().isoformat(),
                    "mapped_event_day": ed.date().isoformat(),
                    "d_gap": n.d_gap,
                    "gap": n.gap,
                    "ticker": r["ticker"],
                    "reward": float(r["reward"]),
                    "baseline_action": float(r["baseline_action"]),
                    "ours_action": float(r["policy_action"]),
                    "variant_action": (float(r["variant_action"]) if pd.notna(r["variant_action"]) else np.nan),
                    "comparator_name": r["comp_name"],
                    "comparator_action": float(r["comp_action"]),
                    "contrast_contribution": float(r["contrast_contribution"]),
                    "sentiment": sent,
                    "confidence": (None if drow is None else drow.get("confidence")),
                    "title": (None if drow is None else drow.get("title")),
                    "text_preview": (None if drow is None else drow.get("text_preview")),
                    "event_id": (None if drow is None else drow.get("event_id")),
                    "published_at": (None if drow is None else drow.get("published_at")),
                    "sentiment_action_aligned": ali,
                }
            )
        dfe = pd.DataFrame(picks)
        # select rows: aligned first, then highest contribution
        dfe["has_text"] = dfe["text_preview"].fillna("").astype(str).str.strip().ne("") | dfe["title"].fillna("").astype(str).str.strip().ne("")
        dfe = dfe.sort_values(["sentiment_action_aligned", "has_text", "contrast_contribution"], ascending=[False, False, False])
        rows.extend(dfe.head(top_tickers).drop(columns=["has_text"]).to_dict(orient="records"))

    return pd.DataFrame(rows)


def _plot(ts: pd.DataFrame, nodes: list[Node], source: str, kol: str, comp_label: str, out_png: Path, out_pdf: Path, dpi: int) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(ts["date"], ts["comp"], color="#2E86DE", lw=1.2, ls="--", label=comp_label)
    ax.plot(ts["date"], ts["ours"], color="#f39c12", lw=1.6, label="KICL (Ours)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.22, color="#bdbdbd")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.35, alpha=0.16, color="#d6d6d6")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=9)

    y_all = pd.concat([ts["ours"], ts["comp"]], axis=0).dropna()
    ymin, ymax = float(y_all.min()), float(y_all.max())
    ypad = max(0.01, 0.05 * (ymax - ymin))
    ax.set_ylim(ymin - ypad, ymax + ypad)
    y_span = max(0.03, 0.30 * (ymax - ymin))

    for n in nodes:
        rr = ts[ts["date"] == n.focus_day]
        if rr.empty:
            continue
        y = float(rr["ours"].iloc[0])
        e = Ellipse(
            (mdates.date2num(n.focus_day), y),
            width=35,
            height=y_span,
            fill=False,
            edgecolor="red",
            linewidth=1.8,
            alpha=0.9,
            zorder=7,
        )
        ax.add_patch(e)
        ax.text(n.focus_day, y + y_span * 0.55, f"#{n.node_id}", color="red", fontsize=10, ha="center", fontweight="bold")

    ax.set_title(f"{_platform(source)} · {kol.replace('_', ' ')}", fontsize=17, fontweight="bold", pad=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    source, kol = args.source, args.kol
    kicl_eq, kicl_run = _load_kicl_eq(args.case_root, args.kicl_root, source, kol)
    kpos_path = kicl_run / "event" / "positions_test.csv"
    if not kpos_path.exists():
        kpos_path = args.case_root / "raw_kicl" / source / kol / "positions_test.csv"
    kpos = _load_pos(kpos_path)

    # comparator
    if args.contrast == "baseline":
        ts = kicl_eq[["date", "equity_trained", "equity_baseline"]].rename(
            columns={"equity_trained": "ours", "equity_baseline": "comp"}
        )
        ts["gap"] = ts["ours"] - ts["comp"]
        comp_label = "Baseline"
        variant_run = None
        cpos = None
    else:
        variant_run = _latest_run(args.ab_root, source, kol)
        veq = _load_eq(variant_run / "daily" / "equity_daily.csv", "comp")
        ts = kicl_eq[["date", "equity_trained"]].rename(columns={"equity_trained": "ours"}).merge(veq, on="date", how="inner")
        ts["gap"] = ts["ours"] - ts["comp"]
        comp_label = args.variant_label
        cpos = _load_pos(variant_run / "event" / "positions_test.csv")

    ts = ts.sort_values("date").reset_index(drop=True)
    ts["d_gap"] = ts["gap"].diff().fillna(0.0)
    if args.focus_days.strip():
        nodes = _pick_nodes_manual(ts, args.focus_days)
    else:
        nodes = _pick_nodes(ts, args.num_nodes, args.min_gap_days)
    if not nodes:
        raise RuntimeError("No positive divergence nodes found.")

    disc_reward = _load_reward_discourse(source, kol, args.x_reward_root, args.youtube_reward_root)
    disc_trace = _load_trace_discourse(args.case_root, source, kol)
    if disc_reward.empty and disc_trace.empty:
        disc = pd.DataFrame()
    else:
        disc = pd.concat([disc_reward, disc_trace], ignore_index=True, sort=False).drop_duplicates()

    evidence = _build_evidence(
        nodes=nodes,
        kpos=kpos,
        comp_pos=cpos,
        disc=disc,
        source=source,
        kol=kol,
        contrast=args.contrast,
        variant_label=args.variant_label,
        top_tickers=args.top_tickers,
    )

    out = args.output_dir / args.contrast / source / kol
    fig_png = out / "case_single_contrast.png"
    fig_pdf = out / "case_single_contrast.pdf"
    _plot(ts, nodes, source, kol, comp_label, fig_png, fig_pdf, args.dpi)

    nodes_df = pd.DataFrame([{"node_id": n.node_id, "focus_day": n.focus_day.date().isoformat(), "d_gap": n.d_gap, "gap": n.gap} for n in nodes])
    out.mkdir(parents=True, exist_ok=True)
    ts.to_csv(out / "case_single_timeseries.csv", index=False)
    nodes_df.to_csv(out / "case_single_nodes.csv", index=False)
    evidence.to_csv(out / "case_single_evidence_all.csv", index=False)
    evidence[evidence["sentiment_action_aligned"] == 1.0].to_csv(out / "case_single_evidence_aligned.csv", index=False)

    md = []
    md.append(f"# Case ({args.contrast}) {source}/{kol}")
    md.append("")
    md.append(f"- KICL run: `{kicl_run.name}`")
    if variant_run is not None:
        md.append(f"- Variant run: `{variant_run.name}`")
    md.append(f"- Figure: `{fig_png}`")
    md.append("")
    md.append("## Nodes")
    for _, r in nodes_df.iterrows():
        md.append(f"- #{int(r['node_id'])}: day={r['focus_day']}, d_gap={float(r['d_gap']):.4f}, gap={float(r['gap']):.4f}")
    md.append("")
    md.append("- Evidence (all): `case_single_evidence_all.csv`")
    md.append("- Evidence (sentiment-action aligned only): `case_single_evidence_aligned.csv`")
    (out / "CASE_SINGLE_NOTES.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Saved: {fig_png}")
    print(f"Saved: {fig_pdf}")
    print(f"Saved: {out / 'case_single_nodes.csv'}")
    print(f"Saved: {out / 'case_single_evidence_aligned.csv'}")


if __name__ == "__main__":
    main()
