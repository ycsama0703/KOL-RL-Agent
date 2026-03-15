"""Report signal-vs-silence statistics from replay buffers.

Core definitions (default):
- signal sample:  abs(baseline_action_t) > signal_threshold
- silence sample: abs(baseline_action_t) <= signal_threshold

Outputs:
1) overall_counts.csv
2) per_ticker_counts.csv
3) per_kol_counts.csv
4) report.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute D_sig / D_sil statistics from replay buffers.")
    parser.add_argument(
        "--replay-root",
        required=True,
        help="Root directory containing replay .pt files (typically <kol>/<split>.pt).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output tables and markdown summary.",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=1e-8,
        help="Threshold for deciding signal vs silence from action magnitude.",
    )
    parser.add_argument(
        "--signal-key",
        default="auto",
        choices=["auto", "baseline_actions", "baseline_action", "actions"],
        help="Which field in replay buffer to use as the signal variable.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma separated split names to include. Empty means include all.",
    )
    return parser.parse_args()


def find_replay_files(root: Path, allow_splits: set[str]) -> List[Path]:
    files: List[Path] = []
    for path in sorted(root.rglob("*.pt")):
        split = path.stem
        if allow_splits and split not in allow_splits:
            continue
        files.append(path)
    return files


def choose_signal_key(data: Dict, requested: str) -> str:
    if requested != "auto":
        if requested not in data:
            raise KeyError(f"Requested signal key '{requested}' not found in replay buffer.")
        return requested
    for key in ("baseline_actions", "baseline_action", "actions"):
        if key in data:
            return key
    raise KeyError("No valid signal key found in replay buffer (expected baseline_actions/baseline_action/actions).")


def as_1d_float(x) -> np.ndarray:
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.astype(np.float64).reshape(-1)


def safe_ratio(num: int, den: int) -> float:
    if den == 0:
        if num == 0:
            return np.nan
        return np.inf
    return float(num) / float(den)


def make_count_row(scope: str, d_sig: int, d_sil: int) -> Dict:
    total = int(d_sig + d_sil)
    return {
        "scope": scope,
        "d_sig": int(d_sig),
        "d_sil": int(d_sil),
        "d_total": total,
        "rho_sig": (float(d_sig) / total) if total > 0 else np.nan,
        "rho_sil": (float(d_sil) / total) if total > 0 else np.nan,
        "sil_sig_ratio": safe_ratio(d_sil, d_sig),
    }


def percentile(series: pd.Series, q: float) -> float:
    if series.empty:
        return np.nan
    return float(series.quantile(q))


def main() -> None:
    args = parse_args()
    replay_root = Path(args.replay_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_set = {s.strip() for s in args.splits.split(",") if s.strip()}
    files = find_replay_files(replay_root, split_set)
    if not files:
        raise FileNotFoundError(f"No replay .pt files found under {replay_root}")

    # Aggregators
    global_sig = 0
    global_sil = 0
    by_split = defaultdict(lambda: {"sig": 0, "sil": 0})
    by_kol = defaultdict(lambda: {"sig": 0, "sil": 0})
    by_kol_split = defaultdict(lambda: {"sig": 0, "sil": 0})
    by_ticker_sig = defaultdict(int)
    by_ticker_sil = defaultdict(int)

    file_rows = []
    signal_key_used = None

    for path in files:
        data = torch.load(path, map_location="cpu")
        signal_key = choose_signal_key(data, args.signal_key)
        signal_key_used = signal_key if signal_key_used is None else signal_key_used

        signal_values = as_1d_float(data[signal_key])
        n = int(signal_values.shape[0])
        is_sig = np.abs(signal_values) > args.signal_threshold
        is_sil = ~is_sig
        sig_count = int(is_sig.sum())
        sil_count = int(is_sil.sum())

        split = path.stem
        kol = path.parent.name
        global_sig += sig_count
        global_sil += sil_count
        by_split[split]["sig"] += sig_count
        by_split[split]["sil"] += sil_count
        by_kol[kol]["sig"] += sig_count
        by_kol[kol]["sil"] += sil_count
        by_kol_split[(kol, split)]["sig"] += sig_count
        by_kol_split[(kol, split)]["sil"] += sil_count

        tickers = None
        meta = data.get("meta", {})
        if isinstance(meta, dict) and "ticker" in meta:
            tickers = np.asarray(meta["ticker"], dtype=object).reshape(-1)
            if tickers.shape[0] != n:
                m = min(tickers.shape[0], n)
                tickers = tickers[:m]
                is_sig_local = is_sig[:m]
                is_sil_local = is_sil[:m]
            else:
                is_sig_local = is_sig
                is_sil_local = is_sil

            sig_ticker_counts = pd.Series(tickers[is_sig_local]).value_counts()
            sil_ticker_counts = pd.Series(tickers[is_sil_local]).value_counts()
            for ticker, c in sig_ticker_counts.items():
                by_ticker_sig[str(ticker)] += int(c)
            for ticker, c in sil_ticker_counts.items():
                by_ticker_sil[str(ticker)] += int(c)

        file_rows.append(
            {
                "file": str(path),
                "kol": kol,
                "split": split,
                "signal_key": signal_key,
                "d_sig": sig_count,
                "d_sil": sil_count,
                "d_total": n,
                "rho_sig": sig_count / n if n > 0 else np.nan,
                "rho_sil": sil_count / n if n > 0 else np.nan,
                "sil_sig_ratio": safe_ratio(sil_count, sig_count),
            }
        )

    # Overall table
    overall_rows = [make_count_row("ALL", global_sig, global_sil)]
    for split in sorted(by_split.keys()):
        row = by_split[split]
        overall_rows.append(make_count_row(f"SPLIT::{split}", row["sig"], row["sil"]))
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(output_dir / "overall_counts.csv", index=False)

    # Per-KOL
    kol_rows = []
    for kol in sorted(by_kol.keys()):
        row = by_kol[kol]
        out = make_count_row(kol, row["sig"], row["sil"])
        out["kol"] = kol
        kol_rows.append(out)
    per_kol_df = pd.DataFrame(kol_rows).sort_values("d_total", ascending=False)
    per_kol_df.to_csv(output_dir / "per_kol_counts.csv", index=False)

    # Per-KOL-Split (useful audit)
    kol_split_rows = []
    for (kol, split), row in sorted(by_kol_split.items()):
        out = make_count_row(f"{kol}::{split}", row["sig"], row["sil"])
        out["kol"] = kol
        out["split"] = split
        kol_split_rows.append(out)
    per_kol_split_df = pd.DataFrame(kol_split_rows).sort_values(["kol", "split"])
    per_kol_split_df.to_csv(output_dir / "per_kol_split_counts.csv", index=False)

    # Per-ticker
    tickers = sorted(set(by_ticker_sig.keys()) | set(by_ticker_sil.keys()))
    ticker_rows = []
    for t in tickers:
        d_sig = by_ticker_sig.get(t, 0)
        d_sil = by_ticker_sil.get(t, 0)
        row = make_count_row(t, d_sig, d_sil)
        row["ticker"] = t
        row["sil_gt_sig"] = bool(d_sil > d_sig)
        ticker_rows.append(row)
    per_ticker_df = pd.DataFrame(ticker_rows).sort_values("d_total", ascending=False)
    per_ticker_df.to_csv(output_dir / "per_ticker_counts.csv", index=False)

    # Distribution summary on ticker ratios
    ticker_ratio = per_ticker_df["sil_sig_ratio"]
    finite_ratio = ticker_ratio[np.isfinite(ticker_ratio)]
    dist = {
        "num_files": len(files),
        "signal_key_used": signal_key_used,
        "signal_threshold": args.signal_threshold,
        "d_sig": int(global_sig),
        "d_sil": int(global_sil),
        "d_total": int(global_sig + global_sil),
        "global_sil_sig_ratio": safe_ratio(global_sil, global_sig),
        "num_tickers": int(len(per_ticker_df)),
        "num_tickers_sil_gt_sig": int(per_ticker_df["sil_gt_sig"].sum()),
        "pct_tickers_sil_gt_sig": float(per_ticker_df["sil_gt_sig"].mean()) if len(per_ticker_df) > 0 else np.nan,
        "num_tickers_sig_eq_0": int((per_ticker_df["d_sig"] == 0).sum()),
        "ratio_mean_finite": float(finite_ratio.mean()) if len(finite_ratio) > 0 else np.nan,
        "ratio_median_finite": float(finite_ratio.median()) if len(finite_ratio) > 0 else np.nan,
        "ratio_p25_finite": percentile(finite_ratio, 0.25),
        "ratio_p75_finite": percentile(finite_ratio, 0.75),
    }
    with (output_dir / "distribution_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(dist, fp, indent=2)

    # File-level audit
    file_df = pd.DataFrame(file_rows).sort_values(["kol", "split", "file"])
    file_df.to_csv(output_dir / "per_file_counts.csv", index=False)

    # Markdown summary for paper drafting
    md = []
    md.append("# Signal vs Silence Report")
    md.append("")
    md.append(f"- replay_root: `{replay_root}`")
    md.append(f"- files: `{len(files)}`")
    md.append(f"- signal_key: `{signal_key_used}`")
    md.append(f"- signal_threshold: `{args.signal_threshold}`")
    md.append("")
    md.append("## Global")
    md.append(f"- |D_sig| = {global_sig}")
    md.append(f"- |D_sil| = {global_sil}")
    md.append(f"- |D_sil| / |D_sig| = {safe_ratio(global_sil, global_sig):.6f}")
    md.append("")
    md.append("## Ticker Distribution")
    md.append(f"- #tickers = {dist['num_tickers']}")
    md.append(f"- pct(|D_sil^(i)| > |D_sig^(i)|) = {dist['pct_tickers_sil_gt_sig']:.6f}")
    md.append(f"- mean ratio (finite) = {dist['ratio_mean_finite']:.6f}")
    md.append(f"- median ratio (finite) = {dist['ratio_median_finite']:.6f}")
    md.append(f"- p25 / p75 (finite) = {dist['ratio_p25_finite']:.6f} / {dist['ratio_p75_finite']:.6f}")
    md.append("")
    md.append("## Output Files")
    md.append("- `overall_counts.csv`")
    md.append("- `per_kol_counts.csv`")
    md.append("- `per_kol_split_counts.csv`")
    md.append("- `per_ticker_counts.csv`")
    md.append("- `per_file_counts.csv`")
    md.append("- `distribution_summary.json`")
    with (output_dir / "report.md").open("w", encoding="utf-8") as fp:
        fp.write("\n".join(md) + "\n")

    print(f"Saved report to: {output_dir}")
    print(f"Global |D_sig|={global_sig}, |D_sil|={global_sil}, ratio={safe_ratio(global_sil, global_sig):.6f}")


if __name__ == "__main__":
    main()
