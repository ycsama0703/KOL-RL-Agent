#!/usr/bin/env python
"""Plot daily equity for beta sweep vs test_v2 per KOL."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare beta sweep vs test_v2 (daily).")
    parser.add_argument("--sweep-root", default="outputs/sweep_beta/test")
    parser.add_argument("--testv2-root", default="outputs/test_v2")
    parser.add_argument("--output-root", default="outputs/sweep_beta/compare_testv2")
    parser.add_argument("--expectile", default="0.7")
    parser.add_argument("--betas", default="5 8")
    parser.add_argument("--label-testv2", default="test_v2")
    parser.add_argument("--no-rebase", action="store_true")
    parser.add_argument("--only-kol", action="append", default=[])
    return parser.parse_args()


def latest_run_dir(root: Path, kol: str) -> Path | None:
    pattern = re.compile(rf"^{re.escape(kol)}_\d{{8}}_\d{{6}}$")
    candidates = [p for p in root.iterdir() if p.is_dir() and pattern.match(p.name)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def list_kols(root: Path) -> set[str]:
    pattern = re.compile(r"^(.*)_\d{8}_\d{6}$")
    kols: set[str] = set()
    if not root.exists():
        return kols
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            kols.add(match.group(1))
    return kols


def read_equity_daily(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "equity_trained" in df.columns:
        return df[["date", "equity_trained"]].rename(columns={"equity_trained": "equity"})
    if "equity" in df.columns:
        return df[["date", "equity"]]
    return None


def merge_series(series: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for label, df in series.items():
        cur = df.rename(columns={"equity": label})
        merged = cur if merged is None else merged.merge(cur, on="date", how="inner")
    return merged.sort_values("date") if merged is not None else pd.DataFrame()


def rebase(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "date":
            continue
        out[col] = out[col] / out[col].iloc[0]
    return out


def main() -> None:
    args = parse_args()
    sweep_root = Path(args.sweep_root)
    testv2_root = Path(args.testv2_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    betas = [b.strip() for b in args.betas.split() if b.strip()]
    if not betas:
        raise SystemExit("No betas provided.")

    kols = list_kols(testv2_root)
    if not kols and sweep_root.exists():
        sample_dir = sweep_root / f"exp{args.expectile}_beta{betas[0]}"
        kols = list_kols(sample_dir)

    if args.only_kol:
        kols = {kol for kol in kols if kol in args.only_kol}

    if not kols:
        raise SystemExit("No KOLs found in test_v2 or sweep results.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting.") from exc

    for kol in sorted(kols):
        series: Dict[str, pd.DataFrame] = {}

        for beta in betas:
            sweep_dir = sweep_root / f"exp{args.expectile}_beta{beta}"
            if not sweep_dir.exists():
                continue
            run = latest_run_dir(sweep_dir, kol)
            if not run:
                continue
            df = read_equity_daily(run / "daily" / "equity_daily.csv")
            if df is not None:
                series[f"beta{beta}"] = df

        run_v2 = latest_run_dir(testv2_root, kol)
        if run_v2:
            df_v2 = read_equity_daily(run_v2 / "daily" / "equity_daily.csv")
            if df_v2 is not None:
                series[args.label_testv2] = df_v2

        if len(series) < 2:
            print(f"Skip {kol} (missing series)")
            continue

        merged = merge_series(series)
        if merged.empty:
            print(f"Skip {kol} (no overlapping dates)")
            continue
        if not args.no_rebase:
            merged = rebase(merged)

        out_dir = output_root / kol
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "compare_beta_daily.csv"
        merged.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        for label in merged.columns:
            if label == "date":
                continue
            ax.plot(merged["date"], merged[label], label=label, linewidth=1.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (Rebased)" if not args.no_rebase else "Equity")
        ax.set_title(f"{kol} Beta Sweep vs {args.label_testv2}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()

        fig_path = out_dir / "compare_beta_daily.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
