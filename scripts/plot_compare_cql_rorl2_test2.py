#!/usr/bin/env python
"""Plot daily equity comparison for CQL vs RORL2 vs test2 (ours) per KOL."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CQL/RORL2/test2 daily equity per KOL.")
    parser.add_argument("--test2-root", default="outputs/test_v2")
    parser.add_argument("--cql-root", default="bencmarks/CQL")
    parser.add_argument("--rorl2-root", default="bencmarks/RORL 2")
    parser.add_argument("--output-root", default="outputs/benchmark_compare")
    parser.add_argument("--label-test2", default="test2")
    parser.add_argument("--label-cql", default="CQL")
    parser.add_argument("--label-rorl2", default="RORL2")
    parser.add_argument("--no-rebase", action="store_true")
    parser.add_argument("--only-kol", action="append", default=[])
    return parser.parse_args()


def list_kols_from_runs(root: Path) -> set[str]:
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


def latest_run_dir(root: Path, kol: str) -> Path | None:
    pattern = re.compile(rf"^{re.escape(kol)}_\d{{8}}_\d{{6}}$")
    candidates = [p for p in root.iterdir() if p.is_dir() and pattern.match(p.name)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def read_test2_equity(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "daily" / "equity_daily.csv"
    if not path.exists():
        path = run_dir / "equity_daily.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "equity_trained" in df.columns:
        equity = df[["date", "equity_trained"]].rename(columns={"equity_trained": "equity"})
        return equity
    if "equity" in df.columns:
        return df[["date", "equity"]]
    return None


def read_benchmark_equity(perf_path: Path) -> pd.DataFrame | None:
    if not perf_path.exists():
        return None
    df = pd.read_csv(perf_path)
    if "Date" not in df.columns or "NAV" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["Date"])
    return df[["date", "NAV"]].rename(columns={"NAV": "equity"})


def merge_series(series: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for label, df in series.items():
        cur = df.rename(columns={"equity": label})
        merged = cur if merged is None else merged.merge(cur, on="date", how="inner")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("date")


def rebase(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "date":
            continue
        out[col] = out[col] / out[col].iloc[0]
    return out


def main() -> None:
    args = parse_args()
    test2_root = Path(args.test2_root)
    cql_root = Path(args.cql_root)
    rorl2_root = Path(args.rorl2_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    kols_test2 = list_kols_from_runs(test2_root)
    kols_cql = {p.name for p in cql_root.iterdir() if p.is_dir()} if cql_root.exists() else set()
    kols_rorl2 = {p.name for p in rorl2_root.iterdir() if p.is_dir()} if rorl2_root.exists() else set()
    kols = kols_test2 & kols_cql & kols_rorl2

    if args.only_kol:
        kols = {kol for kol in kols if kol in args.only_kol}

    if not kols:
        raise SystemExit("No KOLs with data in test2/CQL/RORL2.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting.") from exc

    for kol in sorted(kols):
        run_dir = latest_run_dir(test2_root, kol)
        if not run_dir:
            print(f"Skip {kol} (missing test2 run)")
            continue
        test2 = read_test2_equity(run_dir)
        cql = read_benchmark_equity(cql_root / kol / "performance.csv")
        rorl2 = read_benchmark_equity(rorl2_root / kol / "performance.csv")

        if test2 is None or cql is None or rorl2 is None:
            print(f"Skip {kol} (missing equity series)")
            continue

        series = {
            args.label_test2: test2,
            args.label_cql: cql,
            args.label_rorl2: rorl2,
        }
        merged = merge_series(series)
        if merged.empty:
            print(f"Skip {kol} (no overlapping dates)")
            continue
        if not args.no_rebase:
            merged = rebase(merged)

        out_dir = output_root / kol
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "compare_daily.csv"
        merged.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        for label in [args.label_test2, args.label_cql, args.label_rorl2]:
            ax.plot(merged["date"], merged[label], label=label, linewidth=1.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (Rebased)" if not args.no_rebase else "Equity")
        ax.set_title(f"{kol} Daily Equity Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()

        fig_path = out_dir / "compare_daily.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
