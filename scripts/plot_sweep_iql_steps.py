#!/usr/bin/env python
"""Plot per-KOL equity curves for all IQL-step sweep results."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sweep results per KOL.")
    parser.add_argument("--test-root", default="outputs/sweep_iql_steps/test")
    parser.add_argument("--output-root", default="outputs/sweep_iql_steps/plots")
    parser.add_argument("--use-daily", action="store_true", help="Use daily/equity_daily.csv if present.")
    parser.add_argument("--no-rebase", action="store_true")
    parser.add_argument("--only-kol", action="append", default=[])
    return parser.parse_args()


def equity_from_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["weighted_return"] = df["weight"] * df["reward"]
    daily = df.groupby("date", as_index=False)["weighted_return"].sum().sort_values("date")
    daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
    return daily[["date", "equity"]]


def find_positions(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "event" / "positions_test.csv",
        run_dir / "positions_test.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def equity_from_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "equity_trained" in df.columns:
        out = df[["date", "equity_trained"]].rename(columns={"equity_trained": "equity"})
        return out
    if "equity" in df.columns:
        return df[["date", "equity"]]
    raise ValueError(f"Unknown daily equity columns in {path}")


def find_daily(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "daily" / "equity_daily.csv",
        run_dir / "equity_daily.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    args = parse_args()
    test_root = Path(args.test_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not test_root.exists():
        raise SystemExit(f"Test root not found: {test_root}")

    pattern = re.compile(r"^(.*)_\d{8}_\d{6}$")
    series: Dict[str, Dict[int, pd.DataFrame]] = {}

    for step_dir in sorted(test_root.glob("steps_*")):
        step_str = step_dir.name.replace("steps_", "")
        if not step_str.isdigit():
            continue
        step = int(step_str)
        for run in step_dir.iterdir():
            if not run.is_dir():
                continue
            match = pattern.match(run.name)
            if not match:
                continue
            kol = match.group(1)
            if args.only_kol and kol not in args.only_kol:
                continue
            if args.use_daily:
                daily = find_daily(run)
                if daily:
                    series.setdefault(kol, {})[step] = equity_from_daily(daily)
                    continue
            positions = find_positions(run)
            if positions:
                series.setdefault(kol, {})[step] = equity_from_positions(positions)

    if not series:
        raise SystemExit("No sweep positions_test.csv found.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting.") from exc

    for kol, step_map in sorted(series.items()):
        merged = None
        for step, df in sorted(step_map.items()):
            cur = df.rename(columns={"equity": f"step_{step}"})
            merged = cur if merged is None else merged.merge(cur, on="date", how="inner")

        if merged is None or merged.empty:
            print(f"Skip {kol} (no overlapping dates)")
            continue

        if not args.no_rebase:
            for col in merged.columns:
                if col == "date":
                    continue
                merged[col] = merged[col] / merged[col].iloc[0]

        out_dir = output_root / kol
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "equity_steps.csv"
        merged.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        for col in merged.columns:
            if col == "date":
                continue
            label = col.replace("step_", "iql_steps=")
            ax.plot(merged["date"], merged[col], label=label, linewidth=1.8)

        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (Rebased)" if not args.no_rebase else "Equity")
        ax.set_title(f"{kol} Sweep: IQL Steps")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()

        fig_path = out_dir / "equity_steps.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
