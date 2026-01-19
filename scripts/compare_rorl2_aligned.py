"""Align outputs/test_v2 daily results to RORL2 dates and export comparison CSVs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align test_v2 results to RORL2 dates.")
    parser.add_argument(
        "--test-root",
        default="outputs/test_v2",
        help="Root directory containing <KOL>_<timestamp>/daily/equity_daily.csv",
    )
    parser.add_argument(
        "--rorl-root",
        default="bencmarks/RORL 2",
        help="Root directory containing <KOL>/performance.csv",
    )
    parser.add_argument(
        "--output-summary",
        default="bencmarks/RORL 2/compare_aligned_summary.csv",
        help="Path to write the summary CSV.",
    )
    return parser.parse_args()


def compute_metrics(daily_returns: np.ndarray) -> Dict[str, float]:
    cumulative_return = float(np.prod(1 + daily_returns) - 1)
    sharpe = 0.0
    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-8:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252))
    equity = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / (peak + 1e-8)
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    return {
        "cumulative_return": cumulative_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def parse_kol_and_stamp(name: str) -> Tuple[str | None, str | None]:
    parts = name.split("_")
    if len(parts) < 3:
        return None, None
    date_part = parts[-2]
    time_part = parts[-1]
    if len(date_part) == 8 and len(time_part) == 6 and date_part.isdigit() and time_part.isdigit():
        kol = "_".join(parts[:-2])
        stamp = date_part + time_part
        return kol, stamp
    return None, None


def latest_runs(test_root: Path) -> Dict[str, Path]:
    runs: Dict[str, Tuple[str, Path]] = {}
    for path in test_root.iterdir():
        if not path.is_dir():
            continue
        kol, stamp = parse_kol_and_stamp(path.name)
        if not kol or not stamp:
            continue
        if kol not in runs or stamp > runs[kol][0]:
            runs[kol] = (stamp, path)
    return {kol: run for kol, (_, run) in runs.items()}


def daily_returns_from_equity(equity: pd.Series) -> np.ndarray:
    returns = equity.pct_change().to_numpy()
    if len(returns) > 0:
        returns[0] = equity.iloc[0] - 1.0
    return np.nan_to_num(returns, nan=0.0)


def _map_returns_to_next_trading_day(
    dates: pd.Series,
    returns: np.ndarray,
    trading_dates: pd.Series,
) -> pd.Series:
    trading = pd.to_datetime(trading_dates).dt.normalize().sort_values().unique()
    if len(trading) == 0:
        return pd.Series([], dtype=float)

    dates = pd.to_datetime(dates).dt.normalize()
    returns = np.nan_to_num(returns, nan=0.0)

    agg: Dict[pd.Timestamp, float] = {}
    for date, ret in zip(dates, returns):
        idx = np.searchsorted(trading, date, side="left")
        if idx >= len(trading):
            continue
        mapped = pd.Timestamp(trading[idx])
        if mapped in agg:
            agg[mapped] = (1.0 + agg[mapped]) * (1.0 + float(ret)) - 1.0
        else:
            agg[mapped] = float(ret)

    mapped_series = pd.Series(agg)
    mapped_series.index.name = "date"
    return mapped_series


def build_aligned_frame(ours: pd.DataFrame, rorl: pd.DataFrame) -> pd.DataFrame:
    rorl = rorl.copy()
    rorl["date"] = pd.to_datetime(rorl["Date"]).dt.normalize()
    rorl = rorl.sort_values("date")

    ours = ours.copy()
    ours["date"] = pd.to_datetime(ours["date"]).dt.normalize()
    ours = ours.sort_values("date")
    ours_returns = daily_returns_from_equity(ours["equity_trained"])
    mapped = _map_returns_to_next_trading_day(ours["date"], ours_returns, rorl["date"])
    aligned = mapped.reindex(rorl["date"], fill_value=0.0).reset_index()
    aligned.columns = ["date", "daily_return_trained"]

    aligned["nav_trained"] = (1.0 + aligned["daily_return_trained"]).cumprod()
    aligned["cumulative_return_trained"] = aligned["nav_trained"] - 1.0

    aligned["daily_return_rorl"] = rorl["Daily_Return"].to_numpy()
    aligned["nav_rorl"] = (1.0 + aligned["daily_return_rorl"]).cumprod()
    aligned["cumulative_return_rorl"] = aligned["nav_rorl"] - 1.0
    return aligned


def main() -> None:
    args = parse_args()
    test_root = Path(args.test_root)
    rorl_root = Path(args.rorl_root)

    runs = latest_runs(test_root)
    summaries: List[Dict[str, object]] = []

    for rorl_dir in sorted(rorl_root.iterdir()):
        if not rorl_dir.is_dir():
            continue
        perf_path = rorl_dir / "performance.csv"
        if not perf_path.exists():
            continue
        kol = rorl_dir.name
        run = runs.get(kol)
        if not run:
            print(f"Skip {kol}: no test_v2 run found")
            continue

        equity_path = run / "daily" / "equity_daily.csv"
        if not equity_path.exists():
            print(f"Skip {kol}: missing {equity_path}")
            continue

        ours = pd.read_csv(equity_path)
        rorl = pd.read_csv(perf_path)
        aligned = build_aligned_frame(ours, rorl)

        out_path = rorl_dir / "compare_aligned.csv"
        aligned.to_csv(out_path, index=False)

        metrics_trained = compute_metrics(aligned["daily_return_trained"].to_numpy())
        metrics_rorl = compute_metrics(aligned["daily_return_rorl"].to_numpy())
        summaries.append(
            {
                "kol": kol,
                "trained_cumulative_return": metrics_trained["cumulative_return"],
                "trained_sharpe": metrics_trained["sharpe"],
                "trained_max_drawdown": metrics_trained["max_drawdown"],
                "rorl_cumulative_return": metrics_rorl["cumulative_return"],
                "rorl_sharpe": metrics_rorl["sharpe"],
                "rorl_max_drawdown": metrics_rorl["max_drawdown"],
                "aligned_days": len(aligned),
                "test_run": str(run),
            }
        )
        print(f"Saved aligned comparison -> {out_path}")

    if summaries:
        summary_path = Path(args.output_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
