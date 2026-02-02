#!/usr/bin/env python
"""Batch plot per-KOL equity comparison with optional baseline and SPY benchmark."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.analyzer import run_policy
from src.training.data import load_buffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-KOL comparison curves with SPY.")
    parser.add_argument("--output-root", default="outputs/ablation/compare_kols")
    parser.add_argument("--full-modified-dir", default="outputs/ablation/test/full_modified")
    parser.add_argument("--full-modified-label", default="full_modified")
    parser.add_argument("--highlight-label", default="")
    parser.add_argument("--bc-only-dir", default="outputs/ablation/test/bc_only")
    parser.add_argument("--iql-modified-dir", default="outputs/ablation/test/iql_modified_only")
    parser.add_argument("--bc-plus-dir", default="outputs/ablation/test/bc_plus_vanilla_iql")
    parser.add_argument("--iql-vanilla-dir", default="outputs/ablation/test/iql_vanilla_only")
    parser.add_argument("--train-v2-dir", default="outputs/test_v2")
    parser.add_argument("--buffer-root", default="")
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--benchmark-label", default="SPY")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--use-daily", action="store_true", help="Use daily equity outputs instead of event-time.")
    parser.add_argument("--action-threshold", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only-kol", action="append", default=[])
    return parser.parse_args()


def default_buffer_root() -> str:
    if Path("data/buffer_22-24_end1231").exists():
        return "data/buffer_22-24_end1231"
    return "data/buffer_22-24"


def latest_run(base_dir: Path, kol: str) -> Path | None:
    if not base_dir.exists():
        return None
    pattern = re.compile(rf"^{re.escape(kol)}_\d{{8}}_\d{{6}}$")
    runs = [p for p in base_dir.iterdir() if p.is_dir() and pattern.match(p.name)]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.name)[-1]


def find_positions(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "test/event/positions_test.csv",
        run_dir / "event/positions_test.csv",
        run_dir / "positions_test.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_daily_equity(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "daily/equity_daily.csv",
        run_dir / "test/daily/equity_daily.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def equity_from_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["weighted_return"] = df["weight"] * df["reward"]
    daily = df.groupby("date", as_index=False)["weighted_return"].sum().sort_values("date")
    daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
    return daily[["date", "equity"]]


def equity_from_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "equity_trained" in df.columns:
        equity = df["equity_trained"]
    elif "equity" in df.columns:
        equity = df["equity"]
    else:
        raise ValueError(f"No equity column found in {path}")
    return pd.DataFrame({"date": df["date"], "equity": equity})


def baseline_from_daily(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    if "equity_baseline" not in df.columns:
        return None
    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "equity": df["equity_baseline"],
        }
    )


def add_series(merged: pd.DataFrame | None, series: pd.DataFrame, name: str) -> pd.DataFrame:
    cur = series.rename(columns={"equity": name})
    if merged is None:
        return cur
    return merged.merge(cur, on="date", how="inner")


def build_series_map(
    kol: str,
    dirs: Dict[str, Path],
    use_daily: bool,
) -> Dict[str, Path]:
    series = {}
    for label, base in dirs.items():
        run = latest_run(base, kol)
        if not run:
            continue
        if use_daily:
            daily = find_daily_equity(run)
            if daily:
                series[label] = daily
        else:
            positions = find_positions(run)
            if positions:
                series[label] = positions
    return series


def fetch_benchmark(dates: Iterable[pd.Timestamp], ticker: str) -> pd.Series:
    try:
        import yfinance as yf  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit("yfinance is required for benchmark plotting.") from exc

    date_index = pd.to_datetime(pd.Series(list(dates))).dropna().sort_values()
    if date_index.empty:
        return pd.Series(dtype=float)

    start = date_index.iloc[0]
    end = date_index.iloc[-1]
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)  # type: ignore[arg-type]
    if data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float)
    bench_close = data["Close"]
    if isinstance(bench_close, pd.DataFrame):
        bench_close = bench_close.iloc[:, 0]
    bench_ret = bench_close.pct_change().fillna(0.0)
    bench_eq = (1.0 + bench_ret).cumprod()
    bench_eq = bench_eq.reindex(date_index, method="ffill").bfill()
    if isinstance(bench_eq, pd.DataFrame):
        bench_eq = bench_eq.iloc[:, 0]
    return bench_eq.squeeze()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    buffer_root = args.buffer_root or default_buffer_root()
    full_dir = Path(args.full_modified_dir)
    if not full_dir.exists():
        raise SystemExit(f"full_modified dir not found: {full_dir}")

    kol_names = []
    pattern = re.compile(r"^(.*)_\d{8}_\d{6}$")
    for run in full_dir.iterdir():
        if not run.is_dir():
            continue
        match = pattern.match(run.name)
        if match:
            kol_names.append(match.group(1))

    if args.only_kol:
        kol_names = [kol for kol in kol_names if kol in args.only_kol]

    if not kol_names:
        raise SystemExit("No KOL runs found in full_modified dir.")

    anchor_label = args.full_modified_label
    highlight_label = args.highlight_label or anchor_label
    series_dirs = {
        anchor_label: Path(args.full_modified_dir),
        "bc_only": Path(args.bc_only_dir),
        "iql_modified_only": Path(args.iql_modified_dir),
        "bc_plus_vanilla_iql": Path(args.bc_plus_dir),
        "iql_vanilla_only": Path(args.iql_vanilla_dir),
        "train_v2": Path(args.train_v2_dir),
    }

    for kol in sorted(set(kol_names)):
        series_map = build_series_map(kol, series_dirs, args.use_daily)
        if anchor_label not in series_map:
            kind = "daily equity" if args.use_daily else "positions_test.csv"
            print(f"Skip {kol} (missing {anchor_label} {kind})")
            continue
        if not series_map:
            kind = "daily equity" if args.use_daily else "positions_test.csv"
            print(f"Skip {kol} (no {kind} found)")
            continue

        merged = None
        for label, path in series_map.items():
            if args.use_daily:
                series = equity_from_daily(path)
            else:
                series = equity_from_positions(path)
            if series.empty:
                print(f"Skip {kol} {label} (empty series: {path})")
                continue
            merged = add_series(merged, series, label)

        if merged is None or merged.empty:
            print(f"Skip {kol} (no overlapping dates)")
            continue

        if not args.no_baseline:
            base_series = None
            if args.use_daily:
                anchor_daily = series_map.get(anchor_label)
                if anchor_daily:
                    base_series = baseline_from_daily(anchor_daily)
            if base_series is None:
                buffer_path = Path(buffer_root) / kol / "test.pt"
                if buffer_path.exists():
                    class ZeroActor(torch.nn.Module):
                        def forward(self, state: torch.Tensor) -> torch.Tensor:
                            return torch.zeros((state.size(0), 1), device=state.device)

                    device = torch.device(args.device)
                    buffer = load_buffer(buffer_path)
                    _, base_positions = run_policy(
                        ZeroActor().to(device),
                        buffer,
                        device,
                        action_threshold=args.action_threshold,
                    )
                    base_series = equity_from_positions_df(base_positions)
                else:
                    print(f"Skip baseline for {kol} (missing buffer: {buffer_path})")
            if base_series is not None and not base_series.empty:
                merged = add_series(merged, base_series, "baseline")

        if not args.no_benchmark:
            bench_eq = fetch_benchmark(merged["date"], args.benchmark_ticker)
            if not bench_eq.empty:
                bench = pd.DataFrame({"date": merged["date"], args.benchmark_label: bench_eq.values})
                merged = merged.merge(bench, on="date", how="inner")

        # Rebase all series to 1.
        for col in merged.columns:
            if col != "date":
                merged[col] = merged[col] / merged[col].iloc[0]

        out_dir = output_root / kol
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / ("compare_equity_daily.csv" if args.use_daily else "compare_equity.csv")
        merged.to_csv(csv_path, index=False)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]
        except ImportError as exc:
            raise SystemExit("matplotlib is required for plotting.") from exc

        fig, ax = plt.subplots(figsize=(10, 5))
        for col in merged.columns:
            if col == "date":
                continue
            linestyle = "-"
            if col == "baseline":
                linestyle = "-."
            elif not args.no_benchmark and col == args.benchmark_label:
                linestyle = "--"
            linewidth = 1.8
            zorder = 2
            if col == highlight_label:
                linewidth = 2.6
                zorder = 3
            ax.plot(
                merged["date"],
                merged[col],
                label=col,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=zorder,
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (Rebased)")
        title = "Daily Equity Comparison" if args.use_daily else "Equity Comparison (Event-Time)"
        ax.set_title(f"{kol} {title}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig_path = out_dir / ("compare_equity_daily.png" if args.use_daily else "compare_equity.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {fig_path}")


def equity_from_positions_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weighted_return"] = df["weight"] * df["reward"]
    daily = df.groupby("date", as_index=False)["weighted_return"].sum().sort_values("date")
    daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
    return daily[["date", "equity"]]


if __name__ == "__main__":
    main()
