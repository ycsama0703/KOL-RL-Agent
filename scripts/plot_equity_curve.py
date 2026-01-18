"""Plot equity curves for baseline vs trained policy over the test period.

依赖输入：由 `scripts/export_signal_decisions.py` 生成的 CSV（每行一个视频）。

示例：

  python scripts/export_signal_decisions.py \
    --checkpoint outputs/Everything_Money_20251119_171039/checkpoints/policy.pt \
    --reward-csv data/processed/reward/Everything_Money/test.csv \
    --vocab-path models/embedding/ticker_vocab.json \
    --embedding-path models/embedding/ticker_embedding.pt \
    --output outputs/Everything_Money_20251119_171039/signal_decisions_test.csv

  python scripts/plot_equity_curve.py \
    --signal-decisions outputs/Everything_Money_20251119_171039/signal_decisions_test.csv \
    --output-figure outputs/Everything_Money_20251119_171039/equity_test.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.analyzer import load_actor, run_policy
from src.training.data import load_buffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline vs trained equity curves over time.")
    parser.add_argument(
        "--signal-decisions",
        help="CSV produced by scripts/export_signal_decisions.py.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional policy checkpoint; if provided with --buffer, plot from replay buffer.",
    )
    parser.add_argument(
        "--buffer",
        help="Optional replay buffer; if provided with --checkpoint, plot from replay buffer.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for buffer-based plotting.",
    )
    parser.add_argument(
        "--output-figure",
        required=True,
        help="Path to save the equity curve figure (e.g., PNG).",
    )
    parser.add_argument(
        "--benchmark-ticker",
        help="Optional benchmark symbol (e.g., SPY or ^GSPC) to plot market equity.",
    )
    parser.add_argument(
        "--benchmark-label",
        help="Label for the benchmark curve (default: same as ticker).",
    )
    parser.add_argument(
        "--title",
        default="Baseline vs Trained Equity (Test Period)",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def rebase_to_one(series: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        if series.empty:
            return series
        first = series.iloc[0]
        if pd.isna(first) or first == 0:
            return series
        return series / first

    if args.checkpoint and args.buffer:
        buffer = load_buffer(args.buffer)
        state_dim = buffer["states"].shape[1]
        device = torch.device(args.device)

        actor = load_actor(Path(args.checkpoint), state_dim, device)

        class ZeroActor(torch.nn.Module):
            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return torch.zeros((state.size(0), 1), device=state.device)

        def equity_series(positions: pd.DataFrame) -> pd.DataFrame:
            df = positions.copy()
            df["date"] = pd.to_datetime(df["date"])
            df["weighted_return"] = df["weight"] * df["reward"]
            daily = df.groupby("date", as_index=False)["weighted_return"].sum()
            daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
            return daily

        _, pos_base = run_policy(ZeroActor().to(device), buffer, device)
        _, pos_train = run_policy(actor, buffer, device)

        base = equity_series(pos_base).rename(columns={"equity": "equity_baseline"})
        train = equity_series(pos_train).rename(columns={"equity": "equity_trained"})
        daily = pd.merge(base, train, on="date", how="inner").sort_values("date")
    else:
        if not args.signal_decisions:
            raise ValueError("Provide either --signal-decisions or (--checkpoint and --buffer).")
        csv_path = Path(args.signal_decisions)
        if not csv_path.exists():
            raise FileNotFoundError(f"Signal decisions CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "date" not in df.columns or "equity_baseline" not in df.columns or "equity_trained" not in df.columns:
            raise ValueError(
                "CSV must contain 'date', 'equity_baseline', and 'equity_trained' columns. "
                "Make sure it comes from export_signal_decisions.py."
            )

        # 每个日期可能有多条视频记录，这里按日期聚合（取该日最后一条记录的净值即可）。
        df["date"] = pd.to_datetime(df["date"])
        df_sorted = df.sort_values(["date"])
        daily = df_sorted.groupby("date", as_index=False).last()

    if not daily.empty:
        daily["equity_baseline"] = rebase_to_one(daily["equity_baseline"])
        daily["equity_trained"] = rebase_to_one(daily["equity_trained"])

    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "matplotlib is required for plotting. "
            "Please install it with `pip install matplotlib`."
        ) from exc

    benchmark_series = None
    benchmark_label = None
    if args.benchmark_ticker:
        try:
            import yfinance as yf  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise SystemExit(
                "yfinance is required for benchmark plotting. "
                "Please install it with `pip install yfinance`."
            ) from exc

        start = daily["date"].min().date()
        end = daily["date"].max().date()
        data = yf.download(args.benchmark_ticker, start=start, end=end, auto_adjust=False)  # type: ignore[arg-type]
        if not data.empty and "Close" in data.columns:
            bench_ret = data["Close"].pct_change().fillna(0.0)
            bench_eq = (1.0 + bench_ret).cumprod()
            # 对齐到 daily 的日期索引
            bench_eq = bench_eq.reindex(pd.to_datetime(daily["date"]), method="ffill")
            bench_eq = bench_eq.bfill()
            bench_eq = rebase_to_one(bench_eq)
            benchmark_series = bench_eq.values
            benchmark_label = args.benchmark_label or args.benchmark_ticker

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily["date"], daily["equity_baseline"], label="Baseline", linewidth=1.8)
    ax.plot(daily["date"], daily["equity_trained"], label="Trained", linewidth=1.8)
    if benchmark_series is not None:
        ax.plot(daily["date"], benchmark_series, label=benchmark_label, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    output_path = Path(args.output_figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved equity curve figure to {output_path}")


if __name__ == "__main__":
    main()
