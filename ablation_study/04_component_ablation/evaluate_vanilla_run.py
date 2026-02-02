"""Evaluate vanilla IQL policy on a replay buffer split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio.layer import PortfolioLayer
from src.training.data import load_buffer
from src.training.models import MLP, CriticNetwork, ValueNetwork


class VanillaActorNetwork(torch.nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.backbone = MLP(
            input_dim=state_dim,
            hidden_dims=(512, 512, 256),
            output_dim=256,
            output_activation=torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(torch.nn.Linear(256, 1), torch.nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(state))


def load_actor(checkpoint_path: Path, state_dim: int, device: torch.device) -> VanillaActorNetwork:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("actor_state_dict", ckpt)
    actor = VanillaActorNetwork(state_dim).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def compute_metrics(daily_returns: np.ndarray) -> Dict[str, float]:
    cumulative_return = float(np.prod(1 + daily_returns) - 1)
    sharpe = 0.0
    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-8:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
    equity = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / (peak + 1e-8)
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    return {"cumulative_return": cumulative_return, "sharpe": sharpe, "max_drawdown": max_drawdown}


def _classify_action(prev_weight: float, new_weight: float, threshold: float) -> str:
    abs_prev = abs(prev_weight)
    abs_new = abs(new_weight)
    delta = new_weight - prev_weight
    if abs_new < threshold and abs_prev < threshold:
        return "HOLD"
    if abs_new < threshold <= abs_prev:
        return "CLOSE"
    if abs_prev < threshold <= abs_new:
        return "OPEN"
    if delta > threshold:
        return "INCREASE"
    if delta < -threshold:
        return "DECREASE"
    return "HOLD"


def run_policy(
    actor: VanillaActorNetwork,
    buffer: Dict[str, Any],
    device: torch.device,
    action_threshold: float = 0.01,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    states = buffer["states"]
    rewards = buffer["rewards"].numpy()
    dates = buffer["meta"]["published_at"]
    tickers = buffer["meta"]["ticker"]

    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, states.size(0), 1024):
            batch = states[start : start + 1024].to(device)
            preds.append(actor(batch).squeeze(-1).cpu())
    raw_scores = torch.cat(preds).numpy()

    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": tickers,
            "reward": rewards,
            "raw_score": raw_scores,
        }
    )

    portfolio = PortfolioLayer()
    daily_returns: list[float] = []
    position_rows: list[dict] = []
    prev_weights: Dict[str, float] = {}

    for date, group in df.groupby("date"):
        raw_dict = {row["ticker"]: row["raw_score"] for _, row in group.iterrows()}
        allocation = portfolio.allocate(raw_dict, prev_weights=prev_weights)
        new_weights = {ticker: info["weight"] for ticker, info in allocation.items()}

        rewards_today: Dict[str, float] = {
            row["ticker"]: float(row["reward"]) for _, row in group.iterrows()
        }

        day_return = 0.0
        tickers_today = sorted(set(new_weights.keys()) | set(prev_weights.keys()))
        for ticker in tickers_today:
            prev_weight = float(prev_weights.get(ticker, 0.0))
            weight = float(new_weights.get(ticker, 0.0))
            delta = weight - prev_weight
            allocation_val = weight * portfolio.config.capital
            allocation_delta = delta * portfolio.config.capital
            reward = float(rewards_today.get(ticker, 0.0))
            raw_score = float(raw_dict.get(ticker, 0.0))

            day_return += weight * reward

            position_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "reward": reward,
                    "raw_score": raw_score,
                    "prev_weight": prev_weight,
                    "weight": weight,
                    "weight_delta": delta,
                    "allocation": allocation_val,
                    "allocation_delta": allocation_delta,
                    "action": _classify_action(prev_weight, weight, action_threshold),
                }
            )

        daily_returns.append(day_return)
        prev_weights = new_weights

    metrics = compute_metrics(np.array(daily_returns)) if daily_returns else {
        "cumulative_return": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
    }
    return metrics, pd.DataFrame(position_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vanilla IQL policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Optional path to dump metrics as JSON.")
    parser.add_argument("--positions-output", help="Optional CSV path to log positions.")
    parser.add_argument("--output-dir", help="Optional directory to write metrics_test.json and positions_test.csv.")
    parser.add_argument("--daily-output-dir", help="Optional directory to write daily metrics/plot.")
    parser.add_argument("--daily-price-update", action="store_true", help="Use daily price mark-to-market.")
    parser.add_argument("--plot", action="store_true", help="Generate equity curve plot.")
    parser.add_argument("--plot-output", help="Optional path for equity curve plot (PNG).")
    parser.add_argument("--action-threshold", type=float, default=0.01)
    return parser.parse_args()


def daily_equity(positions: pd.DataFrame, ref_dates: pd.Series | None = None) -> pd.DataFrame:
    if positions.empty or "date" not in positions.columns:
        if ref_dates is None or ref_dates.empty:
            return pd.DataFrame(columns=["date", "weighted_return", "equity"])
        base_dates = pd.to_datetime(ref_dates, errors="coerce").dropna().drop_duplicates().sort_values()
        daily = pd.DataFrame({"date": base_dates, "weighted_return": 0.0})
        daily["equity"] = 1.0
        return daily

    df = positions.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["weighted_return"] = df["weight"] * df["reward"]
    daily = df.groupby("date", as_index=False)["weighted_return"].sum().sort_values("date")
    daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
    return daily


def safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(dtype=object)


def sanitize_ticker(ticker: str) -> str:
    return ticker.strip().replace(".", "-").upper()


def fetch_close_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit("yfinance is required for daily price update. Please install it.") from exc

    mapping = {t: sanitize_ticker(t) for t in tickers}
    unique = sorted(set(mapping.values()))
    if not unique:
        return pd.DataFrame()

    data = yf.download(
        tickers=" ".join(unique),
        start=start.date(),
        end=(end + pd.Timedelta(days=1)).date(),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()

    frames: list[pd.Series] = []
    for original, yf_ticker in mapping.items():
        if isinstance(data.columns, pd.MultiIndex):
            if yf_ticker not in data.columns.get_level_values(0):
                continue
            close = data[yf_ticker]["Close"].copy()
        else:
            close = data["Close"].copy()
        close.name = original
        frames.append(close)
    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.sort_index()


def map_weights_to_next_trading_day(
    weights: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    mapped = pd.DataFrame(index=trading_dates, columns=weights.columns, dtype=float)
    dates = pd.to_datetime(weights.index).to_list()
    for date, row in zip(dates, weights.itertuples(index=False, name=None)):
        idx = trading_dates.searchsorted(date, side="left")
        if idx < len(trading_dates) and trading_dates[idx] == date:
            idx += 1
        if idx >= len(trading_dates):
            continue
        mapped.iloc[idx] = list(row)
    return mapped


def daily_equity_price_update(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if positions.empty or prices.empty:
        return pd.DataFrame(columns=["date", "daily_return", "equity"])

    df = positions.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    weights = (
        df.pivot_table(index="date", columns="ticker", values="weight", aggfunc="last")
        .sort_index()
    )

    returns = prices.pct_change().fillna(0.0)
    returns = returns.reindex(sorted(returns.columns), axis=1)
    weights = weights.reindex(columns=returns.columns, fill_value=0.0)
    mapped = map_weights_to_next_trading_day(weights, returns.index)
    mapped = mapped.ffill().fillna(0.0)

    daily_return = (mapped * returns).sum(axis=1)
    equity = (1.0 + daily_return).cumprod()
    return pd.DataFrame(
        {
            "date": daily_return.index,
            "daily_return": daily_return.to_numpy(),
            "equity": equity.to_numpy(),
        }
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    buffer_path = Path(args.buffer)
    device = torch.device(args.device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not buffer_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {buffer_path}")

    buffer = load_buffer(buffer_path)
    state_dim = buffer["states"].shape[1]
    actor = load_actor(checkpoint_path, state_dim, device)
    metrics, positions_df = run_policy(actor, buffer, device, action_threshold=args.action_threshold)

    if args.daily_price_update:
        tickers = sorted(safe_series(positions_df, "ticker").dropna().unique().tolist())
        dates = pd.to_datetime(safe_series(positions_df, "date"), errors="coerce")
        start = dates.min()
        end = dates.max()
        if tickers and pd.notna(start) and pd.notna(end):
            price_frame = fetch_close_prices(tickers, start=start, end=end)
            daily_train = daily_equity_price_update(positions_df, price_frame)
            daily_returns = daily_train["daily_return"].to_numpy()
        else:
            daily_train = pd.DataFrame(columns=["date", "daily_return", "equity"])
            daily_returns = daily_train["daily_return"].to_numpy()
    else:
        daily_train = daily_equity(positions_df)
        daily_returns = daily_train["weighted_return"].to_numpy()
    daily_metrics = compute_metrics(daily_returns) if len(daily_returns) else metrics
    metrics_out = {**metrics, "daily_metrics": daily_metrics}
    print(json.dumps(metrics_out, indent=2))

    output_path = Path(args.output) if args.output else None
    positions_path = Path(args.positions_output) if args.positions_output else None
    if args.output_dir:
        base = Path(args.output_dir)
        base.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            output_path = base / "metrics_test.json"
        if positions_path is None:
            positions_path = base / "positions_test.csv"

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics_out, fp, indent=2)
        print(f"Saved metrics to {output_path}")
    if positions_path:
        positions_path.parent.mkdir(parents=True, exist_ok=True)
        positions_df.to_csv(positions_path, index=False)
        print(f"Saved positions log to {positions_path}")

    base_positions = None
    if args.plot or args.plot_output or args.daily_output_dir:
        plot_path = Path(args.plot_output) if args.plot_output else None
        if plot_path is None:
            if not args.output_dir:
                raise SystemExit("Plot requested but no --output-dir or --plot-output provided.")
            plot_path = Path(args.output_dir) / "equity_test.png"

        class ZeroActor(torch.nn.Module):
            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return torch.zeros((state.size(0), 1), device=state.device)

        _, base_positions = run_policy(ZeroActor().to(device), buffer, device, action_threshold=args.action_threshold)

        def equity_series(positions: pd.DataFrame, ref_dates: pd.Series | None = None) -> pd.DataFrame:
            if positions.empty or "date" not in positions.columns:
                if ref_dates is None or ref_dates.empty:
                    return pd.DataFrame(columns=["date", "weighted_return", "equity"])
                base_dates = pd.to_datetime(ref_dates, errors="coerce").dropna().drop_duplicates().sort_values()
                daily = pd.DataFrame({"date": base_dates, "weighted_return": 0.0})
                daily["equity"] = 1.0
                return daily

            df = positions.copy()
            df["date"] = pd.to_datetime(df["date"])
            df["weighted_return"] = df["weight"] * df["reward"]
            daily = df.groupby("date", as_index=False)["weighted_return"].sum()
            daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
            return daily

        train = equity_series(positions_df).rename(columns={"equity": "equity_trained"})
        base = equity_series(base_positions, ref_dates=train["date"]).rename(columns={"equity": "equity_baseline"})
        daily = pd.merge(base, train, on="date", how="inner")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]
        except ImportError as exc:
            raise SystemExit("matplotlib is required for plotting.") from exc

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily["date"], daily["equity_baseline"], label="Baseline", linewidth=1.8)
        ax.plot(daily["date"], daily["equity_trained"], label="Trained", linewidth=1.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.set_title("Baseline vs Trained Equity (Test)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved equity curve figure to {plot_path}")

    if args.daily_output_dir:
        daily_dir = Path(args.daily_output_dir)
        daily_dir.mkdir(parents=True, exist_ok=True)

        if base_positions is None:
            class ZeroActor(torch.nn.Module):
                def forward(self, state: torch.Tensor) -> torch.Tensor:
                    return torch.zeros((state.size(0), 1), device=state.device)

            _, base_positions = run_policy(
                ZeroActor().to(device),
                buffer,
                device,
                action_threshold=args.action_threshold,
            )

        if args.daily_price_update:
            tickers = sorted(
                pd.concat(
                    [
                        safe_series(positions_df, "ticker"),
                        safe_series(base_positions, "ticker"),
                    ],
                    ignore_index=True,
                )
                .dropna()
                .unique()
                .tolist()
            )
            dates = pd.to_datetime(
                pd.concat(
                    [
                        safe_series(positions_df, "date"),
                        safe_series(base_positions, "date"),
                    ],
                    ignore_index=True,
                ),
                errors="coerce",
            )
            start = dates.min()
            end = dates.max()
            if tickers and pd.notna(start) and pd.notna(end):
                price_frame = fetch_close_prices(tickers, start=start, end=end)
                daily_train = daily_equity_price_update(positions_df, price_frame)
                daily_base = daily_equity_price_update(base_positions, price_frame)
            else:
                daily_train = pd.DataFrame(columns=["date", "daily_return", "equity"])
                daily_base = pd.DataFrame(columns=["date", "daily_return", "equity"])
        else:
            daily_train = daily_equity(positions_df)
            daily_base = daily_equity(base_positions, ref_dates=daily_train["date"])
        if args.daily_price_update:
            train_returns = daily_train["daily_return"].to_numpy()
            base_returns = daily_base["daily_return"].to_numpy()
        else:
            train_returns = daily_train["weighted_return"].to_numpy()
            base_returns = daily_base["weighted_return"].to_numpy()
        metrics_daily = {
            "trained": compute_metrics(train_returns) if len(train_returns) else metrics,
            "baseline": compute_metrics(base_returns) if len(base_returns) else metrics,
        }
        metrics_path = daily_dir / "metrics_daily.json"
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics_daily, fp, indent=2)
        print(f"Saved daily metrics to {metrics_path}")

        daily_merge = pd.merge(
            daily_base.rename(columns={"equity": "equity_baseline"}),
            daily_train.rename(columns={"equity": "equity_trained"}),
            on="date",
            how="inner",
        )
        daily_csv = daily_dir / "equity_daily.csv"
        daily_merge.to_csv(daily_csv, index=False)
        print(f"Saved daily equity CSV to {daily_csv}")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]
        except ImportError as exc:
            raise SystemExit("matplotlib is required for plotting.") from exc

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_merge["date"], daily_merge["equity_baseline"], label="Baseline", linewidth=1.8)
        ax.plot(daily_merge["date"], daily_merge["equity_trained"], label="Trained", linewidth=1.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.set_title("Baseline vs Trained Equity (Daily)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()

        fig_path = daily_dir / "equity_daily.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved daily equity plot to {fig_path}")


if __name__ == "__main__":
    main()
