"""Evaluate generic RL benchmarks with the same artifact layout as evaluate_run.py."""

from __future__ import annotations

import argparse
import json
import math
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
from src.training.models import MLP


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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("actor_state_dict", checkpoint)
    actor = VanillaActorNetwork(state_dim).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def compute_metrics(daily_returns: np.ndarray) -> Dict[str, float]:
    cumulative_return = float(np.prod(1 + daily_returns) - 1)
    sharpe = 0.0
    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-8:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252))

    equity = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / (peak + 1e-8)
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    return {"cumulative_return": cumulative_return, "sharpe": sharpe, "max_drawdown": max_drawdown}


def compute_betrayal_metrics(
    baseline_action: torch.Tensor,
    policy_action: torch.Tensor,
    *,
    entry_threshold: float,
    action_threshold: float,
) -> dict:
    baseline = baseline_action.detach().float().view(-1)
    policy = policy_action.detach().float().view(-1)

    eps = 1e-8
    has_signal = baseline.abs() >= float(entry_threshold)
    no_signal = ~has_signal

    prod = baseline * policy
    reversed_mask = has_signal & (prod < 0.0)
    entry_violation = no_signal & (policy.abs() > float(action_threshold))

    delta = policy - baseline
    abs_delta = delta.abs()

    def safe_mean(x: torch.Tensor) -> float:
        if x.numel() == 0:
            return 0.0
        return float(x.mean().item())

    def safe_rate(mask: torch.Tensor, denom_mask: torch.Tensor) -> float:
        denom = float(denom_mask.sum().item())
        if denom <= 0:
            return 0.0
        return float(mask.sum().item()) / denom

    metrics = {
        "num_samples": int(baseline.numel()),
        "num_has_signal": int(has_signal.sum().item()),
        "num_no_signal": int(no_signal.sum().item()),
        "reversal_rate": safe_rate(reversed_mask, has_signal),
        "reversal_mean_abs_action": safe_mean(policy[reversed_mask].abs()),
        "reversal_mean_abs_delta": safe_mean(abs_delta[reversed_mask]),
        "entry_violation_rate": safe_rate(entry_violation, no_signal),
        "entry_violation_mean_abs_action": safe_mean(policy[entry_violation].abs()),
        "mean_abs_deviation": safe_mean(abs_delta),
        "mean_normalized_deviation": safe_mean(abs_delta[has_signal] / (baseline[has_signal].abs() + eps)),
    }

    same_sign = has_signal & (prod > 0.0)
    metrics["sign_agreement_rate"] = safe_rate(same_sign, has_signal)

    b = baseline[has_signal]
    p = policy[has_signal]
    if b.numel() >= 2 and float(b.std().item()) > 0 and float(p.std().item()) > 0:
        corr = torch.corrcoef(torch.stack([b, p]))[0, 1]
        metrics["baseline_policy_corr"] = float(corr.item())
    else:
        metrics["baseline_policy_corr"] = float("nan")
    return metrics


def _classify_action(prev_weight: float, new_weight: float, delta: float, threshold: float) -> str:
    abs_prev = abs(prev_weight)
    abs_new = abs(new_weight)
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
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    states = buffer["states"]
    rewards = buffer["rewards"].numpy()
    dates = buffer["meta"]["published_at"]
    tickers = buffer["meta"]["ticker"]

    preds = []
    with torch.no_grad():
        for start in range(0, states.size(0), 1024):
            batch = states[start : start + 1024].to(device)
            preds.append(actor(batch).squeeze(-1).cpu())
    policy_actions = torch.cat(preds).numpy()

    df = pd.DataFrame({"date": dates, "ticker": tickers, "reward": rewards, "raw_score": policy_actions})
    portfolio = PortfolioLayer()
    daily_returns: list[float] = []
    position_rows: list[dict] = []
    prev_weights: Dict[str, float] = {}

    for date, group in df.groupby("date"):
        raw_dict = {row["ticker"]: row["raw_score"] for _, row in group.iterrows()}
        allocation = portfolio.allocate(raw_dict, prev_weights=prev_weights)
        new_weights = {ticker: info["weight"] for ticker, info in allocation.items()}
        rewards_today = {row["ticker"]: float(row["reward"]) for _, row in group.iterrows()}

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
                    "action": _classify_action(prev_weight, weight, delta, action_threshold),
                }
            )
            day_return += weight * reward

        prev_weights = new_weights
        daily_returns.append(day_return)

    metrics = compute_metrics(np.array(daily_returns)) if daily_returns else {
        "cumulative_return": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
    }
    return metrics, pd.DataFrame(position_rows), policy_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generic RL benchmark.")
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
    parser.add_argument("--entry-threshold", type=float, default=5e-4)
    return parser.parse_args()


def normalize_to_naive_day(values) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def normalize_index_to_naive_day(index_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index_like, errors="coerce", utc=True)
    idx = pd.DatetimeIndex(idx).tz_localize(None).normalize()
    return idx


def daily_equity(positions: pd.DataFrame) -> pd.DataFrame:
    df = positions.copy()
    df["date"] = normalize_to_naive_day(df["date"])
    df["weighted_return"] = df["weight"] * df["reward"]
    daily = df.groupby("date", as_index=False)["weighted_return"].sum().sort_values("date")
    daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
    return daily


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


def map_weights_to_next_trading_day(weights: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    mapped = pd.DataFrame(index=trading_dates, columns=weights.columns, dtype=float)
    dates = normalize_index_to_naive_day(weights.index).to_list()
    for date, row in zip(dates, weights.itertuples(index=False, name=None)):
        if pd.isna(date):
            continue
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
    df["date"] = normalize_to_naive_day(df["date"])
    weights = df.pivot_table(index="date", columns="ticker", values="weight", aggfunc="last").sort_index()

    returns = prices.pct_change().fillna(0.0)
    returns.index = normalize_index_to_naive_day(returns.index)
    returns = returns.reindex(sorted(returns.columns), axis=1)

    weights = weights.reindex(columns=returns.columns, fill_value=0.0)
    mapped = map_weights_to_next_trading_day(weights, returns.index)
    mapped = mapped.ffill().fillna(0.0)

    daily_return = (mapped * returns).sum(axis=1)
    equity = (1.0 + daily_return).cumprod()
    return pd.DataFrame({"date": daily_return.index, "daily_return": daily_return.to_numpy(), "equity": equity.to_numpy()})


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
    metrics, positions_df, policy_np = run_policy(actor, buffer, device, action_threshold=args.action_threshold)

    baseline_tensor = buffer.get("baseline_actions")
    if baseline_tensor is None:
        baseline_tensor = buffer.get("baseline_action")
    if baseline_tensor is None:
        baseline_tensor = buffer["actions"]
    baseline_tensor = baseline_tensor.float()

    policy_tensor = torch.from_numpy(policy_np).view(-1, 1)
    action_df = pd.DataFrame(
        {
            "date": buffer["meta"]["published_at"],
            "ticker": buffer["meta"]["ticker"],
            "baseline_action": baseline_tensor.view(-1).cpu().numpy(),
            "policy_action": policy_tensor.view(-1).cpu().numpy(),
        }
    )
    betrayal_metrics = compute_betrayal_metrics(
        baseline_action=baseline_tensor,
        policy_action=policy_tensor,
        entry_threshold=args.entry_threshold,
        action_threshold=args.action_threshold,
    )

    if args.daily_price_update:
        tickers = sorted(positions_df["ticker"].dropna().unique().tolist())
        pos_dates = normalize_to_naive_day(positions_df["date"])
        start = pos_dates.min()
        end = pos_dates.max()
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
    metrics_out = {**metrics, "daily_metrics": daily_metrics, "betrayal_metrics": betrayal_metrics}

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
        positions_out = positions_df.merge(action_df, on=["date", "ticker"], how="left")
        positions_out.to_csv(positions_path, index=False)
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

        _, base_positions, _ = run_policy(ZeroActor().to(device), buffer, device, action_threshold=args.action_threshold)

        def equity_series(positions: pd.DataFrame) -> pd.DataFrame:
            df = positions.copy()
            df["date"] = pd.to_datetime(df["date"])
            df["weighted_return"] = df["weight"] * df["reward"]
            daily = df.groupby("date", as_index=False)["weighted_return"].sum()
            daily["equity"] = (1.0 + daily["weighted_return"]).cumprod()
            return daily

        base = equity_series(base_positions).rename(columns={"equity": "equity_baseline"})
        train = equity_series(positions_df).rename(columns={"equity": "equity_trained"})
        daily = pd.merge(base, train, on="date", how="inner")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]
        except ImportError as exc:
            raise SystemExit("matplotlib is required for plotting. Please install it with `pip install matplotlib`.") from exc

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

            _, base_positions, _ = run_policy(ZeroActor().to(device), buffer, device, action_threshold=args.action_threshold)

        if args.daily_price_update:
            tickers = sorted(pd.concat([positions_df["ticker"], base_positions["ticker"]]).dropna().unique().tolist())
            all_dates = normalize_to_naive_day(pd.concat([positions_df["date"], base_positions["date"]]))
            start = all_dates.min()
            end = all_dates.max()
            if tickers and pd.notna(start) and pd.notna(end):
                price_frame = fetch_close_prices(tickers, start=start, end=end)
                daily_base = daily_equity_price_update(base_positions, price_frame)
                daily_train = daily_equity_price_update(positions_df, price_frame)
            else:
                daily_base = pd.DataFrame(columns=["date", "daily_return", "equity"])
                daily_train = pd.DataFrame(columns=["date", "daily_return", "equity"])
        else:
            daily_base = daily_equity(base_positions)
            daily_train = daily_equity(positions_df)

        train_returns = daily_train["daily_return"].to_numpy() if args.daily_price_update else daily_train["weighted_return"].to_numpy()
        base_returns = daily_base["daily_return"].to_numpy() if args.daily_price_update else daily_base["weighted_return"].to_numpy()

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
            raise SystemExit("matplotlib is required for plotting. Please install it with `pip install matplotlib`.") from exc

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
