from moonshot import Moonshot
import pandas as pd


class MoonshotMinimalTestStrategy(Moonshot):
    """
    Moonshot Minimal Viable Strategy
    Purpose:
    - Verify Moonshot integration
    - Verify price datasource wiring
    - Verify Pandas-based backtest pipeline
    """

    # === Required identifiers ===
    CODE = "moonshot-minimal-test"
    DB = "usstock-1day"   # ⚠️ 改成你平台里真实存在的 price DB

    def prices_to_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate deterministic signals:
        - Always long (1) for all symbols after the first available date
        """
        closes = prices.loc["Close"]

        # Signal = 1 everywhere (simple, deterministic)
        signals = pd.DataFrame(
            1,
            index=closes.index,
            columns=closes.columns
        )

        return signals

    def signals_to_target_weights(self, signals, prices):
        """
        Evenly distribute capital across all active signals each day
        """
        signal_count = signals.abs().sum(axis=1)

        weights = signals.div(signal_count, axis=0).fillna(0)

        return weights

    def target_weights_to_positions(self, weights, prices):
        """
        Enter positions one period AFTER signals (Moonshot convention)
        """
        positions = weights.shift()

        return positions

    def positions_to_gross_returns(self, positions, prices):
        """
        Compute close-to-close returns
        """
        closes = prices.loc["Close"]

        gross_returns = closes.pct_change() * positions.shift()

        return gross_returns