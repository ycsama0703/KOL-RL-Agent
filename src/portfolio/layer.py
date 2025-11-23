"""Portfolio layer to convert raw scores into dollar allocations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List

import numpy as np


@dataclass
class PortfolioConfig:
    capital: float = 10_000.0
    epsilon: float = 1e-6
    max_long: float = 0.2  # 单票最大多头权重（正权重上限，<=0 表示不设上限）
    max_short: float = 0.2  # 单票最大空头权重（绝对值上限，<=0 表示不设上限）
    hold_decay: float = 1.0  # 未被当日信号提到的旧仓位的衰减系数（1.0 表示完全保持）


class PortfolioLayer:
    """Normalizes raw scores into portfolio weights and allocations."""

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()
        # 允许通过环境变量覆盖单票上限，便于快速 sweep
        env_max = os.getenv("PORTFOLIO_MAX_WEIGHT")
        if env_max:
            try:
                val = float(env_max)
                if val > 0:
                    self.config.max_long = val
                    self.config.max_short = val
            except ValueError:
                pass

    def allocate(
        self,
        raw_scores: Dict[str, float],
        prev_weights: Dict[str, float] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """Allocate capital by combining前一日仓位 (prev_weights) 与当日信号 raw_scores."""

        # 允许正负持仓：prev_weights/raw_scores 都按原符号保留
        prev_weights = {
            ticker: float(weight)
            for ticker, weight in (prev_weights or {}).items()
            if weight is not None
        }
        subset_scores = {
            ticker: float(score)
            for ticker, score in raw_scores.items()
            if score is not None
        }
        result_weights: Dict[str, float] = {}

        if not prev_weights and not subset_scores:
            return {}

        # 未被当日信号提到的旧仓位：按 hold_decay 保留
        keep_weights = {
            ticker: weight * self.config.hold_decay
            for ticker, weight in prev_weights.items()
            if ticker not in subset_scores
        }

        # 汇总今日候选权重（可正可负）
        candidate = {**keep_weights, **subset_scores}
        total_abs = sum(abs(w) for w in candidate.values())
        if total_abs < self.config.epsilon:
            return {}

        # 先按绝对值归一化
        result_weights = {t: w / total_abs for t, w in candidate.items()}

        # 长/空上限截断后再按绝对值归一化
        if result_weights:
            for ticker in list(result_weights.keys()):
                w = result_weights[ticker]
                if w > 0 and self.config.max_long > 0:
                    w = min(w, self.config.max_long)
                if w < 0 and self.config.max_short > 0:
                    w = max(w, -self.config.max_short)
                result_weights[ticker] = w

            capped_abs = sum(abs(w) for w in result_weights.values())
            if capped_abs > self.config.epsilon:
                for ticker in list(result_weights.keys()):
                    result_weights[ticker] = result_weights[ticker] / capped_abs
            else:
                return {}

        allocations = {ticker: weight * self.config.capital for ticker, weight in result_weights.items()}
        return {
            ticker: {"weight": float(weight), "allocation": float(allocations[ticker])}
            for ticker, weight in result_weights.items()
        }
