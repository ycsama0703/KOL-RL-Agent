# KOL-RL-Agent 方法说明（策略 & 评估）

面向技术报告的概要，聚焦算法与评估逻辑（少谈工程细节）。

## 目标与假设
- 目标：从 KOL 文本/情感信号学习可执行的多空组合策略，重点解决“无新信号时的持有/退出”。
- 账户假设：无现金头寸，日级满仓；单票绝对权重上限（默认 20%）；无交易成本/滑点。
- 回报计算：每日收益 = 持仓权重 × 单票收益求和；净值 = `cumprod(1 + daily_returns)`，累积收益 = 终值 - 1。
- 频率假设：按自然日（或交易日）顺序回放，调仓在日初完成，使用当日的 reward 作为日收益。

## 数据到 Replay Buffer
- 清洗后文本 + ModernBERT embedding + 行情窗口 + reward → enriched/reward CSV。
- 基线策略：文本情感分数经 `tanh` 归一化，生成有符号的 baseline raw score；组合层保持连续持仓（`hold_decay`）。
- Replay buffer 关键字段：`states`（含 silence_days、last_position 等）、`actions`（baseline 权重）、`rewards`（逐票收益）、`meta`（date/ticker/文本）。
- 状态特征示例：文本 embedding；行情窗口特征（近 N 日收益/波动等）；`silence_days`（距上次提及天数）；`last_position`（上一日权重）；可选基线打分。
- Reward 构建：逐票日收益（如收盘价对数收益）；可选平滑/截断；当前未扣交易成本。

## 策略与组合
- 组合层（`PortfolioLayer`）：合并昨日仓位与当日 raw score，按绝对值归一化，截断单票上限，再归一化；无信号时旧仓位按 `hold_decay` 保留。
- 残差策略（训练文件 `experiments/residual_sweep/train_residual.py`）：
  - 有信号分支：`policy_sig = baseline * (1 + residual_scale * tanh(delta_sig))`（同向缩放，不翻转）。
  - 无信号分支：`policy_nosig = last_position * sigmoid(decay_scale * delta_decay)`（沉默期的持有/衰减/退出由 RL 学习）。
  - 合成：`policy = has_signal * policy_sig + (1 - has_signal) * policy_nosig`，`has_signal` 来自基线是否非零。
  - 状态特征：`silence_days`、`last_position` 支撑退出决策。
- 标准版策略（`train.py`）：直接预测 residual 加到 baseline 上，无显式无信号衰减分支。
- 动作解释：输出的是目标权重（-1~1，受单票上限约束）；无现金位，权重会归一化；不涉及订单类型/成交价，假设能按权重即时成交。

## 训练设置
- 先 BC（SFT 角色）预热少量 epoch，再 IQL（Expectile + 温度参数）主训练。
- 关键超参：`fidelity_lambda`（对齐基线强度）、`residual_scale`（同向缩放幅度）、`decay_scale`（沉默期衰减斜率）、`hold_decay`（组合层保留系数）、`max_weight`（单票上限）。
- 设备自适应 CUDA/CPU；batch 默认 256，IQL steps 200k（可调）。
- 优化细节：Adam 学习率 3e-4（actor/critic/value 相同，可分开调）；梯度裁剪可选（未默认开启）；BC 只跑 0~1 epoch，避免过拟合。
- IQL 配置：expectile 默认 0.7；`temperature_beta` 默认 3.0；目标是 Advantage 加权学习 + 轻量 fidelity。

## 评估与产出
- 指标：`metrics_test.json`（cumulative_return / sharpe / max_drawdown）。
- 持仓轨迹：`positions_test*.csv`，字段含 `prev_weight/weight/weight_delta/allocation/raw_score/reward/action`（OPEN/CLOSE/INCREASE/DECREASE/HOLD），用于观察退出/减仓路径。
- 决策明细：`signal_decisions_test.csv`（逐视频），含 `equity_baseline/equity_trained` 可绘净值曲线。
- 可视化：`plot_equity_curve.py` 基于 `signal_decisions_test.csv` 生成净值图，可带基准（如 SPY）。
- 指标计算：Sharpe = `mean(daily_returns)/std(daily_returns) * sqrt(252)`；最大回撤基于净值序列；若无收益序列则指标为 0。
- Fidelity 评估（可选）：`scripts/evaluate_fidelity.py` 比较模型动作与基线动作一致率（`baseline_actions` vs `trained_actions`），用于衡量对齐程度。

## 如何解读“持有/退出”
- 查看 `positions_test*.csv` 中无新信号日的 `action` 列和 `weight` 变化：`DECREASE/CLOSE` 表示学习到的衰减/退出；`HOLD` 表示延续旧仓。
- 对比基线与残差：基线无衰减分支，残差版在沉默期可主动降仓，是本方案学习持有/退出的核心体现。
- 退出力度来源：`decay_scale` 调节 sigmoid 斜率；`hold_decay` 调节旧仓默认保留比；`fidelity_lambda` 越低，模型越敢偏离基线、加大退出。
- 观察方法：按 ticker 绘制随时间的 `weight` 变化，或统计每日 `action` 分布，辨别“沉默期”是否出现系统性减仓。
