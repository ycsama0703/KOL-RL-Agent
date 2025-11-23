# KOL-RL-Agent 综合技术说明（方法+实现现状）

## 任务与目标
- 任务：将 KOL 文本信号转成可执行的多资产多空策略，保持对文本的忠诚度，同时争取更优的收益/风险表现。
- 思路：先构建基线（完全由文本情感驱动），再用强化学习学一个“残差”微调，控制偏离幅度。

## 数据与基线
- 预处理：`scripts/generate_reward.py` 生成 `data/processed/reward/<KOL>/{train,val,test}.csv`，包含 sentiment、confidence、reward_1d、baseline_raw_score 等。
- 基线打分：`baseline_raw_score = tanh(2 * sentiment * confidence)`，乘情感符号得到多/空原始分数。
- 组合规则（PortfolioLayer）：多空、连续持仓，未提到的票可继承昨仓（可设 hold_decay），按绝对值归一化，单票上限默认 20%（可通过 env `PORTFOLIO_MAX_WEIGHT` 调整）。
- 当日基线持仓：将签名打分送入组合层得到基线权重，作为行为策略。

## 残差策略（当前模型）
- 动作形式：`action = baseline_weight * (1 + residual_scale * tanh(delta))`，delta 由 Actor 输出；同向缩放，不翻转基线方向。
- 状态：文本嵌入 + ticker 嵌入 + sentiment/confidence + last_position。
- 奖励：单票 `reward_1d`（多空收益），组合收益由持仓回放累积。
- 关键超参：`fidelity_lambda`（贴合基线的惩罚强度）、`residual_scale`（残差幅度）、`hold_decay`（旧仓衰减）、单票上限。

## 训练流程（experiments/residual_sweep/train_residual.py）
- BC（轻量，默认 1 epoch）：残差初始化为 0，使动作接近基线。
- IQL（带形状奖励）：
  - 策略动作：`policy = baseline * (1 + scale * delta)`
  - 形状奖励：`reward_aug = reward_1d - fidelity_lambda * (policy - baseline)^2`
  - Critic 目标：`target_q = reward_aug + gamma * (1 - done) * V(next)`（已修复维度对齐 [B,1]）
  - Critic 用行为动作（基线）训练，Value 用 expectile_loss，Actor 用优势加权的偏离惩罚 `(policy-baseline)^2`
- 输出：`outputs/<run_id>/Everything_Money_<timestamp>/checkpoints/{policy,actor,critic,value}.pt` + `run_summary.json`。

## Replay Buffer（实验版残差流水线）
- 脚本：`experiments/residual_sweep/build_replay_buffer_residual.py`
- 参数：`--max-weight`、`--hold-decay` 控制组合层；输出 `data/replay_buffer_residual*/<KOL>/{train,val,test}.pt`
- 字段：states、actions（基线签名权重）、rewards、portfolio_rewards、next_states、dones、meta（ticker/video/date/baseline_raw_score 等）
- `last_position`/`baseline_weight` 在构建时按给定上限/衰减回放得到。

## 评测与导出（实验版）
- 评测：`experiments/residual_sweep/evaluate_residual.py`，输入 checkpoint + buffer（需与训练一致的 scale/上限/衰减），输出 `metrics_test.json`、`positions_test.csv`。
- 按视频导出决策：`experiments/residual_sweep/export_decisions_residual.py`，重算 last_position/baseline_weight 后，导出 `signal_decisions_test.csv`（基线/训练后动作、持仓前后、净值轨迹）。
- 可视化：读取 `signal_decisions_test.csv` 的 `equity_baseline/equity_trained` 画净值曲线；可叠加不同超参/单票上限。

## 快速命令示例（单组，w=0.2，fid=0.3，scale=0.2，decay=1.0）
```bash
# 构建 buffer
python experiments/residual_sweep/build_replay_buffer_residual.py \
  --reward-dir data/processed/reward \
  --output-dir data/replay_buffer_residual_w0.2 \
  --max-weight 0.2 --hold-decay 1.0

# 训练
python experiments/residual_sweep/train_residual.py \
  --kol Everything_Money \
  --replay-dir data/replay_buffer_residual_w0.2 \
  --output-dir outputs/Everything_Money_residual_w0.2 \
  --fidelity-lambda 0.3 --residual-scale 0.2 \
  --max-weight 0.2 --hold-decay 1.0 --bc-epochs 1

# 评估
latest=$(ls -td outputs/Everything_Money_residual_w0.2/Everything_Money_* | head -n 1)
python experiments/residual_sweep/evaluate_residual.py \
  --checkpoint "$latest/checkpoints/policy.pt" \
  --buffer data/replay_buffer_residual_w0.2/Everything_Money/test.pt \
  --output "$latest/metrics_test.json" \
  --positions-output "$latest/positions_test.csv" \
  --residual-scale 0.2 --max-weight 0.2 --hold-decay 1.0

# 决策导出
python experiments/residual_sweep/export_decisions_residual.py \
  --checkpoint "$latest/checkpoints/policy.pt" \
  --reward-csv data/processed/reward/Everything_Money/test.csv \
  --vocab-path models/embedding/ticker_vocab.json \
  --embedding-path models/embedding/ticker_embedding.pt \
  --output "$latest/signal_decisions_test.csv" \
  --residual-scale 0.2 --max-weight 0.2 --hold-decay 1.0
```

## 关键注意
- 旧版训练有目标维度广播错误，已修复；旧结果需重训后再用。
- fidelity_lambda/residual_scale/hold_decay/单票上限决定与基线的偏离度：过紧或 scale 太小会几乎重合，放松后才可能超越基线。
- 基线完全由文本情感驱动，强化学习只是受控的微调，保持可解释性。***
