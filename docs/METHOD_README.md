# KOL-RL-Agent 方法说明（与当前 `train.py` 对齐）

本说明聚焦当前主线训练脚本 `train.py` 的方法与评估逻辑。

## 1) 任务定义与数据约定

- 训练目标：学习一个相对 KOL baseline 动作的残差策略（residual policy）。
- Replay batch 关键字段：
  - `state`, `next_state`
  - `action`（behavior action）
  - `baseline_action`, `next_baseline_action`
  - `reward`, `done`
- 策略输出形式：
  - `policy_action = baseline_action + delta`

## 2) 策略结构与约束

- `ActorNetwork` 兼容两种输出：
  - 直接输出 tensor `delta`
  - 输出 dict（`delta_signal` / `delta_decay`），按 baseline 是否接近 0 选择分支
- 训练期（soft）使用：
  - `build_policy_action_for_training`: 仅做 `delta` 截断，不做硬门控
  - `intent_penalties_soft`: 可微的 no-entry / no-reversal 软惩罚
- 评估期（hard）使用：
  - `apply_intent_constraints` 强制执行：
    - baseline≈0 时不允许新开仓
    - 不允许与 baseline 方向反转

## 3) 训练流程（BC -> IQL）

- BC 阶段：
  - 默认 `bc_epochs=10`
  - 默认 `bc_batch_size=256`
  - 损失由以下部分组成（可加权）：
    - 拟合 behavior action
    - anchor 到 baseline action
    - soft no-entry penalty
    - soft no-reversal penalty
- IQL 阶段：
  - 默认 `iql_steps=200000`
  - critic/value 输入是 `concat(state, baseline_action)`
  - critic 回归 `target_q = reward_aug + gamma*(1-done)*V(next_state)`
  - value 使用 expectile loss（默认 `expectile=0.7`）
  - actor 使用 advantage-weighted 回归 + 软对齐/软约束：
    - `loss_fit`（拟合 behavior）
    - `actor_align_lambda * loss_align`
    - `entry_penalty_lambda * loss_entry`
    - `reversal_penalty_lambda * loss_rev`

## 4) 奖励与保真度 shaping

- IQL 中使用 reward shaping：
  - `reward_aug = reward - fidelity_lambda * ||policy_action - baseline_action||^2`
- 目的：控制偏离 baseline 的幅度，避免策略过度漂移。

## 5) 评估逻辑

- 评估入口：`evaluate(...)`（在 `train.py` 内）
- 流程：
  - actor 输出 `delta`
  - 应用 `apply_intent_constraints`（hard 约束）
  - 通过 `PortfolioLayer.allocate` 得到组合权重
  - 按日聚合收益得到 `daily_returns`
- 输出指标：
  - `cumulative_return`
  - `sharpe`（`sqrt(252)` 年化）
  - `max_drawdown`

## 6) 当前默认配置（主线）

- 默认 KOL：`Ale_s_World_of_Stocks`
- 默认 replay 路径：`data/buffer_22-24_end1231`
- 其他关键默认超参：
  - `bc_batch_size=256`, `iql_batch_size=256`
  - `actor/critic/value lr = 3e-4`
  - `gamma=0.99`, `temperature_beta=3.0`
  - `fidelity_lambda=0.1`
  - `actor_align_lambda=0.1`
  - `entry_penalty_lambda=0.1`
  - `reversal_penalty_lambda=0.1`
