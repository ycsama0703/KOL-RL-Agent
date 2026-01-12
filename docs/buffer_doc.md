# Benchmark ② 详细使用说明  
## —— 如何规范使用 train.pt / test.pt（Offline RL / 收益优化）

> 本文档用于 **组内统一规范**：  
> 任何成员在使用 `train.pt` / `test.pt` 跑 Benchmark ② 前，**必须完整阅读本说明**。  
> 目标是：**保证所有方法在同一个“世界”里比较，结果才有意义。**

---

## 0. 一句话总览（先读这个）

- `train.pt / test.pt` 是 **离线强化学习的 replay buffer**
- 它定义了一个 **已经冻结的历史交易世界**
- 你 **只能换算法，不能换世界**
- 所有 Benchmark ② 的公平性，都来自这一点

---

## 1. train.pt / test.pt 到底是什么？

### 1.1 技术层面

- 文件格式：PyTorch `.pt`
- 序列化方式：`torch.save()`
- 加载方式：`torch.load()`

本质上，它是一个 **Python 对象（dict）+ Tensor 的集合**。

---

### 1.2 语义层面（非常重要）

`train.pt / test.pt` 不是简单的数据表，而是一个 **MDP（Markov Decision Process）的离线记录**。

它记录的是：

- 当时看到了什么状态（state）
- 当时做了什么动作（action）
- 市场给了什么反馈（reward）
- 世界如何演化到下一步（next_state）
- 是否结束（done）

这正是 Offline RL 所需的最小完备信息。

---

### 1.3 在你们项目里的真实含义

你可以把 replay buffer 理解为：

> “如果我们过去**严格按照 KOL baseline 策略交易**，  
> 在真实市场中，会经历怎样的一条历史轨迹。”

之后所有 Benchmark ② 方法，只是在 **这条既定历史上学习更好的策略**。

---

## 2. 它解决的是什么问题？（Benchmark ② 的问题定义）

Benchmark ② 的问题是：

> 在 **固定的历史数据（replay buffer）** 下，  
> 不同策略优化 / 强化学习方法，  
> 谁能获得更好的 **风险调整后收益**？

因此：

- 我们比较的是 **算法能力**
- 不是数据质量
- 不是信号来源
- 更不是“谁看到了更多信息”

---

## 3. 适用与不适用范围（边界一定要清楚）

### 3.1 允许使用的任务（✔）

- Behavior Cloning（BC-only）
- Offline RL（IQL / CQL / 其变体）
- 去约束 vs 加约束的 RL 对比
- Market-only RL（屏蔽文本 embedding）
- Benchmark ② 的所有收益类对比

---

### 3.2 明确不允许的任务（✘）

- 文本 → 策略忠诚度（Benchmark ①）
- 情绪分类 / 文本分类
- 任何需要原始文本输入的模型
- 任何形式的 Online RL
- 使用未来信息的模型

原因很简单：  
**文本已经被 embedding，不可逆；世界已经被冻结。**

---

## 4. 如何正确加载 replay buffer

推荐的标准加载方式如下（示例）：

    import torch

    buffer = torch.load(
        "data/replay_buffer/<KOL>/train.pt",
        map_location="cpu"
    )

    print(buffer.keys())

正常情况下，你会看到如下字段：

- states
- actions
- rewards
- portfolio_rewards
- next_states
- dones
- meta

如果字段不一致，**不要直接用，先确认版本**。

---

## 5. Replay Buffer 字段逐一详解（核心部分）

### 5.1 states（状态）

**类型**
- torch.Tensor
- 形状：[N, state_dim]

**语义**
- 表示时间 t 时刻，策略可观测到的完整状态
- 一般由以下部分拼接而成：
  - KOL 文本 embedding（已处理，不是原文）
  - 市场特征（价格、波动率等）
  - 上一期仓位（last_position，用于建模持有 / 退出）

**规则**
- ✔ 允许作为模型输入
- ✘ 不允许改维度
- ✘ 不允许新增特征

---

### 5.2 actions（行为动作）

**类型**
- torch.Tensor
- 形状：[N, action_dim]

**语义**
- 历史中实际采取的 **baseline / 行为策略动作**
- 定义了 replay buffer 的行为分布

**用途**
- Behavior Cloning 的监督信号
- Offline RL 的行为约束

**规则**
- ✔ 允许用于训练
- ✘ 不允许修改或重定义语义

---

### 5.3 rewards（奖励）

**类型**
- torch.Tensor
- 形状：[N]

**语义**
- 单步 reward
- 在当前项目中：**next-day return（下一交易日收益）**

**非常重要的规则**
- 这是 **唯一允许用于训练的 reward**
- ✘ 禁止重新计算
- ✘ 禁止替换为长期 reward
- ✘ 禁止混入 portfolio_rewards

---

### 5.4 portfolio_rewards（组合收益，辅助字段）

**语义**
- 已计算好的组合层面收益
- 通常用于：
  - 分析
  - 可视化
  - sanity check

**规则**
- ✔ 允许用于分析
- ✘ 禁止作为训练 reward
- ✘ 禁止作为 state 特征

---

### 5.5 next_states（下一状态）

**类型**
- torch.Tensor
- 形状：[N, state_dim]

**语义**
- 时间 t+1 时刻的状态
- 用于价值函数 / Q 函数学习

**规则**
- ✔ Offline RL 可用
- ✘ BC-only 不应使用

---

### 5.6 dones（终止标记）

**类型**
- torch.Tensor
- 形状：[N]

**语义**
- 标记一个 episode 是否结束
- 通常是：
  - 某个 ticker 的最后一天
  - 或时间序列终点

---

### 5.7 meta（元信息，只读）

**可能包含**
- ticker
- 日期
- KOL / video id

**允许用途**
- Debug
- Case study
- 结果分组分析

**严格禁止**
- 作为模型输入
- 作为训练特征
- 参与决策逻辑

---

## 6. 合法使用范式（你应该怎么写代码）

### 6.1 Behavior Cloning（BC-only）

    pred_action = actor(states)
    loss = ((pred_action - actions) ** 2).mean()

- 不使用 reward
- 不使用 next_states

---

### 6.2 Offline RL（IQL / CQL 等）

使用完整 transition：

- states
- actions
- rewards
- next_states
- dones

规则：
- offline only
- 不修改 buffer
- 不引入新数据

---

### 6.3 Market-only 基线

允许操作：
- 在 states 中 **屏蔽 / 删除文本 embedding 维度**
- 保留 market + last_position
- train.pt / test.pt 必须保持不变

---

## 7. 明确禁止的行为（出现即判无效）

以下任一行为，都会 **直接导致 benchmark 结果不可用**：

- 修改或重新生成 replay buffer
- 修改 reward 定义
- 使用 portfolio_rewards 作为训练信号
- 向 state 中添加新特征
- 使用 meta 作为模型输入
- 在 test.pt 上调参 / early stopping
- 使用任何未来信息

---

## 8. 标准评估流程（必须一致）

1. 仅使用 train.pt 训练模型
2. 训练完成后冻结参数
3. 在 test.pt 上进行策略回放
4. 使用项目统一的评估脚本计算：
   - cumulative return
   - Sharpe ratio
   - max drawdown
   - turnover

---

## 9. 一个非常重要的理解方式

> Replay buffer = 世界  
> 算法 = 在世界中的行为方式  

你可以改变算法，但不能改变世界。

---

## 10. 组内硬性规则（一句话）

Benchmark ②：  
**Same buffer · Same reward · Same evaluation · Different algorithms only**

---

## 11. 强烈建议的项目规范

- 将 replay buffer 标记为只读
- 明确 buffer 版本号（如 ReplayBuffer-v1.0）
- 所有实验结果必须注明所用 buffer 版本

---

（建议保存为：docs/benchmark2_replay_buffer_usage.md）