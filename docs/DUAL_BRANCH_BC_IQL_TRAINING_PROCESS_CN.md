# 训练动态图绘制说明（最终版）

这张图的定位是：

**一张训练导向的方法结构图，用来说明同一个 baseline-anchored dual-branch residual policy 是如何被两阶段离线训练更新，并最终形成 constrained completion policy 的。**

这张图不是全文总流程图，也不是代码流程图。
它不需要再讲：
- YouTube / X 数据源
- KOL discourse 抽取过程
- structured KOL signals 的形成细节
- portfolio layer / execution layer

这些内容已经由全文总图负责。

这张图只聚焦一件事：

**同一个共享策略主干，在前向时如何做双分支补全，在训练时如何先初始化、再精修，并始终受到意图保持约束。**

---

## 一、这张图希望读者看懂什么

读者看完后，需要立刻明白以下 5 点：

1. **只有一个共享策略主干**
   - 不是两个独立策略
   - 不是 signal / silence 各自训练完再拼起来

2. **前向时有两个分支**
   - baseline 激活时走 signal head
   - baseline 不激活时走 silence / decay head

3. **动作不是自由生成，而是 baseline + residual**
   - 先按 regime 选 residual
   - 再与 baseline anchor 融合
   - 再施加约束，形成最终动作

4. **训练有两个顺序阶段**
   - 第一阶段：behavior-aligned initialization
   - 第二阶段：value-guided refinement

5. **约束塑形贯穿训练与部署**
   - 不是训练结束后才附加
   - 而是持续影响 actor 更新与动作构造

---

## 二、整张图的推荐结构

整张图采用“三层结构”：

### 1. 上层：唯一主干（最重要）
从左到右是一条清晰的策略前向主干：

**Anchor-Aware Inputs**
→ **Baseline-Anchored Dual-Branch Residual Policy**
→ **Baseline-Aware Action Composition**
→ **Final Policy**

这里表达的是：
**同一个策略在一次 forward 中如何从输入走到输出。**

---

### 2. 下层：训练时间轴
在主干下方，只放一条简洁训练时间轴：

**Behavior-Aligned Initialization**
→ **Value-Guided Refinement**

这里表达的是：
**同一个策略参数先被初始化，再被进一步精修。**

注意：
这两部分不是额外模块，不是另一个子系统，而是**对同一个共享策略主干的参数更新阶段**。

---

### 3. 侧边或外侧：约束塑形层
单独保留一个轻量 shaping 模块：

**Intent-Preserving Shaping**

它持续作用于：
- Silence / Decay Head
- Action Composition
- Final Policy

这里表达的是：
**约束不是附加说明，而是对训练与动作形成全过程的持续塑形。**

---

## 三、各模块具体该画什么

---

### A. Anchor-Aware Inputs
这是左侧输入模块，画紧凑即可。

标题：
**Anchor-Aware Inputs**

建议内部只保留 4 组输入语义：
- text / ticker embeddings
- execution context
- compact market factors
- baseline / behavior actions

可保留一个状态公式：

\[
s_t=[e_t^{text}\,\|\,e_t^{ticker}\,\|\,x_t^{core}\,\|\,x_t^{mkt}]
\]

如果要提示市场特征，只在角落用很小的字写：
`ret, vol, volume_z, dist_sma`

不要把这里画成一个长长的 feature table。

---

### B. Baseline-Anchored Dual-Branch Residual Policy
这是整张图的中心，也是视觉最重要的模块。

标题：
**Baseline-Anchored Dual-Branch Residual Policy**

副标题可选：
**one shared policy backbone**

它内部只需要画出四个核心子结构：

#### 1. Shared Backbone
表示共享特征提取 / 策略骨干。

#### 2. Regime Routing
表示根据 baseline 激活情况进行路由。

这里可以写成：

\[
|a_t^{base}|>\epsilon \; ?
\]

或者写：
**Regime Routing by Baseline Activation**

#### 3. Signal Head
标题：
**Signal Head**

副标题：
**refine active signal**

可选小字：
- preserve direction
- sizing adjustment

输出为：

\[
\delta_t^{sig}
\]

#### 4. Silence / Decay Head
标题：
**Silence / Decay Head**

副标题：
**decay / reduce under no-new-entry**

可选小字：
- hold / exit semantics
- no new entry

输出为：

\[
\delta_t^{sil}
\]

注意：
signal head 和 silence / decay head 必须明确画成**同一个 actor 的两个 head**，
而不是两个独立策略模块。

---

### C. Baseline-Aware Action Composition
这是主干中的关键结构模块，不能只画成公式卡片。

标题：
**Baseline-Aware Action Composition**

这个模块表达三件事：
1. 根据 routing 选择 residual
2. residual 与 baseline anchor 融合
3. 在这里施加硬约束

保留两条最关键公式即可：

\[
\delta_t=
\begin{cases}
\delta_t^{sig}, & |a_t^{base}|>\epsilon \\
\delta_t^{sil}, & |a_t^{base}|\le\epsilon
\end{cases}
\]

\[
a_t^\pi=a_t^{base}+\delta_t
\]

旁边可以很小地写一句：
**hard constraints applied here**

不要再放更多 loss 或说明文字。

---

### D. Final Policy
最右侧输出模块。

标题：
**Final Policy**

保留输出记号：

\[
\pi(s_t,a_t^{base}) \rightarrow a_t^\pi
\]

副标题可写：
**constrained completion policy**

不要在这张图里再展开 portfolio execution / portfolio weights。

---

## 四、训练部分应该怎么画（最关键）

这里最容易被画错。

### 错误画法
- 不要把 initialization 画成一个大框
- 不要把 refinement 画成一个大框
- 不要把它们画成主干下面的外挂模块
- 不要让人感觉“还有两个额外模型”

### 正确画法
训练要画成：

**一条时间轴 + 向上更新箭头**

也就是说：

#### 下方时间轴
**Behavior-Aligned Initialization**
→
**Value-Guided Refinement**

这里表达的是：
**训练阶段顺序**

#### 从下往上的短箭头
表达的是：
**参数更新 / 监督信号注入**

---

### 训练阶段 1：Behavior-Aligned Initialization
这是左下第一个训练节点。

标题：
**Behavior-Aligned Initialization**

建议只保留极少量说明：
- fit executable behavior
- initialize same policy

从这个节点往上打箭头到：
- Shared Backbone
- Signal Head
- Silence / Decay Head

这里的语义非常重要：

**这表示第一阶段训练在更新同一个共享策略主干及其两个 head 的参数。**

不是新增一个模块。

---

### 训练阶段 2：Value-Guided Refinement
这是第二个训练节点。

标题：
**Value-Guided Refinement**

建议只保留极少量说明：
- extended state
- behavior residuals
- actor refinement

如果要放公式，只保留很小的两个：

\[
\tilde s_t=[s_t\|a_t^{base}]
\]

\[
\delta_t^{beh}=a_t^{beh}-a_t^{base}
\]

从这个节点往上打箭头到：
- Signal Head
- Silence / Decay Head
- Baseline-Aware Action Composition

这里表达的是：

**第二阶段训练继续更新同一个策略，并进一步塑造动作构造方式。**

不是另一套独立策略。

---

## 五、约束塑形怎么画

不要把约束画成一个巨大的底部横条，也不要画成一个独立训练模块。

标题：
**Intent-Preserving Shaping**

里面只保留 4 个关键词：
- no unsupported entry
- no reversal
- baseline alignment
- fidelity shaping

从这个模块打短箭头到：
- Silence / Decay Head
- Baseline-Aware Action Composition
- Final Policy

这里表达的是：

**约束塑形持续作用于最敏感的动作生成部分，而不是训练结束后才附加。**

---

## 六、箭头语义必须区分清楚

这是整张图最重要的画法要求。

### 1. 上层主干箭头
表示：
**前向数据流**

只用于：
Inputs → Policy → Composition → Final Policy

---

### 2. 下方时间轴箭头
表示：
**训练阶段顺序**

只用于：
Initialization → Refinement

---

### 3. 从下往上的箭头
表示：
**参数更新 / 监督信号注入**

只用于：
- Initialization 更新 actor 参数
- Refinement 更新 actor 与 composition 相关参数

---

### 4. 右侧 shaping 的短箭头
表示：
**约束塑形作用对象**

只用于：
- Silence / Decay Head
- Action Composition
- Final Policy

一定要避免把这些不同语义的箭头画成一样的大弯箭头，否则读者会看不清。

---

## 七、整张图最需要避免的错误

1. **不要把两个分支画成两个独立策略**
   - 它们是同一个 residual actor 的两个 head

2. **不要把训练阶段画成外挂模块**
   - initialization 和 refinement 是更新同一个策略，不是额外系统

3. **不要把 Action Composition 画成纯公式卡片**
   - 它是一个真正的结构模块

4. **不要堆太多字**
   - 每个模块只保留标题 + 一行副标题 + 必要公式

5. **不要画成长得像全文总流程图**
   - 这张图不需要数据源、文本解析、portfolio execution

6. **不要用大回环把整张图包起来**
   - 容易让训练箭头和前向箭头混在一起

---

## 八、一句话总结（可作为绘图核心依据）

这张图的本质是：

**一个共享的 baseline-anchored dual-branch residual policy，在前向时按 regime 分流，在训练时先做 behavior-aligned initialization，再做 value-guided refinement，并持续受到 intent-preserving shaping 约束，最终输出一个 constrained completion policy。**