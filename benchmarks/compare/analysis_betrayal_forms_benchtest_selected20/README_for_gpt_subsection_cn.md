# 背叛行为分析 Subsection（重排版，按论文叙事）

## 1) Experiment A: Betrayal Form Profile（结构画像）

目标：回答“各方法主要以哪种形式背叛”。

指标：
- `UER`（unsupported entry rate）
- `DRR`（direction reversal rate）
- `BD`（mean abs deviation）
- `CG`（1 - baseline policy corr）
- `HVC = UER + DRR`（硬背叛强度，可在文中补充）

输出：
- 每平台一张 heatmap（scaled）+ raw table  

主图：
- `benchmarks/compare/analysis_betrayal_forms_benchtest_selected20/betrayal_forms_heatmap_scaled.png`

主结论（可直接写）：
- KICL 在 `UER/DRR` 上几乎被压到 0（硬背叛极低）。
- KICL 的 `BD` 也保持在低位（软偏离可控）。
- CQL/TD3BC 在两平台都呈现明显硬背叛主导。

---

## 2) Experiment B: Profit-linked Betrayal（收益-背叛耦合）

目标：回答“收益提升是否来自背叛”。

条件事件：
- `event_return > 0`（profit event）

本轮采用的新口径（你最新要求）：
- 直接统计“超额收益事件下”的背叛类型概率：
  - `P(B_hard | excess>0)`
  - `P(B_soft | excess>0)`
  - 其中默认 `excess = (policy_action - baseline_action) * reward`

再做分解（关键）：
- `B_hard = reversal OR unsupported_entry`
- `B_soft = dev_flag`（并可辅以连续 `|dev|` 解读）

核心判断：
- 如果 uplift 主要来自 `B_soft`，且 `B_hard` 近零，则应解释为“意图内执行补全”，而非“硬背叛套利”。

主图：
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/betrayal_hard_soft_decomposition_story.png`
- 新图（按“超额收益条件概率”直接统计）：
  - `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/excess_betrayal_type_probability.png`

补充文件：
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/excess_return_betrayal_pooled.csv`
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/excess_return_betrayal_hard_soft_decomposition.csv`
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/hard_only_betrayal_summary.csv`
- 新口径主表：
  - `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/excess_betrayal_type_by_method_source.csv`

关键观察（新口径，20/20覆盖）：
- KICL 在 `excess>0` 时：
  - X：`P(B_hard|excess>0)=0.000327`，`P(B_soft|excess>0)=0.870485`
  - YouTube：`P(B_hard|excess>0)=0.000222`，`P(B_soft|excess>0)=0.826229`
- 其他方法通常具有显著更高的硬背叛概率（尤其 CQL/TD3BC）：
  - X：
    - CQL：`P(B_hard|excess>0)=0.406890`
    - TD3BC：`P(B_hard|excess>0)=0.405698`
  - YouTube：
    - CQL：`P(B_hard|excess>0)=0.425204`
    - TD3BC：`P(B_hard|excess>0)=0.593909`

一句话解释：
- KICL 在超额收益事件里几乎不触发硬背叛，主要是软补全；
- 其他若干方法在超额收益事件里仍伴随较高硬背叛概率。

---

## 3) 论文里建议怎么讲（连贯故事）

- A 实验证明“背叛形态分布”：KICL 的硬背叛几乎被压住，且软偏离可控。  
- B 实验证明“收益关联来源”：KICL 的收益关联 uplift 主要来自 soft completion，不是 hard violation。  
- 两个实验合起来支撑主张：KICL 的超额收益来自“受约束的执行补全”，不是偏离 KOL 核心意图。  

---

## 4) 口径说明（建议在文中短句交代）

- `B_any` 口径用于与既有图表保持可比性。  
- 关键结论以 hard/soft 分解为准：KICL 的 hard 相关量级接近 0。  
- 若审稿人偏好严格定义，可附 `hard-only` 版本（仅把 `B_hard` 计为背叛）。  
