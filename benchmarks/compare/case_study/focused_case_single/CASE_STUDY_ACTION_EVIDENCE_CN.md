# Case Study Action Evidence (for paper writing)

## 1) 目的
本文件用于支撑 case study 叙事中的核心论点：

- `KICL (Ours)` 在不发生硬背叛（UER/DRR 约束）的前提下，
- 学到执行层补全（仓位强弱、加减仓时机），
- 从而形成比 baseline 更优的仓位路径（exposure path）。

注意：建议写成“支持证据（supporting evidence）”，不要写成“证明模型预测了未来价格”。

---

## 2) 数据来源

- X case（Jake）：[positions_test.csv](/Users/yuncongliu/Documents/nus%20academic/KOL-RL-Agent/benchmarks/compare/case_study/raw_kicl/x/Jake__Wujastyk/positions_test.csv)
- YouTube case（Maverick）：[positions_test.csv](/Users/yuncongliu/Documents/nus%20academic/KOL-RL-Agent/benchmarks/compare/case_study/raw_kicl/youtube/The_Maverick_of_Wall_Street/positions_test.csv)
- 对齐后的节点证据（X）：[case_single_evidence_aligned.csv](/Users/yuncongliu/Documents/nus%20academic/KOL-RL-Agent/benchmarks/compare/case_study/focused_case_single/baseline/x/Jake__Wujastyk/case_single_evidence_aligned.csv)
- 对齐后的节点证据（YouTube vs WO-H）：[case_single_evidence_aligned.csv](/Users/yuncongliu/Documents/nus%20academic/KOL-RL-Agent/benchmarks/compare/case_study/focused_case_single/variant/youtube/The_Maverick_of_Wall_Street/case_single_evidence_aligned.csv)

---

## 3) Case A: FXI (X / Jake__Wujastyk, statement day = 2024-12-02)

### 3.1 节点语义（已对齐）
- Ticker: `FXI`
- Sentiment: `+0.5`
- 文本片段：`...china names may be the highest performers. $fxi`

### 3.2 Ours vs Baseline 动作对照（节点后）

| Date | Baseline action | Ours action | 解释 |
|---|---:|---:|---|
| 2024-12-02 | 0.0554 | 0.0938 | Ours 在同方向上更积极加仓 |
| 2024-12-23 | 0.0000 | 0.0000 | 双方都平仓 |
| 2024-12-24 | 0.1321 | 0.1558 | 再开仓时 Ours 仓位仍更高 |
| 2024-12-26 | 0.0000 | 0.0000 | 双方都再次平仓 |

### 3.3 可用于论文的表述
- 在 FXI 该节点上，Ours 没有改变“多头方向”语义，而是通过更高权重完成执行层补全，体现了对上行窗口更强的风险暴露配置。

---

## 4) Case B: TSLA (YouTube / The_Maverick_of_Wall_Street, statement day = 2024-11-11)

### 4.1 节点语义（已对齐）
- Ticker: `TSLA`
- Sentiment: `0.0`（中性/观望语气）
- 文本片段：`...tesla leave alone for a little bit...`

### 4.2 Ours vs Baseline 动作对照（节点后）

| Date | Baseline action | Ours action | 解释 |
|---|---:|---:|---|
| 2024-11-11 | 0.0000 | 0.0000 | 双方先平仓 |
| 2024-11-27 | 0.0554 | 0.0483 | Ours 略保守 |
| 2024-12-13 | 0.0615 | 0.0281 | Ours 显著更保守 |
| 2024-12-24 | 0.1795 | 0.1754 | 接近，但 Ours 仍略低 |
| 2025-01-06 | 0.1817 | 0.1215 | Ours 继续控制暴露 |
| 2025-01-21 | 0.0000 | 0.0000 | 双方平仓 |

### 4.3 可用于论文的表述
- 在 TSLA 该节点上，Ours 没有引入反向交易，而是持续以更低仓位执行，体现“同意图下的风险约束型补全”。

---

## 5) 面向论文的结论句模板

可直接给 GPT 使用的主句：

1. `These two nodes show that KICL does not rely on hard-intent violations to gain excess return; instead, it refines execution intensity under the same directional intent.`
2. `For FXI, KICL increases exposure more aggressively than the baseline after a bullish cue; for TSLA, it keeps a more conservative exposure path under a neutral/uncertain cue.`
3. `This supports the claim that KICL performs intent-preserving policy completion rather than unconstrained signal chasing.`

---

## 6) 写作边界（避免审稿风险）

- 建议写：`better exposure path`, `execution-level refinement`, `risk-aware completion`
- 避免写：`predicts future prices correctly`, `proves market forecasting ability`
- 建议用词：`supports`, `is consistent with`, `provides evidence for`

