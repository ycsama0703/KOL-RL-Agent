# Case Study Narrative Draft (CN)

## Case A: KICL vs Baseline (X · Jake__Wujastyk)

在 `2024-12-05`（映射事件日 `2024-12-02`）出现了一个典型的“结构性分叉”节点。该节点之前，KICL 与 Baseline 的平均净值差为 `-0.0191`；节点后 10 天，平均净值差上升到 `+0.0982`，净值差增量为 `+0.1173`。这不是简单的同向平移，而是策略层面的再分配：在明显正向语料上，KICL 提升了仓位强度（如 `SOUN`，sentiment=`0.8`，`0.067 -> 0.091`）；在偏负向语料上，KICL 主动降到接近空仓（如 `LTC`，sentiment=`-0.2`，`0.028 -> 0.000`）；在中高置信的正向语料上继续加仓（如 `FXI`，sentiment=`0.5`，`0.055 -> 0.094`）。因此，这一分叉可解释为“在不违背语义方向的前提下，提高正向信号权重并剔除负向敞口”，从而形成可观的净值领先。

## Case B: KICL vs WO_HARD (YouTube · The_Maverick_of_Wall_Street)

该案例包含两个拐点节点，且都属于“先拐点、后拉开”而非“峰值追认”。

第一个节点是 `2024-11-21`（映射事件日 `2024-11-11`）。节点前 10 天平均净值差为 `-0.0084`，节点后 10 天提升至 `+0.0743`，净值差增量 `+0.0827`。从动作看，KICL 在偏负或中性语料上更快收缩敞口（如 `AMD` sentiment=`-0.2`，`0.027 -> 0.000`；`TSLA` sentiment=`0.0`，`0.030 -> 0.000`），而 WO_HARD 仍保留非零暴露。换言之，KICL 的领先来自“风险清理更及时”，不是语义反向交易。

第二个节点是 `2025-02-18`（同日映射）。节点前 10 天平均净值差 `+0.0869`，节点后 10 天进一步扩大到 `+0.1839`，增量 `+0.0969`。在核心标的上（`BABA`，sentiment=`0.9`），KICL 保持了正向配置（`0.210 -> 0.241`，相对 WO_HARD 的 comparator action），同时组合层面呈现“ours 上行、WO_HARD 走弱”的持续扩张。这个结果说明，去掉硬约束后并不会稳定提升收益，反而更容易在关键区间暴露出不必要的偏离；而 KICL 的优势更像“受约束的执行补全”。

## One-line Takeaway

两个 case 共同支持同一结论：KICL 的超额收益主要来自 **intent-consistent 的仓位补全与风险清理**，而不是通过背离语料方向来“硬换收益”。

---

## Evidence Sources

- Baseline case figure: `baseline/x/Jake__Wujastyk/case_single_contrast.png`
- Baseline case aligned evidence: `baseline/x/Jake__Wujastyk/case_single_evidence_aligned.csv`
- Ablation case figure: `variant/youtube/The_Maverick_of_Wall_Street/case_single_contrast.png`
- Ablation case aligned evidence: `variant/youtube/The_Maverick_of_Wall_Street/case_single_evidence_aligned.csv`
