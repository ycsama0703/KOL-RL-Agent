# KOL-RL-Agent 报告素材（动机与方法详述）

## 核心动机
- **现象**：KOL 文本大量给出“买入/建仓”建议，却很少明确“持有多久/何时卖”。跨三家 KOL（Everything_Money / Invest_with_Henry / MarketBeat）统计显示：
  - 正面信号占 59–72%，负面仅 9–20%；中性约 18–22%。
  - 40% 左右的股票只被提及 1 次（建仓后再无信号）。
  - 仅看“≥2 次提及”的票：尾部沉默中位数 22–104 天，90% 分位可达 187–259 天；负面沉默期更长（最长超 300 天）。
  - 情感翻转率低或翻转很晚（中位 44–112 天，长尾 >200 天；MarketBeat 翻转率仅 ~18%）。
- **结论**：原始语料缺乏退出指引，大量持仓长期沉默或从未反向，这为“让 RL 学习沉默期的衰减/退出”提供正当性。

## 基线策略（文本驱动、可解释）
- 打分：`baseline_raw_score = tanh(2 * sentiment * confidence)`，乘情感符号得到多/空原始分数。
- 组合：多空、连续持仓，未提及的票默认继承（可设 `hold_decay`），按绝对值归一化，单票上限（默认 20%）。
- 当日基线持仓：将签名打分送入组合层得到基线权重，作为行为策略。

## 改进的 RL 训练方法（解决退出缺失）
- **动作分支**：
  - 有信号的 ticker：同向缩放基线权重 `policy_sig = baseline * (1 + residual_scale * tanh(delta_sig))`，不翻转方向。
  - 无信号的 ticker：对昨仓衰减 `policy_nosig = last_position * sigmoid(decay_scale * delta_decay)`，由 RL 学习“沉默期减仓/退出”。
  - 最终动作：`policy = has_signal * policy_sig + (1 - has_signal) * policy_nosig`，`has_signal` 来自基线是否非零。
- **状态增强**：在 replay buffer 中加入 `silence_days`（距上次提及天数）和 `last_position`；文本/情感/ticker 嵌入保持不变。
- **约束放松**：
  - Actor 惩罚与 `fidelity_lambda` 绑定，λ=0 时不再钉死基线。
  - BC 仅作轻量初始化（可 0–1 epoch），不约束无信号分支。
  - Actor 损失：优势加权 Q + 信号分支的轻量 fidelity 正则；无信号分支自由学习衰减。
- **奖励与风险**：主奖励仍为组合收益；可选加入轻量持仓/库存或时间成本，鼓励长时间无信号时适度减仓；保持多空上限/总权重归一。
- **训练细节**：Critic/Value 在行为动作上训练（可扩展混采策略），优势驱动策略偏离；组合层保持多空、上限、连续持仓。

## 数据管线（cleaned → replay buffer）
一键脚本：`python scripts/run_full_pipeline.py --model <HF或本地模型> --cleaned data/processed/cleaned --embeddings data/processed/embeddings --enriched data/processed/enriched --reward data/processed/reward --replay data/replay_buffer --vocab models/embedding/ticker_vocab.json --ticker-emb models/embedding/ticker_embedding.pt --price-days 5 --batch-size 32 --device cuda`
- 起点：`data/processed/cleaned/<KOL>/<split>.csv`（文本/情感/公司）。
- 终点：`data/replay_buffer/<KOL>/{train,val,test}.pt`，包含 state（含 silence_days）、基线动作、奖励、元信息。
- 只有 embedding 步骤需要显卡，其余 CPU 即可。

## 训练/评测示例（双分支策略）
```bash
# 训练
python experiments/residual_sweep/train_residual.py \
  --kol Everything_Money --replay-dir data/replay_buffer \
  --output-dir outputs/Everything_Money_residual \
  --fidelity-lambda 0.3 --residual-scale 0.2 --decay-scale 0.5 \
  --max-weight 0.2 --hold-decay 1.0 --bc-epochs 1

# 评测
latest=$(ls -td outputs/Everything_Money_residual/Everything_Money_* | head -n 1)
python experiments/residual_sweep/evaluate_residual.py \
  --checkpoint "${latest}/checkpoints/policy.pt" \
  --buffer data/replay_buffer/Everything_Money/test.pt \
  --output "${latest}/metrics_test.json" \
  --positions-output "${latest}/positions_test.csv" \
  --residual-scale 0.2 --decay-scale 0.5 --max-weight 0.2 --hold-decay 1.0

# 决策导出
python experiments/residual_sweep/export_decisions_residual.py \
  --checkpoint "${latest}/checkpoints/policy.pt" \
  --reward-csv data/processed/reward/Everything_Money/test.csv \
  --vocab-path models/embedding/ticker_vocab.json \
  --embedding-path models/embedding/ticker_embedding.pt \
  --output "${latest}/signal_decisions_test.csv" \
  --residual-scale 0.2 --decay-scale 0.5 --max-weight 0.2 --hold-decay 1.0
```
可 sweep：`fidelity_lambda`（0/0.1/0.3）、`residual_scale`（0.2/0.3）、`decay_scale`（0.5/1.0）、`hold_decay`（1.0/0.99），单票上限先固定 0.2。

## 结论与价值主张
- 通过数据统计，实证“KOL 信号缺退出/反向指引、沉默期长、单次提及多”，为 RL 补全退出策略提供依据。
- 基线保持文本可解释性，RL 专注沉默期的持有/衰减，提升组合收益/风控的可能性。
- 可扩展：增加“距上次提及”特征、时间成本，或对未提及票的自然衰减，以进一步优化退出行为。***
