# KOL-RL-Agent（KOL 文本 → 交易策略智能体）

本仓库实现了从 **KOL 文本语料 + 行情特征** 到 **多空组合决策** 的完整离线 RL Pipeline：

- **线上输入**  
  - `kol_text`：当日 KOL 文本（或视频文案）  
  - `market_state`：当日行情特征（如 returns / volatility / turnover 等）  
- **线上输出**  
  - `target_position`：目标仓位（-1 ~ 1，多空）  
  - `confidence`：置信度（可选）  
  - `timestamp`：决策时间戳  

当前完成的“第一阶段”打通了：

> 原始语料 → 清洗/切分 → ModernBERT embedding → Reward 构建 → Baseline + Ticker Embedding →  
> Replay Buffer（含连续持仓）→ BC + IQL 训练 → 测试集回放 → 决策明细导出 → 净值曲线可视化。

下文按 **目录结构** → **Pipeline 阶段** → **训练/评估** → **推理对接** 说明。


============================================================
0. Quick Start（最短路径）
============================================================

1）**生成 cleaned 数据（按 KOL / 时间切分）**

```bash
# 原始 CSV → 统一格式
python scripts/build_dataset.py --input data/input --output data/processed/total/kol_text_with_sentiment.csv

# 按 video_id 时间划分 train/val/test
python scripts/split_by_video_time.py --input data/processed/total/kol_text_with_sentiment.csv --output data/processed/splits

# 文本清洗，生成 cleaned 数据
python scripts/clean_dataset.py --input data/processed/splits --output data/processed/cleaned
```

2）**embedding + reward + baseline + buffer**

```bash
# ModernBERT 文本 embedding
python scripts/generate_embeddings.py \
  --model answerdotai/modernbert-base \
  --input data/processed/cleaned \
  --output data/processed/embeddings \
  --batch-size 32 --normalize

# 行情特征 + reward（如已跑过可跳过）
python scripts/augment_with_market_data.py --input data/processed/cleaned --output data/processed/enriched ...
python scripts/generate_reward.py --input data/processed/enriched --output data/processed/reward

# baseline + ticker embedding + replay buffer（含 last_position）
python scripts/run_replay_pipeline.py \
  --reward-dir data/processed/reward \
  --vocab-path models/embedding/ticker_vocab.json \
  --embedding-path models/embedding/ticker_embedding.pt \
  --replay-dir data/replay_buffer
```

3）**训练（BC → IQL）**

```bash
python train.py \
  --kol Everything_Money \
  --replay-dir data/replay_buffer \
  --ticker-vocab models/embedding/ticker_vocab.json \
  --ticker-embedding models/embedding/ticker_embedding.pt \
  --output-dir outputs
```

4）**测试 + 指标**

```bash
# 在 test.pt 上回放策略，输出整体指标 + 逐 ticker 持仓轨迹（可选）
python scripts/evaluate_run.py \
  --checkpoint outputs/Everything_Money_<时间戳>/checkpoints/policy.pt \
  --buffer data/replay_buffer/Everything_Money/test.pt \
  --output outputs/Everything_Money_<时间戳>/metrics_test.json \
  --positions-output outputs/Everything_Money_<时间戳>/positions_test.csv \
  --action-threshold 0.02
```

5）**按视频查看决策细节 + 画净值曲线（可选）**

```bash
# 逐视频决策明细（带原文）
python scripts/export_signal_decisions.py \
  --checkpoint outputs/Everything_Money_<时间戳>/checkpoints/policy.pt \
  --reward-csv data/processed/reward/Everything_Money/test.csv \
  --vocab-path models/embedding/ticker_vocab.json \
  --embedding-path models/embedding/ticker_embedding.pt \
  --output outputs/Everything_Money_<时间戳>/signal_decisions_test.csv

# 净值曲线 + 市场基准（例如 SPY）
python scripts/plot_equity_curve.py \
  --signal-decisions outputs/Everything_Money_<时间戳>/signal_decisions_test.csv \
  --output-figure outputs/Everything_Money_<时间戳>/equity_test_with_spy.png \
  --benchmark-ticker SPY \
  --benchmark-label "SPY (market)"
```


============================================================
0+. 一键数据 Pipeline（cleaned → replay buffer）
============================================================
从 `data/processed/cleaned/<KOL>/<split>.csv` 一键生成可训练的 replay buffer（含基线、ticker 词表/向量），覆盖所有 KOL 子目录：

```bash
python scripts/run_full_pipeline.py \
  --model answerdotai/modernbert-base \            # HF 模型名或本地路径；生成文本 embedding，需显卡更快
  --cleaned data/processed/cleaned \               # 起点：cleaned CSV 根目录
  --embeddings data/processed/embeddings \         # 输出：文本 embedding
  --enriched data/processed/enriched \             # 输出：enriched（含行情窗口）
  --reward data/processed/reward \                 # 输出：reward CSV
  --replay data/replay_buffer \                    # 输出：replay buffers（终点，可直接训练）
  --vocab models/embedding/ticker_vocab.json \     # ticker 词表
  --ticker-emb models/embedding/ticker_embedding.pt \ # ticker 向量
  --price-days 5 \
  --batch-size 32 \
  --device cuda            # 可选；仅 embedding 步骤用显卡加速，其余步骤 CPU 即可
```

起点：`data/processed/cleaned/<KOL>/<split>.csv`  
终点：`data/replay_buffer/<KOL>/{train,val,test}.pt`，可直接用于 `train.py`/`evaluate_run.py`。  
说明：管道步骤依次为 embedding → enrich（行情补全） → reward → 基线/词表 → replay buffer；除 embedding 外，其他步骤不依赖 GPU。


============================================================
1. 目录结构 & 依赖
============================================================

```text
data/
  input/                          # 原始语料（TikTok/YouTube/公司列表等）
  processed/
    total/                        # build_dataset.py 输出全集（大文件，已忽略）
    top_channels/                 # 高频 KOL 拆分结果
    splits/<KOL>/<split>.csv      # train/val/test（按视频时间划分）
    cleaned/<KOL>/<split>.csv     # 清洗后的训练输入
    embeddings/                   # ModernBERT 文本 embedding
    reward/<KOL>/<split>.csv      # 含 reward_1d / baseline_raw_score 的表
data/replay_buffer/<KOL>/<split>.pt  # Offline RL 用的 replay buffer

models/
  embedding/
    ticker_vocab.json             # 股票词表
    ticker_embedding.pt           # 股票 embedding 权重

outputs/
  <KOL>_<时间戳>/
    logs/training.log             # 本次训练日志
    checkpoints/                  # actor.pt / critic.pt / value.pt / policy.pt
    run_summary.json              # 配置 + 关键指标
    metrics_test.json             # 测试集整体指标（可选）
    positions_test.csv            # 测试集逐 ticker 持仓轨迹（可选）
    signal_decisions_test.csv     # 测试集逐视频决策明细（可选）
    equity_test*.png              # 净值曲线图（可选）

src/
  preprocessing/                  # 文本清洗/切块/构建 dataset
  embedding/                      # 文本 encoder（ModernBERT 接入点）
  state/                          # 状态构建（含 ticker embedding）
  portfolio/                      # PortfolioLayer（raw_score → 权重/资金分配）
  rl/                             # BC + IQL/CQL 等 RL 模块
  training/                       # ReplayDataset, Actor/Critic/Value 网络等
  inference/                      # RLKolAgent 推理接口
  evaluation/                     # 策略回放 / 持仓分析工具
  pipeline/                       # replay buffer 构建等共享工具
  utils/                          # logger 等

scripts/
  build_dataset.py                # 原始 CSV → 统一格式
  split_top_channels.py           # 挑选高频 KOL
  split_by_video_time.py          # 按 video_id 时间划分 train/val/test
  clean_dataset.py                # 文本清洗/归一化
  generate_embeddings.py          # ModernBERT 文本 embedding
  augment_with_market_data.py     # 行情特征补齐
  generate_reward.py              # 基于 yfinance 构建 reward_1d
  add_baseline_action.py          # 基线动作 baseline_raw_score
  build_ticker_embedding.py       # ticker_vocab + ticker_embedding
  build_replay_buffer.py          # 构造 replay buffer（含 last_position）
  run_replay_pipeline.py          # 一键运行 baseline → vocab → buffer
  evaluate_run.py                 # 在某个 split 上回放策略
  compare_decisions.py            # 训练前后策略对比
  export_signal_decisions.py      # 逐视频决策明细导出
  plot_equity_curve.py            # 净值曲线 + 市场基准可视化

train.py                          # BC + IQL 训练入口
infer.py                          # CLI 推理 demo
```

**依赖安装（最小）**：

```bash
pip install torch transformers sentence-transformers numpy pandas scikit-learn d3rlpy tqdm yfinance
# 如需画图：
pip install matplotlib
```


============================================================
2. 阶段一：数据构造 Pipeline（从 CSV 到 Replay Buffer）
============================================================

> 目标：得到 `data/replay_buffer/<KOL>/train|val|test.pt`，每个样本包含  
> `state, action, reward, next_state, done, meta`，且 `state` 中显式包含上一期持仓 `last_position`。

### 2.1 原始语料 → 统一格式

1. **聚合多源 CSV**（如需要）  
   ```bash
   python scripts/build_dataset.py \
     --input data/input \
     --output data/processed/total/kol_text_with_sentiment.csv
   ```
   聚合 TikTok / YouTube 等多源 CSV，统一字段（`text/company/sentiment/confidence/...`）。

2. **按 KOL / 时间切分 train/val/test**  
   ```bash
   # 挑选重点 KOL（可选）
   python scripts/split_top_channels.py ...

   # 按 video_id 时间顺序划分 train/val/test
   python scripts/split_by_video_time.py \
     --input data/processed/total/kol_text_with_sentiment.csv \
     --output data/processed/splits/<KOL>/
   ```

3. **清洗文本**  
   ```bash
   python scripts/clean_dataset.py \
     --input data/processed/splits \
     --output data/processed/cleaned
   ```

### 2.2 文本 Embedding + 行情特征 + Reward

4. **ModernBERT 文本 embedding**  
   ```bash
   python scripts/generate_embeddings.py \
     --model answerdotai/modernbert-base \
     --input data/processed/cleaned \
     --output data/processed/embeddings \
     --batch-size 32 --normalize
   ```

5. **行情特征补齐**（可选但推荐）  
   ```bash
   python scripts/augment_with_market_data.py \
     --input data/processed/cleaned \
     --output data/processed/enriched \
     --company-list data/input/top_500_companies_list.xlsx
   ```

6. **Reward 构建（next-day return）**  
   ```bash
   python scripts/generate_reward.py \
     --input data/processed/enriched \
     --output data/processed/reward
   ```
   输出 `data/processed/reward/<KOL>/<split>.csv`，主要列包括：  
   `ticker, published_at, text, sentiment, confidence, reward_1d, done, embedding_* ...`。

### 2.3 Baseline + Ticker Embedding + Replay Buffer

7. **一键构建 Replay Buffer（推荐）**

```bash
python scripts/run_replay_pipeline.py \
  --reward-dir data/processed/reward \
  --vocab-path models/embedding/ticker_vocab.json \
  --embedding-path models/embedding/ticker_embedding.pt \
  --replay-dir data/replay_buffer
```

等价于依次运行：

1）`add_baseline_action.py`：在 reward CSV 上加 **基线 raw_score**

```text
baseline_raw_score = tanh(2 * sentiment * confidence)
```

2）`build_ticker_embedding.py`：从 reward CSV 中收集所有 `ticker`，构建：

- `models/embedding/ticker_vocab.json`  
- `models/embedding/ticker_embedding.pt`（随机初始化，后续可训练）

3）`build_replay_buffer.py`：核心逻辑提取到 `src/pipeline/replay_utils.py`：

- `annotate_positions(df)`：  
  - 对每个交易日，用 `PortfolioLayer` 在 baseline_raw_score 上回放一遍，且**保留上一日仓位**；  
  - 得到连续持仓 `baseline_weight` 以及上一期权重 `last_position`。  
- `build_states(df, ticker_embedder)`：  
  - 构造 `state = [ModernBERT embedding || ticker embedding || sentiment || confidence || last_position]`。  
- 按 ticker 时间序列构造 `next_states` 和 `dones`，最终写出：  
  `data/replay_buffer/<KOL>/train|val|test.pt`。

> **提示**：如果你之前已经有旧版 `data/replay_buffer`，需要重新跑一遍  
> `python scripts/run_replay_pipeline.py`，才能得到包含 `last_position` 的新版 buffer。


============================================================
3. 阶段二：离线强化学习训练（BC → IQL）
============================================================

训练入口：`train.py`。

```bash
python train.py \
  --kol Everything_Money \
  --replay-dir data/replay_buffer \
  --ticker-vocab models/embedding/ticker_vocab.json \
  --ticker-embedding models/embedding/ticker_embedding.pt \
  --output-dir outputs
```

### 3.1 训练流程概要

1. **加载 replay buffer**  
   - 从 `data/replay_buffer/<KOL>/train.pt` / `val.pt` 读取数据，`state_dim` 自动根据 buffer 推断；  
   - `state` 已含 `last_position`，代表上一期 baseline 仓位。

2. **行为克隆（BC）阶段**  
   - 网络：`ActorNetwork(state_dim)` 输出 `raw_score ∈ [-1, 1]`；  
   - 目标：给定 `state`，用 MSE 拟合 `baseline_raw_score`，模仿基线策略；  
   - 默认：`epochs=10`，`batch_size=256`，`lr=3e-4`。

3. **IQL 阶段**  
   - 使用 (state, action, reward, next_state, done) 做离线 RL；  
   - `CriticNetwork` 学 Q(s,a)，`ValueNetwork` 学 V(s)（expectile 回归）；  
   - Actor 在 Advantage 加权下更新，从行为策略偏向高优势动作；  
   - 默认：`steps=200k`，`batch_size=256`，`actor/critic/value lr=3e-4`，`expectile=0.7`，`temperature_beta=3.0`。

4. **验证集评估**  
   - 如存在 `val.pt`：在验证集上回放 Actor：  
     - 每日用 `PortfolioLayer` 在**上一日持仓基础上**按 `raw_score` 调整组合（资金 10000 美元）；  
     - 累积得到 `cumulative_return / sharpe / max_drawdown`。
   - 若希望同时优化“还原度”，可以通过 `--fidelity-lambda`（默认 0.1）引入额外奖励项：  
     - IQL 过程中会用 `reward_aug = reward_1d - λ * (actor(state) - baseline_action)^2`，  
     - λ 越大，策略越倾向保持与 KOL（baseline）一致；λ=0 则仅以收益为目标。

5. **训练输出目录**  
   - 每次运行自动创建 `outputs/<KOL>_<时间戳>/`：  
     - `logs/training.log`：训练日志；  
     - `checkpoints/actor.pt, critic.pt, value.pt, policy.pt`；  
     - `run_summary.json`：训练配置 + BC loss + 验证集指标（若有）。  
   - 可通过 `--output-dir` 调整根目录。

> 计算资源建议：单卡 8~12GB GPU 足够（batch 256、state 维度 ~800），CPU 8 核+、内存 16GB+ 较为舒适。


============================================================
4. 阶段三：测试集回放 / 决策分析 / 可视化
============================================================

本阶段主要基于 `src/evaluation/analyzer.py` 与若干脚本。

### 4.1 在 test.pt 上回放策略（指标 + 持仓轨迹）

```bash
python scripts/evaluate_run.py \
  --checkpoint outputs/<KOL>_<时间戳>/checkpoints/policy.pt \
  --buffer data/replay_buffer/<KOL>/test.pt \
  --output outputs/<KOL>_<时间戳>/metrics_test.json \
  --positions-output outputs/<KOL>_<时间戳>/positions_test.csv \
  --action-threshold 0.02
```

- `metrics_test.json`：`cumulative_return / sharpe / max_drawdown`；  
- `positions_test.csv`：逐样本（ticker 粒度）记录：  
  `raw_score, prev_weight, weight, weight_delta, allocation, action`（OPEN / CLOSE / INCREASE / DECREASE / HOLD），用于检查某只股票在整个测试期的持仓和调仓路径。

### 4.2 对比训练前后策略（同一测试集）

```bash
python scripts/compare_decisions.py \
  --trained outputs/<KOL>_<时间戳>/checkpoints/policy.pt \
  --buffer data/replay_buffer/<KOL>/test.pt \
  --output outputs/<KOL>_<时间戳>/decision_diff.csv \
  --metrics-output outputs/<KOL>_<时间戳>/metrics_compare.json \
  --action-threshold 0.02
```

- `decision_diff.csv`：合并“训练后策略 vs 基线/随机策略”的仓位和动作差异，包含：  
  `weight_trained / weight_baseline / action_trained / action_baseline / relative_action (MORE_LONG / MORE_SHORT / SIMILAR)`。

### 4.3 按视频粒度导出决策明细（含原文）

希望“一行 = 一个视频/一次发文”，并带原文、所有股票动作、组合构成和净值路径：

```bash
python scripts/export_signal_decisions.py \
  --checkpoint outputs/<KOL>_<时间戳>/checkpoints/policy.pt \
  --reward-csv data/processed/reward/<KOL>/test.csv \
  --vocab-path models/embedding/ticker_vocab.json \
  --embedding-path models/embedding/ticker_embedding.pt \
  --output outputs/<KOL>_<时间戳>/signal_decisions_test.csv
```

`signal_decisions_test.csv` 每行对应一个 `video_id`，包含：

- `date, video_id, text`：发文日期 / 视频 ID / 原文文本；  
- `tickers`：视频涉及的所有股票；  
- `baseline_actions, trained_actions`：形如 `AAPL:INCREASE;MSFT:HOLD`；  
- `portfolio_before_/after_baseline`：baseline 在该日调仓前/后的整体组合构成；  
- `portfolio_before_/after_trained`：训练后策略在该日调仓前/后组合构成；  
- `equity_baseline / equity_trained`：从测试起点到当前日期的净值；  
- `cum_return_baseline / cum_return_trained`：对应累计收益率。

### 4.4 净值曲线可视化 + 市场基准

```bash
python scripts/plot_equity_curve.py \
  --signal-decisions outputs/<KOL>_<时间戳>/signal_decisions_test.csv \
  --output-figure outputs/<KOL>_<时间戳>/equity_test_with_spy.png \
  --benchmark-ticker SPY \
  --benchmark-label "SPY (market)"
```

- 画出 baseline / trained 两条净值曲线；  
- 可选指定 `--benchmark-ticker`（如 `SPY` 或 `^GSPC`）叠加市场基准净值曲线，方便对比策略 vs 整体市场。


============================================================
5. 推理接口与对接建议
============================================================

线上/回测系统只需依赖两类产物：

1. **模型 checkpoint**
   - 从 `outputs/<KOL>_<时间戳>/checkpoints/` 中选择一个 `policy.pt`：
     ```text
     outputs/<KOL>_<时间戳>/checkpoints/policy.pt
     ```
   - 该文件封装了 Actor 的权重，可直接用于推理。

2. **推理接口**
   - `src/inference/agent.py` 中的 `RLKolAgent`：
     ```python
     from src.inference.agent import RLKolAgent

     agent = RLKolAgent(model_path="outputs/<KOL>_<时间戳>/checkpoints/policy.pt")
     out = agent.predict(kol_text, market_state)
     # out["target_position"], out["confidence"], out["timestamp"]
     ```
   - 回测/实盘侧只需根据 `target_position` 调整组合：
     ```python
     portfolio.adjust_to(out["target_position"])
     ```

> 回测逻辑（手续费、滑点、再平衡规则、风险约束等）由你们的回测/交易系统实现，本模块只负责给出“下一步仓位建议”。


============================================================
6. 小结
============================================================

- 本仓库提供了一个从 **KOL 文本 → Reward → Replay Buffer（含连续持仓）→ BC + IQL 训练 → 测试回放 → 决策分析/可视化** 的完整第一阶段 Pipeline；  
- 关键“数据生成和拼接”逻辑（baseline 持仓恢复、state 构造）集中在 `src/pipeline/replay_utils.py`，保证训练和分析使用同一套规则；  
- 评估与可视化通过 `src/evaluation` 与若干 `scripts/*` 封装成独立模块，便于按 KOL、按时间段快速复现实验或产出图表；  
- 第二阶段（更复杂的环境/成本建模、在线或半在线训练）可以在现有 `train.py` + `src/evaluation` 的基础上继续扩展。***
