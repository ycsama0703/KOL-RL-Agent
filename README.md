# KOL-RL-Agent 模块文档（KOL 语料 → 交易策略智能体）

本模块负责将 KOL 文本语料转换为回测/实盘可执行的交易动作（目标仓位）。当前仓库已经完成数据管线搭建（收集 → 统一格式 → 切分 → 清洗）以及 ModernBERT embedding 生成脚本，为后续 RL 训练做好准备。

外部只需传入：
- `kol_text`: 当日 KOL 文本
- `market_state`: 当日行情特征（你定义的 features）

智能体将输出：
- `target_position`（-1~1）
- `confidence`（可选）
- `timestamp`


============================================================
当前进展
============================================================

1. **数据沉淀**  
   - `data/input/`：原始 CSV（TikTok、YouTube 等）+ `top_500_companies_list.xlsx`。  
   - `src/preprocessing/build_dataset.py`：聚合所有 CSV，生成统一格式 `data/processed/total/kol_text_with_sentiment.csv`（>100MB，已写入 `.gitignore`）。  
   - 选择 KOL：Everything Money、Invest with Henry、MarketBeat（样本行数与视频数量均居前）。

2. **切分与清洗**  
   - `scripts/split_by_video_time.py`：按 `video_id` 时间顺序划分 train/val/test，输出在 `data/processed/splits/<KOL>/train|val|test.csv`。  
   - `scripts/clean_dataset.py`：对分集数据进行文本清洗、公司名归一化、去噪/去重，生成 `data/processed/cleaned/<KOL>/<split>.csv`（当前训练入口数据）。

3. **Embedding 准备**  
   - `scripts/generate_embeddings.py`：基于 `answerdotai/modernbert-base` 批量生成文本 embedding，并将结果以 `.pt` 写入 `data/processed/embeddings/...`。运行示例：
     ```bash
     python scripts/generate_embeddings.py \
       --model answerdotai/modernbert-base \
       --input data/processed/cleaned \
       --output data/processed/embeddings \
       --batch-size 32 --normalize
     ```
     需要外网下载模型，当前环境未执行；脚本可在本地或服务器上直接运行。

4. **核心代码结构**  
   - `src/preprocessing` 已包含文本清洗/切块/特征提取。  
   - `src/embedding/encoder.py` 将逐步接入 ModernBERT。  
   - `src/state` / `src/rl` / `src/inference` / `train.py` / `infer.py` 构成策略训练与推理框架。


============================================================
目录结构（当前）
============================================================

```
data/
  input/                       # 原始语料（TikTok/YouTube/公司列表等）
  processed/
    total/                     # build_dataset.py 输出的全集（已忽略大文件）
    top_channels/              # 高频 KOL 拆分结果
    splits/<KOL>/<split>.csv   # train/val/test（按视频时间划分）
    cleaned/<KOL>/<split>.csv  # 清洗后的训练输入
    embeddings/                # generate_embeddings.py 生成的 .pt（待运行）
models/
  embedding/ policy/ checkpoints/（占位）
src/
  preprocessing/               # text_cleaner、chunker、build_dataset 等
  embedding/                   # encoder（待接入 ModernBERT）
  state/                       # 状态构建
  rl/                          # buffer/actor_critic/iql/cql/trainer
  inference/                   # agent/predict API
  utils/                       # logger 等
config/
  embedding_config.yaml  rl_config.yaml  env_config.yaml
scripts/
  split_top_channels.py
  split_by_video_time.py
  clean_dataset.py
  generate_embeddings.py
  generate_reward.py
  add_baseline_action.py
  build_ticker_embedding.py
  build_replay_buffer.py
  run_replay_pipeline.py
train.py
infer.py
README.md
```


============================================================
依赖安装
============================================================

pip install torch transformers sentence-transformers numpy pandas scikit-learn d3rlpy tqdm


============================================================
模块功能概述（回顾）
============================================================

1. **文本预处理（`src/preprocessing/`）**  
   清洗、句子切块、市场对齐、情绪/强度/主题特征抽取；`build_dataset.py` 负责统一格式化输入。

2. **文本嵌入与 Reward（`scripts/generate_embeddings.py` / `scripts/generate_reward.py`）**  
   使用 ModernBERT 生成语义向量，并基于 yfinance 自动计算 next-day return 作为 reward，得到含 `reward_1d/next_date/done` 的 CSV。

3. **状态构建 & Portfolio（`src/state/state_builder.py` + `src/state/ticker_embedding.py` + `src/portfolio/layer.py` + `scripts/build_ticker_embedding.py`）**  
   将 ModernBERT 文本 embedding、公司 learned embedding（`ticker_vocab.json` + `ticker_embedding.pt`）、市场特征拼接成 state，并由 portfolio layer 将 Actor raw_score 归一化为 10000 美元资金分配。

4. **Replay Buffer 构建（`scripts/build_replay_buffer.py` 或 `scripts/run_replay_pipeline.py`）**  
   结合 reward CSV、baseline 动作、ticker embedding，将数据写入 `data/replay_buffer/<KOL>/<split>.pt`，可通过单条命令 `python scripts/run_replay_pipeline.py` 自动完成 baseline → ticker vocab → buffer 的构建流程。


============================================================
数据构造 Pipeline（阶段说明）
============================================================

1. **原始语料 → 统一格式**  
   - `scripts/build_dataset.py --input data/input --output data/processed/total/kol_text_with_sentiment.csv`
   - 聚合所有 TikTok/YouTube CSV，生成统一字段（text、company、sentiment、confidence 等）。

2. **按 KOL/时间切分**  
   - `scripts/split_top_channels.py`（可选，挑选重点 KOL）  
   - `scripts/split_by_video_time.py --input data/processed/<KOL>.csv --output data/processed/splits/<KOL>/`
   - 按 `video_id` 时间顺序划分 train/val/test。

3. **清洗与 ModernBERT Embedding**  
   - `scripts/clean_dataset.py --input data/processed/splits --output data/processed/cleaned`
   - `scripts/generate_embeddings.py --input data/processed/cleaned --output data/processed/embeddings --model answerdotai/modernbert-base`

4. **Reward 构建**  
   - `scripts/generate_reward.py --input data/processed/enriched --output data/processed/reward`
   - 参数：默认使用 yfinance 下载 `next_day` 收盘价，生成 `reward_1d/next_date/done`。

5. **Baseline 动作 + Ticker Embedding + Replay Buffer**  
   - 执行 `python scripts/run_replay_pipeline.py`（等价于依次运行 `add_baseline_action.py → build_ticker_embedding.py → build_replay_buffer.py`）  
   - 输出：`models/embedding/ticker_vocab.json`、`models/embedding/ticker_embedding.pt`、`data/replay_buffer/<KOL>/<split>.pt`。

完成以上步骤后，Replay Buffer 即可供 `train.py`（BC + IQL/CQL + Portfolio Layer）直接使用。

4. **离线强化学习（`src/rl/`）**  
   IQL/CQL + LSTM Actor-Critic；流程涵盖 replay buffer、行为克隆预训练、RL 训练、checkpoint。

5. **推理模块（`src/inference/agent.py`）**  
   统一对外接口 `predict(kol_text, market_state)`，输出目标仓位。

6. **行情获取（`src/market/yfinance_client.py` & `scripts/augment_with_market_data.py`）**  
   - `src/market/yfinance_client.py` 基于 `yfinance` 下载指定股票区间的 OHLCV，再生成 `returns / volatility / turnover` 等特征，可直接按照 `(date, ticker)` 查表补齐 `market_state`。  
   - `scripts/augment_with_market_data.py` 会读取 `data/processed/cleaned/<KOL>/<split>.csv` 与对应的 ModernBERT `.pt`，通过 `data/input/top_500_companies_list.xlsx` 映射公司→Ticker，并抓取最近 5 个交易日收盘价，生成包含 `embedding_*` 与 `close_t-*` 列的增强版 CSV（输出至 `data/processed/enriched/...`）。


============================================================
回测框架如何接入（核心）
============================================================

外部系统只调用以下接口：

agent = RLKolAgent(model_path="models/checkpoints/kolA/")
action = agent.predict(kol_text, market_state)

然后回测框架执行：
portfolio.adjust_to(action["target_position"])

你这边不负责调仓、手续费、回测逻辑。


============================================================
训练入口（train.py）
============================================================

运行：
```
python train.py \
  --kol Everything_Money \
  --replay-dir data/replay_buffer \
  --ticker-vocab models/embedding/ticker_vocab.json \
  --ticker-embedding models/embedding/ticker_embedding.pt
```

流程（BC → IQL）：
1. 加载 `data/replay_buffer/<KOL>/train.pt/val.pt` 并构建 `state = [ModernBERT embedding || ticker embedding || sentiment || confidence]`。  
2. 行为克隆（BC）阶段：`epochs=10`，`batch_size=256`，`lr=3e-4`，使用 baseline 动作 `tanh(2 * sentiment * confidence)` 进行 MSE 监督。  
3. IQL 阶段：`steps=200k`，`batch_size=256`，`actor/critic/value lr=3e-4`，`expectile=0.7`，`temperature_beta=3.0`。Actor/critic/value 均为 MLP（512-512-256），Actor 输出经 `tanh` 映射为 raw_score。  
4. 训练结束后在验证集上使用 `PortfolioLayer`（支持多空、资金 10000 美元、`weight_i = raw_i / Σ|raw|`）进行收益回放，输出 cumulative return / Sharpe / max drawdown。  
5. 保存策略到 `models/checkpoints/<KOL>/policy.pt`，并同时保存 actor/critic/value 的独立权重。


============================================================
推理入口（infer.py）
============================================================

运行：
python infer.py --text "今天新能源可能大涨" --market market.json

输出：
target_position 与 confidence 值。


============================================================
需要交付给网站/回测团队的内容
============================================================

1. 模型 checkpoint
   models/checkpoints/<KOL_NAME>/policy.pt

2. 推理接口文件
   src/inference/agent.py

3. 输入字段说明
   - kol_text：KOL 当日文本
   - market_state：行情特征（returns、volatility 等）

4. 测试示例 demo


============================================================
一句话总结
============================================================

本模块是一个 “输入 KOL 文本 → 输出交易动作” 的 RL 智能体，外部回测系统只需调用 predict() 即可使用策略。
