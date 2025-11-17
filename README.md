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

2. **文本嵌入（`src/embedding/encoder.py` + `scripts/generate_embeddings.py`）**  
   使用 ModernBERT（或 SBERT/FinBERT）生成高维语义向量，可按 KOL/时间切片生成批量 embedding。

3. **状态构建（`src/state/state_builder.py`）**  
   将市场特征 + KOL embedding + 历史仓位拼接为 RL 状态。

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
python train.py --config config/rl_config.yaml
```

流程：
1. 加载 `data/processed/cleaned`（或包含 embeddings 的数据集）及对应行情特征  
2. 文本 → ModernBERT embedding  
3. 构建 RL 状态  
4. 生成 replay buffer  
5. 行为克隆预训练  
6. IQL/CQL 强化学习  
7. 保存策略模型（`models/checkpoints/<KOL>/policy.pt`）


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
