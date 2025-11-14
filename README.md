# KOL-RL-Agent 模块文档（KOL 语料 → 交易策略智能体）

本模块负责将 KOL 文本语料 转换为 回测/实盘可执行的交易动作（目标仓位）。

外部只需传入：
- kol_text: 当日 KOL 文本
- market_state: 当日行情特征（你定义的 features）

智能体将输出：
- target_position（-1~1）
- confidence（可选）
- timestamp


============================================================
目录结构
============================================================

kol_rl_agent/
  data/
    raw/
    processed/
    market/
    replay_buffer/
  models/
    embedding/
    policy/
    checkpoints/
  src/
    preprocessing/
      text_cleaner.py
      chunker.py
      aligner.py
      feature_extractor.py
    embedding/
      encoder.py
    state/
      state_builder.py
    rl/
      buffer.py
      actor_critic.py
      iql.py
      cql.py
      trainer.py
    inference/
      agent.py
    utils/
      logger.py
  config/
    embedding_config.yaml
    rl_config.yaml
    env_config.yaml
  train.py
  infer.py
  README.md


============================================================
依赖安装
============================================================

pip install torch transformers sentence-transformers numpy pandas scikit-learn d3rlpy tqdm


============================================================
模块功能概述
============================================================

1. 文本预处理（preprocessing/）
负责文本清洗、句子切块、与市场数据对齐、提取情绪/强度/主题等特征。

2. 文本嵌入（embedding/encoder.py）
使用 SBERT / FinBERT 生成 KOL 文本 embedding，结合语义特征输出一个高维 KOL 特征向量。

3. 状态构建（state/state_builder.py）
将 市场特征 + KOL 语义特征 + 历史仓位 拼接为 RL 状态向量。

4. 离线强化学习（rl/）
使用 IQL 或 CQL 训练策略网络。
网络结构为：LSTM（处理历史） + Actor（输出仓位） + Critic（价值估计）。
训练流程包含：构建 replay buffer、行为克隆预训练、RL 训练、保存模型 checkpoint。

5. 推理模块（inference/agent.py）
外部系统只需要使用这个模块。
提供一个 predict(kol_text, market_state) 接口，输出目标仓位。


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
python train.py --config config/rl_config.yaml

流程：
1. 加载预处理后的 KOL 文本和行情
2. 文本 → embedding
3. 构建 RL 状态
4. 生成 replay buffer
5. 行为克隆预训练
6. IQL/CQL 强化学习
7. 保存策略模型


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
