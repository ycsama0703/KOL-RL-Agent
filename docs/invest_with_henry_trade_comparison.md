# Invest_with_Henry：Baseline vs Trained 交易对比（最新 run: 20251124_132448）

说明：训练使用双分支策略（有信号同向缩放基线、无信号衰减昨仓），这里挑选 10 个信号日，展示基线与训练后操作的差异。关注点：训练版常“减仓而非清仓”，并更集中到当期信号票；无信号衰减样本少，主要差异来自有信号分支。

## 10 个代表性交易日（当日净值：Baseline → Trained）

1) **2024-11-07（AAPL/TSLA 等）**（1.0398 → 1.0442，训练略高）  
   - 基线：AAPL:CLOSE；KR:OPEN；PAYC:INCREASE；TSLA/NVDA 大幅下调。  
   - 训练：AAPL:DECREASE（保留~1.45%，非清零）；KR:OPEN；PAYC:INCREASE；NVDA/ON/TSLA 减仓更显著。  
   - 效果：训练保留旧仓尾仓，同时更集中到新信号，净值略高。

2) **2024-11-08（AAPL/NVDA/TSLA）**（1.0602 → 1.0654，训练略高）  
   - 基线：NVDA:INCREASE；AAPL:DECREASE；TSLA:INCREASE。  
   - 训练：NVDA:INCREASE；AAPL:HOLD（不再减）；TSLA:INCREASE。  
   - 效果：对 AAPL 更温和（减仓 → 持有），NVDA/TSLA 更集中，净值抬升。

3) **2024-11-09（多票打开）**（1.0602 → 1.0654，训练略高）  
   - 基线：WMT/KHC/ABNB/MSFT/XOM/… OPEN；AAPL/TSLA/NVDA/… DECREASE；PYPL:DECREASE。  
   - 训练：大体相同，但 PYPL 从 DECREASE 变 HOLD，AAPL/TSLA/NVDA 仍下调。  
   - 效果：更谨慎减仓，少量旧仓保留，净值略高。

4) **2024-11-12（AAPL/MSFT/NVDA/TSLA 关仓）**（1.0508 → 1.0543，训练略高）  
   - 基线：NVDA/MSFT/TSLA/AAPL:CLOSE；其余 HOLD。  
   - 训练：NVDA/TSLA:DECREASE 而非清零；MSFT:HOLD；AAPL:OPEN（开回少量）。  
   - 效果：关闭信号时更倾向减仓/留尾仓或重新开少量，净值略高。

5) **2024-11-13（MSFT/NVDA/TSLA 开仓）**（1.0386 → 1.0399，训练微高）  
   - 基线：三票 OPEN，等权。  
   - 训练：三票 OPEN 且更集中（~27%/票），旧仓显著压缩。  
   - 效果：对新信号集中配置，旧仓快速稀释，净值微高。

6) **2024-11-15（AAPL/GS/NVDA/PYPL/SBUX/TSLA）**（1.0410 → 1.0413，训练微高）  
   - 基线：PYPL/SBUX INCREASE；AAPL:OPEN；NVDA/TSLA:DECREASE；GS:HOLD。  
   - 训练：PYPL:OPEN（比基线轻）；TSLA:DECREASE 更大；AAPL 仅留极小；NVDA/TSLA 仍较高。  
   - 效果：更保守增加旧仓，强化对新信号的集中，净值微高。

7) **2024-11-18（GS/NVDA/TSLA）**（1.0601 → 1.0589，训练略低）  
   - 基线：GS:CLOSE；TSLA/NVDA INCREASE。  
   - 训练：GS:DECREASE（不清零），TSLA/NVDA 更集中。  
   - 效果：卖出信号变“减仓”，权重集中到 TSLA/NVDA，但当日净值略低。

8) **2024-11-20（GS/NVDA/TSLA）**（1.0476 → 1.0455，训练略低）  
   - 基线：NVDA/TSLA DECREASE；GS:OPEN。  
   - 训练：NVDA/TSLA 同样减仓，但 GS 增加幅度更大。  
   - 效果：更集中权重到 GS，旧仓减得更狠，当日净值略低。

9) **2024-11-26（AAPL/GS/NVDA/TSLA）**（1.0416 → 1.0409，训练略低）  
   - 基线：TSLA:DECREASE；NVDA:INCREASE；AAPL:INCREASE；GS:CLOSE。  
   - 训练：TSLA:DECREASE；NVDA:DECREASE（方向相反于基线增）；AAPL:INCREASE；GS:DECREASE（留尾仓）。  
   - 效果：在同日信号下，训练对 NVDA 改为减仓，对 GS 留少量，净值略低。

10) **2024-11-27/28（NVDA/TSLA/GS/AAPL）**（11-27: 1.0623 → 1.0640；11-28: 1.0623 → 1.0640，训练高）  
    - 11-27 基线：NVDA:DECREASE；TSLA:INCREASE；GS:HOLD。训练：NVDA:INCREASE；TSLA:INCREASE；GS:CLOSE。  
    - 11-28 基线：NVDA:DECREASE；AAPL:HOLD；TGT:OPEN。训练：NVDA:DECREASE；AAPL:INCREASE；TGT:OPEN。  
    - 效果：训练版在连续两日对 NVDA/GS/AAPL 的处理更激进（增 NVDA 或清 GS，增 AAPL），相较基线的保守调整。

## 总结
- **基线**：严格按文本情感建仓/平仓，未提及的持仓继承（无衰减）。  
- **训练后**：建仓方向一致，但在卖出信号时倾向减仓而非清仓，对当期信号票更集中；无信号衰减样本较少，主要差异来自有信号分支的缩放。  
- **改进方向**：增加 carry 样本（无信号持仓）的衰减力度或比重，或收紧有信号分支的偏离，将策略差异集中在“退出/持有”上，以更突出退出学习效果。
