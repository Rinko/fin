# 探索进度清单 (降大盘过重 / 2026-08)

基准: mkt_full (mkt_contri=1.0) 回测 192.5%, 最大回撤 -15.7%
现役: newfeat 211.8%, 夏普 1.60
规则: gate 有效 → 回测; 回测提升(收益↑且回撤不恶化) → 更新本基准
磁盘: 全部实验产物写 external_data/ (模型 pkl → external_data/models/, 审计 → external_data/audit/, 结果 → external_data/explore_night/)

## 阶段 ① 市场中性 target
- 状态: **FAIL** (入场+风控双 FAIL, 市场中性方向整体关闭)
- 入场回测: 总收益 191.4% vs mkt_full 192.5% (-1.0pp); 回撤 -13.0% vs -15.7% (+2.7pp 改善);
  夏普 1.42 (-0.11); 胜率 45.6%
- 风控回测 (入场 newfeat + 风控 mktneutral): 总收益 182.8% vs newfeat 211.8% (**-29pp**);
  回撤 -11.09% vs -11.15% (几乎无改善); 夏普 1.53 (-0.07); 胜率 49.4% (+1.2)
- **IC 悖论**: 入场 RankIC 0.148 (3倍) 未转化回测收益——残差排序可预测 ≠ 方向性可交易收益
- **风控中性化失败根因**: 聚焦个股特异风险后对系统性回撤不敏感 (回撤主要由市场系统性下跌驱动),
  同时错杀可获利持仓 (avg_pnl -241)
- 结论: 市场中性方向不可行, 现役 newfeat 211.8% 仍最优
- 模型留档: external_data/models/chip_accumulation_v6_newfeat_mktneutral.pkl,
  chip_risk_model_v1_newfeat_mktneutral.pkl (均非现役)

## 阶段 a 行业中性化 RS (rs_ind_20/60)
- 状态: **FAIL** (与 AGENTS #14 行业特征双 FAIL 一致)
- 结果: 单因子 IC=-0.0065, vs rs_20_z 相关仍 **0.899** (中性化几乎不去冗余), ΔRankIC=-0.0014, 重要性 5.7%
- 根因: A 股个股 20 日动量 ≈ 行业动量 (beta 主导), 行业中性化后剩余个股特异动量无独立预测力
- 判定: rankic_delta ✗ | low_corr(0.899>0.8) ✗ | not_dominant ✓ | single_ic ✗
- 结论: 行业中性化修正 rs 追行业问题不可行; 行业维度在 A 股难有独立 alpha (与 ind_inner_rank/ind_rank 一致)

## 阶段 d 行业上下文 MKT 特征 (sector_dispersion/up_ratio/momentum_spread)
- 状态: **FAIL** (回测 -53.1pp, 回撤恶化 5.5pp)
- 特征: sector_dispersion (行业日收益横截面std), sector_up_ratio (上涨行业占比),
  sector_momentum_spread (80th-20th 行业动量差) — 每日共享常数, 加到 MKT_FEATURES 10→7
- 回测: 总收益 158.7% vs newfeat 211.8% (-53pp); 夏普 1.30 (-0.30); 最大回撤 -16.7% (恶化5.5pp)
- 根因: 3 个 sector 特征作为共享常数吃掉 MKT 0.45 降权预算, 有效 MKT 信号被噪声稀释;
  sector_dispersion/up_ratio/spread 不贡献预测力 → 模型被迫学习忽略 → 损失容量
- 结论: 行业上下文作为 MKT 类特征同样不可行 (行业维度第 5 次失败)

## 阶段 c 行业领先/滞后传导 (ind_sync_20 + ind_beta_20)
- 状态: **FAIL** (行业维度第 4 次失败, 行业方向整体关闭)
- 结果: ind_sync_20 单因子 IC=0.0004 (≈0), 与 rs_20 最大相关仅 0.221 (独立信息但无预测力),
  ΔRankIC=-0.0015, 重要性 3.1%
- 结论: 个股-行业同步度/贝塔独立但无 alpha; A 股横截面收益不由行业传导关系驱动
- 判定: rankic_delta ✗ | low_corr ✓ (0.221) | not_dominant ✓ | single_ic ✗
- 行业维度累计失败: ind_inner_rank/ind_rank (AGENTS#14) + rs_ind (阶段a) + ind_sync/ind_beta (本阶段)

## 阶段 b 逆势/同步度 (bear_rs_20 + sync_20)
- 状态: **FAIL** (严格标准 ΔRankIC 需 ≥0.02, 实际 +0.0034)
- bear_rs_20: 完全无效 (模型零使用, 重要性 0%)
- sync_20: **7 个实验中唯一正单因子 IC 特征 (0.038, ICIR 0.25)**; 与现有特征低相关
  (max vs ema_turnover_vol_z 0.27); 但模型内增量仅 +0.0034 (0.0402→0.0436) 远低于 0.02 阈值
- 判定 (AGENTS#15 严格标准): rankic_delta_ge_0_02 ✗ (0.0034) | low_corr ✓ | not_dominant ✓ | single_ic ✓
- 启示: sync_20 的独立 alpha 信息已被现有筹码/换手特征覆盖 (相关 0.27 但模型冗余);
  唯一正 IC 特征可留作后续"低同步独立行情"主题的种子, 但当前架构无增量
- 教训: 本次 gate 曾用宽松判据(>0)误判 PASS, 已统一修正为 ≥0.02 (AGENTS#15)

## 探索总结 (2026-08-11 上午)
所有 7 个降大盘/新增特征实验全部 FAIL, 现役 newfeat 211.8% 仍最优:
1. 市场中性 target (入场): FAIL - RankIC 3倍但回测 -1pp
2. 市场中性 target (风控): FAIL - 收益 -29pp, 回撤无改善
3. rs_ind 行业中性化: FAIL - vs rs_20 相关仍 0.899
4. ind_sync/ind_beta 行业传导: FAIL - 独立但无预测力
5. bear_rs_20 逆势: FAIL - 模型零使用
6. sync_20 大盘同步度: FAIL - 单因子IC正但模型增量不足
7. (前置) mkt_full 关闭降权: FAIL - 回测 -19.3pp (AGENTS#11)
结论: 大盘特征降权 0.45 必要且正确 (AGENTS#11 再验证); 行业维度 4 连败 (AGENTS#14);
现有筹码特征已捕捉主要 alpha, 特征工程方向在此架构上已穷尽

## 已执行结论
(每阶段完成后追加: 结果数字 + 决策)
