# Fin 探索日志（逐日流水）

> 本文件由 AGENTS.md 迁出，仅追加不修改；当前态以 AGENTS/USAGE 为准。

## 探索归档状态
- **2026-08-14 前历史探索已归档**, 详见 `ARCHIVE_SUMMARY.md`
- **2026-08-21 当前生产默认** (`explore/signal_level_backtest_20260818` 分支已合并至 main):
  - 区间 2021-01-02 ~ 2026-08-17
  - total_return=**229.41%**, sharpe=**1.50**, sortino=**1.97**, max_drawdown=**-17.13%**, trade_count=**1353**
  - 关键规则：每场景 quota=5；floor 默认关闭（0.0，可选 0.005 待验证）
- 本轮已验证不可行：LambdaRank 各变体（单阶段/两阶段/截断/细粒度）、当前特征空间内的二阶头部质量模型
- 探索实体与报告 → `external_data/explore_night/signal_level_backtest_20260818/`
- 详细结论 → `external_data/explore_night/signal_level_backtest_20260818/exploration_summary_20260821.md`
- 2026-08-22 open-entry target 线索关闭：Gate1 同执行口径面板全维度落后（含2026年崩坏），代码开关保留于 `explore/open_entry_target_20260821`
- 2026-08-22 combo3(turn_vol_mom/stab/price_sync) 重试：审计最优(RankIC 0.105/单调✅)但 Gate2 无floor 191%<基线199%、DD更深，未采纳；开关 `CO_COMBO_FEATURES` 保留
- **2026-08-23 ALIGN 四件套定版**（生产现役候选, explore/align4_20260822）:
  - 组成：①入场 close-target GPR（regen 复核 RankIC 0.103）＋②风控排名 align（RankIC -0.47/七年稳）＋③机会幅度=超额收益 target、原始值+hurdle 消费（`OPPORT_HURDLE` 默认 0.02，修复 opport_mag_z 静默断线）＋④风险幅度=未来5日最差单日%（阈值 -0.05 兼容）
  - 规则：场景化 quota `BUY_QUOTA_RISK=0/NORMAL=2/CAUTION=3/BOTTOM=OPPORTUNITY=5`；floor 默认关；`BASE_TARGET_SIZE=0.04`；`BASELINE_END`/`INITIAL_CASH` env 可覆盖
  - 成绩（截至 2026-08-21 同口径 vs 生产旧四件套）：127.6% vs 111.6%，Sharpe 1.41 vs 1.27，Calmar 0.99 vs 0.94——双指标协议通过；check_trades 通过（Alpha 前5% 占比 98.6%、利润留存 66.9%）
  - 原则沉淀：**幅度模型单位由消费者定义**（阈值型要百分比、缩放型可归一化）；**阈值标定必须全量流式预测**，禁止头部子集样本；**模型×规则冗余时裁撤规则**
  - 幅度训练入口：`external_data/explore_night/signal_level_backtest_20260818/train_magnitude_align.py`（excess_tgt / risk_mag_tgt / hold_dd_tgt 三种 target 可切换；hold-dd 线已挂起）
- **2026-08-23 生产切换完成**：root 四 pkl 已替换为 ALIGN 组（原四件套归档于 `external_data/models/archive_prod4_20260823/`）；`*_newfeat` 审计符号链接恢复指向生产组合；运行时默认 = ALIGN + 场景化 quota
- **2026-08-24 市场特征与闸门体系全链审计**（explore/market-diag-20260824, 产物 `external_data/explore_night/market_diag_20260824/`）:
  - 大盘特征在排名模型中 ΔIC≈0、伪择时（水平漂移零预测力）；三因子历史配置的危害=共线碎片化+churn，非"掩盖"；E0 成对重训确认单主成分价值≤0.1pp
  - 三硬闸门全量规则终审：**净贡献 +20pp/+0.28Sharpe/-8.3ppDD**；money(预测波动)/cong(预测下跌)为主力，risk 标签无择时信息但经 Market_Risk_Clearance 卖出通道有效——"坏天气砍弱持仓"
  - 方法论教训：简化重放测规则得出方向相反结论（缺卖出通道）；测规则必须全量环境；删失路径需反事实重建；新增 GATE_* 四开关为闸门验收工具
  - 详细结论 → 本目录 FINAL_REPORT.md；配置漂移 112.8 vs 127.6 另案排查
- **2026-08-25 p_harm 统一概率引擎探索（自主会话，已归档）**: 阶段一 logit 校准 AUC 0.61（保序法失败弃用）；阶段二挑战者全线 CI 跨 0.5 触发止损线；全量回测三种消费模式（增强清仓/替代硬闸/连续仓位）全部净负 vs V0 → **结论：现阶段不需要单独大盘模型**，现行三闸为净正贡献在役冠军；GATE_*/PHARM_* 开关保留为验收工具；重启条件见 FINAL_REPORT.md 第四节


---

# AGENTS 迁出内容（2026-08-24 瘦身）

## 外接盘存储 (2TB)
- `external_data/` → 软链接 → `/Volumes/MAC外接/fin_data`（1.8Ti 可用）
- **所有大文件一律写外接盘**（审计 CSV、实验模型、回测结果），禁止写系统盘
- 审计 CSV (~15GB/个) 保存到 `external_data/audit/`，避免反复重训
- 生产核心文件（源码、现役模型 pkl）留在项目根

## 探索文件管理
- 探索脚本（gate_*.py, train_*_{variant}.py, analyze_*.py, run_*backtest*.py）→ `external_data/explore_night/scripts/`
- 运行方式：`PYTHONPATH=. python external_data/explore_night/scripts/xxx.py`（cwd=项目根）
- 确认采纳（gate PASS + 回测提升）后才并入项目根；FAIL 的留在外接盘
- 探索过程文件（日志/报告/gate JSON/回测对比）→ `external_data/explore_night/<stage>/`，留存避免重复计算
- 非现役/实验模型 pkl → `external_data/models/`；现役 pkl 留项目根
- 生产文件上的探索改动，未采纳时必须回滚
- **每个探索方向用独立 git 分支**（`explore/<name>`），不污染 main

## 架构
```
main.py → screen.py → backtest.py
                      ↓
            signal_engine.py (买入/卖出/仓位规则)
            co_compute.py (特征工程)
            is_market_ok.py (市场状态)
            local_data_cache.py (SQLite 缓存)
            stock_fetcher_bao.py (BaoStock API)

generate_signals.py      # 输出每日全量候选信号 CSV
signal_level_backtest.py # 固定本金 per trade 评估模型+规则
```

## 架构原则：训练与回测必须走 `co_compute.py` 公共模块

- `co_compute.py` 是唯一的特征工程、标准化与目标生成入口。所有训练脚本、回测脚本、探索脚本必须使用：
  - `co_compute.compute_individual_indicators` 生成个股特征
  - `co_compute.apply_standardization` 做横截面 Z-Score
  - `co_compute.FeatureConfig` 定义特征集合与市场特征口径
  - `co_compute.sync_market_context_file` 生成 `market_context_cache.parquet`
  - `co_compute.build_market_pca_table` 生成或加载大盘 PCA 特征
- **严禁**在训练/回测脚本中：
  - 手动读写项目根目录 `market_context_cache.parquet`
  - monkeypatch `co_compute.build_market_pca_table` 或任何公共函数
  - 内联重算筹码、标准化、GPR/risk target 等已由 `co_compute` 提供的逻辑
- 若需使用预计算 PC 表（如 G_pca1_z），通过以下方式之一注册，确保训练与回测使用同一张表：
  - 设置 `co_compute.FeatureConfig.PC_TABLE_PATH = '<path>.parquet'`
  - 调用 `co_compute.sync_market_context_file(..., pc_table_path='<path>.parquet')`
- 违反上述原则会导致训练与回测口径漂移，是模型/回测结果不可复现的主要根因。


## 模型评估原则

### 1. 用极简排名基准比较模型
- 比较不同入场模型时，必须先过**极简排名基准**：保留 ST/股价/流动性/`is_profit_ok` 过滤，剥离场景阈值、floor、大盘过滤、所有卖出规则；
- 每天按 `ml_rank` 取 top K，次日开盘买入，固定持有 20 天卖出；
- 通过该基准后，再进入带规则的信号级/组合回测。避免业务规则偏袒某个模型。

### 2. LambdaRank 已验证不可行
- 在极简基准下，LambdaRank（单阶段/两阶段/截断/20 档细粒度）均显著低于 L2 回归基线；
- 原因：GPR 排序目标与策略可实现收益存在结构性错位，且 LR 会制造“极少数 jackpot + 大量陷阱”的分数分布，不适合 quota 策略。

### 3. 生产默认规则
- `BUY_QUOTA_OVERRIDE=5`（每场景 5 只）；
- floor 规则保留但**默认关闭**（0.0），需显式设置 `ML_RANK_FLOOR_OPPORTUNITY/CAUTION=0.005` 启用；
- 依据：业务逻辑支撑不足，待分年 walk-forward 验证后再定去留。

- **2026-08-25 hfq 口径迁移定版**（内部全链路后复权，展示层 price_display 换算 qfq）:
  - 动因：废除 qfq 低价精度补丁规则（close>2/≥1.0 三处）；模型层实测新旧 ρ≥0.997、Top5 重合 90%——复权对筹码特征影响为噪声级
  - 事故与修复：现行幅度训练脚本 target 定义漂移（riskmag 被共享函数改为 σ 标准化、opport 丢失超额语义）→ 训练器本地恢复生产语义（raw 未来5日最差单日% / 个股-市场超额）并重训；**阈值不可跨模型传递第三例实证**（-0.05 触发率 7.3%→100%）
  - 校准常数固化：RISK_MAG_SELL_THRESHOLD=-0.055（对齐旧触发率）、OPPORT_PRED_OFFSET=-0.0063（中位对齐）、裁剪带 [0.4,1.8]
  - ⚠️ **8·22 锚点不可复现实锤**：同模型同口径当日重跑仅 108.6%（vs 127.6%）——8·23 全量数据重同步改写 market_context_cache 等环境所致，~19pp 列低优先级悬案；hfq 净效应 -13pp 且 DD 更浅
  - 同环境基线：qfq 对照 108.6%/1.318 vs hfq 现役 95.6%/1.199/DD-13.78%；回退开关 INFERENCE_ADJUST=qfq
  - 幅度训练入口变更为 `external_data/explore_night/magnitude_20260817/scripts/train_magnitude_for_g.py`（target 语义已在脚本内本地固化，不再依赖共享函数）；原 train_magnitude_align.py 未随迁移验证
- **2026-08-26 训练起点前推试点采纳**（TRAIN_START 2012-03-12 → 2010-01-01，入场模型）:
  - 动因：hfq 迁移后低价精度约束消失；数据体检显示缓存含 321 只退市股(5.8%)、2010 年前数据 1364 只——幸存者偏差可控
  - Gate1 真实对比（审计接线修复后）：Top1% 2.12% vs 2.00%、OOS IC_IR +11%、胜率 +3.1pp、年度仅 2025 微降 → 全指标胜出采纳 `chip_accumulation_v6_g_pca1_z_hfq_t2010.pkl`
  - ⚠️ 验证链事故修复：entry/risk 审计器 __main__ 硬编码符号链接旧模型+固定 model_data.csv，此前所有"入场/风控审计"实际审的是旧 qfq 模型；现支持 AUDIT_MODEL_PATH env + 自动配对同名 _data.csv 副档
  - 待办：风控排名模型同法前推试点未做；market_ml 已支持 TRAIN_START_DATE env
