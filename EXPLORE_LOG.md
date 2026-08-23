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

