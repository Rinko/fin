# Fin 项目 — 探索归档总结 (2026-08-14)

> 本文件是项目当前唯一的状态说明。新对话从 main 分支开始前必读。
> 所有探索历史已归档到外接盘 `external_data/archive_20260814/`，项目内不再保留探索残留。

## 一、当前项目状态

- **分支**: main (4d712d8)，干净生产基线
- **现役模型** (项目根, 4 个):
  - `chip_accumulation_v6.pkl` / `chip_risk_model_v1.pkl` — 旧版
  - `chip_accumulation_v6_newfeat.pkl` / `chip_risk_model_v1_newfeat.pkl` — 新版 (29/32 特征)
- **核心数据**: `zzqz_df.xlsx` (交易日历)、`market_context_cache.parquet` (大盘快照)、`financial_reports_all.csv` (基本面)
- **架构**: main.py → screen.py → backtest.py → co_compute.py / is_market_ok.py / market_ml.py

## 二、历史最佳收益目标 = 235.95%

- 值: **235.95% / 年化 25.26% / 回撤 -12.53% / Sharpe 1.56 / 交易 1646 笔**
- 出处: `explore/opport-magnitude` 分支 8/12, `bt_rules_baseline_20260812_223625`
- 配置: 四模型 (入场+风控+机会幅度+风险幅度) + buy_quota 自适应规则
- **⚠️ 可信度警告**: 该值在 **bug 态代码**下跑出 —— `backtest.py` 的大盘因子映射 bug
  导致 mkt 四因子**恒为 0.5** (无真实大盘信息), 且幅度模型列未注册。
  代码修复后同模型仅 **181.59%** (修复态+全规则+幅度关) 或 **119.49%** (幅度开)。
  因此 **235.95% 是 bug 下的虚高, 不可直接作为实盘预期**。作为历史最高分留档。

## 三、各探索分支结论速查 (已归档 tag)

| tag | 方向 | 结论 |
|---|---|---|
| archive/explore/sector-ctx | 行业上下文 (09c9b8f) | 引入 3 组合特征 turn_vol_*，从未 gate |
| archive/explore/bear-sync | 熊市同步 | 未定稿 |
| archive/explore/ind-sync-beta | 行业同步β | 未定稿 |
| archive/explore/market-neutral | 市场中性 | gate FAIL |
| archive/explore/mkt-full | 全量大盘 | 无独有提交 |
| archive/explore/risk-conditional | 条件风控 | 未定稿 |
| archive/explore/rs-ind | 行业相对强度 | 未定稿 |
| archive/explore/opport-magnitude | 幅度模型+规则 | **核心成果**: 幅度过滤(opport_mag>0)=-62pp 应禁用; 规则链+66pp; 修复态基线 181.59%; buy_quota +24pp |

## 四、关键结论 (供新探索参考, 勿重做)

### 模型特征 (静态基线)
- 入场 29 特征 = 21 个股筹码 (横截面 Z) + profit_bias_div_z + 7 大盘 (mkt_trend/vol/liq/position + 广度3)
- 风控 32 特征 = 24 个股 (含 stock_congestion/high_vol_interaction/vp_corr_decay) + 复合 + 7 大盘
- 入场用平滑特征 (use_smooth=True), 风控用原始 (False)
- mkt feature_contri=0.45 降权 (AGENTS.md 记录 9: 不可动, 1.0 实验 -19.3pp)
- **特征→业务含义对照**: 归档 `archive_20260814/explore_night/stagea_market_interaction/FEATURES_SUMMARY.md`

### 已 FAIL 的特征探索 (勿重做)
| 特征 | 结果 |
|---|---|
| beta_20/60, corr_20, downside_beta, beta_change | IC 无增量, 回测 -36pp |
| 显式交互特征 | 模型 IC -0.003 有害 |
| rs_20/rs_60 行业相对强度 | 与乖离/均价冗余 0.72 |
| ind_rank 行业分位 | IC 显著为负 -0.043 |
| turn_vol_* 组合特征 | 全史下 +13.7pp 有益 (walk-g), 保留 |
| 个股×大盘乘法交互 (combo_*) | 共线 0.83-0.88, FAIL |
| 分箱交互 + 残差化 | gate +0.0045 但回测 -44.6pp 劣化 |

### 滚动训练 (walk-forward)
- 近因 8 年窗口训练 → 2023-24 灾难段 (-164pp); 全史 (2012-2019) 更稳
- 静态单段 211.8%(bug) / 181.59%(修复) 均优于滚动; **滚动训练无净优势**
- mirror_static 下 5 fold 模型 md5 全同 → ~40pp 是回测分段税 (机制开销)

### 规则链 (opport-magnitude 结论)
- **幅度过滤 opport_mag>0 = -62pp, 应禁用** (DISABLE_OPPORT_MAG_FILTER=1)
- 规则链核心价值 = 质量控制 (最简 115% → 全规则 181.59%, +66pp)
- 修复态真实基线: **全规则+幅度关 = 181.59%**

## 五、归档清单 (external_data/archive_20260814/)

| 目录 | 内容 |
|---|---|
| results/ | 全部回测结果 (27+ 目录, 含 newfeat 基线、walkforward 系列、beta) |
| explore_night/ | 全部探索脚本 + stage 报告 (stagea_market_interaction 等) |
| models/ | 23 个非现役/探索模型 pkl (no_mkt/mktfull/sectorctx/beta/magnitude 等) |
| task_queue/ | 任务队列日志与审计 |
| reports/ | REPORT_20260812/13、EXPLORE_PROGRESS、log.log |
| audit_integration/ | 项目根 audit/ 目录归档 |

## 六、已知 bug 记录 (重开探索前必读)

1. **大盘因子映射 bug** (AGENTS.md 记录 13): backtest.py 若 `df_temp['date'].map(mkt_factors[col])` 而
   mkt_factors 未 `set_index('date')`, 则 mkt 四因子恒 0.5 → 静默失真 (235.95% 虚高的根源)。
   **正确修复**: `mkt_factors = mkt_factors.set_index('date')`。入口 `co_compute._ensure_date_indexed()`。
2. **幅度模型注册**: backtest.py 需 `register_columns` 含 opport_mag/risk_mag, 否则过滤形同虚设。
3. **feature_gate.py 的 `_z` 后缀假设**: 只适用于 BIZ 特征, combo 特征会漏检 (需 _resolve_col)。
4. **回测侧 fillna 兜底**: 旧版 `fillna(0.5)` 掩盖数据缺失, 新版暴露; 回测期仅影响 2026-04-21/22 两天。

## 七、恢复与下一步

- **恢复某分支**: `git checkout archive/explore/<name>` (tag 即分支状态)
- **新探索建议** (若继续):
  1. 先确认修复态基线 (181.59%) 可复现, 以此为对照
  2. 如需超越 235.95%: 注意那是 bug 态, 真实目标是超越 181.59% (修复态可信最高)
  3. 特征探索 gate 门槛 MIN_OOS_RANKIC=0.02 过高, 建议改 0.002 (与阶段A 一致) 并配合完整回测终审
