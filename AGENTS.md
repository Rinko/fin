# AGENTS.md — Fin 量化交易系统

## 项目概述
中国 A 股量化选股系统，基于 LightGBM 的机器学习选股 + 市场状态检测。

## 探索归档状态
- **2026-08-14 前历史探索已归档**, 详见 `ARCHIVE_SUMMARY.md`
- **2026-08-21 当前生产默认** (`explore/signal_level_backtest_20260818` 分支已合并至 main):
  - 区间 2021-01-02 ~ 2026-08-17
  - total_return=**229.41%**, sharpe=**1.50**, sortino=**1.97**, max_drawdown=**-17.13%**, trade_count=**1353**
  - 关键规则：每场景 quota=5，`ML_RANK_FLOOR_OPPORTUNITY=0.005`, `ML_RANK_FLOOR_CAUTION=0.005`
- 本轮已验证不可行：LambdaRank 各变体（单阶段/两阶段/截断/细粒度）、当前特征空间内的二阶头部质量模型
- 探索实体与报告 → `external_data/explore_night/signal_level_backtest_20260818/`
- 详细结论 → `external_data/explore_night/signal_level_backtest_20260818/exploration_summary_20260821.md`

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
- `ML_RANK_FLOOR_OPPORTUNITY=0.005`, `ML_RANK_FLOOR_CAUTION=0.005`；
- 其他场景 floor 默认 0.0。

## 常用命令
```bash
python main.py                                    # 每日完整流程
python get_base_data.py --task all                # 全部数据同步
python get_base_data.py --task daily --date DATE  # 回填某天
python ml_check.py                                # 模型评估
python check_trades.py                            # 交易复盘
```

## 关键文件
| 文件 | 用途 |
|---|---|
| `backtest.py` | 核心策略逻辑，PyBroker 回测框架 |
| `signal_engine.py` | 纯信号规则层：买入资格、卖出判断、目标仓位 |
| `generate_signals.py` | 每日全量候选信号 CSV 生成（无现金/无 quota） |
| `signal_level_backtest.py` | 固定本金 per trade 信号级回测 |
| `co_compute.py` | 筹码特征工程，横截面 Z-score 标准化 |
| `market_ml.py` | LightGBM 训练流水线 |
| `is_market_ok.py` | 四象限市场状态检测 |
| `local_data_cache.py` | SQLite 数据缓存 |
| `stock_fetcher_bao.py` | BaoStock API 封装 |

## 数据流
1. **数据源**：BaoStock API → SQLite 缓存（`stock_data_cache/`）
2. **特征**：筹码分布、市场广度、动量指标
3. **模型**：两个 LightGBM（入场 + 风控），pkl 格式
4. **状态检测**：四象限（底部/机会/谨慎/正常/风险）

## ML 模型特征
- 入场模型：23 特征（21 个股筹码 + 1 复合 + 1 大盘）— `chip_accumulation_v6_g_pca1_z.pkl`
- 风控模型：24 特征（23 个股筹码 + 1 复合 + 1 大盘）— `chip_risk_model_v1_g_pca1_z.pkl`
- 幅度模型：
  - 机会幅度（超额收益）— `chip_opport_magnitude_excess_for_g.pkl`
  - 风险幅度 — `chip_risk_magnitude_for_g.pkl`
- 大盘特征：G_pca1_z 单主成分 `mkt_macro_regime`（`market_pca_g_pca1_z.parquet`）
- Z-score 标准化：每日横截面，clip(-3, 3)
- 复合特征：`profit_bias_div_z` = 获利盘 z − 乖离 z（筹码锁定未透支）

## 模型审计
- **训练后必须审计**：任何模型训练完成后，立即运行对应审计脚本检查预测分布、IC、区分度
- `ml_check.py`（→ `audit/check_model_entry.py`）：入场模型审计（IC/top-k/特征重要性）
- `ml_check_sell.py`（→ `audit/check_model_risk.py`）：风控模型审计 + 入场/风控协同
- `audit/check_magnitude_model.py <pkl路径>`：**幅度模型审计**（MSE 压缩检测、日Z-score可用性）
- `audit/check_data.py` / `check_base_data.py`：数据健壮性审计
- `audit/check_screen.py` / `check_backtest.py`：海选漏斗审计
- `audit/check_trades.py` / `check_trades.py`：交易归因审计
- **审计发现 MSE 压缩（区分度低）时**：幅度模型预测值须做日 Z-score 标准化后才能用于阈值判断

### 审计脚本分类
| 类型 | 脚本 | 用途 |
|---|---|---|
| 模型审计 | `audit/check_model_entry.py` | 入场模型 IC/分箱/特征重要性/衰减 |
| 模型审计 | `audit/check_model_risk.py` | 风控模型 + 入场/风控协同 |
| 模型审计 | `audit/check_magnitude_model.py` | **幅度模型** MSE压缩/日Z-score/全量特征重要性 |
| 数据/策略审计 | `audit/check_data.py` | 数据健壮性 (NaN/Inf/clip/物理边界) |
| 数据/策略审计 | `audit/check_screen.py` | 海选漏斗 (6层穿透率/配额覆盖) |
| 数据/策略审计 | `audit/check_trades.py` | 交易归因 (卖飞/摩擦/场景矩阵) |

1. **PyBroker Logger Bug**：`backtest.py:40-49` 修补 PyBroker 源码拼写错误
2. **Numba JIT**：首次运行慢
3. **SQLite WAL**：数据缓存用 WAL 模式支持并发读
4. **BaoStock 会话**：错误码 10001001/10001002 时自动重连
5. **特征双通道**：入场用平滑版（`use_smooth=True`），风控用原始版
6. **前瞻约束**：`compute_individual_indicators` 特征只用当日收盘，交易设 `buy_delay>=1`
7. **feature_contri**：LightGBM 4.6.0 `feature_contri` 真实参数，`gain[i]=max(0,contri[i])*gain[i]`。经最新实验，采用等权组合（所有特征 contri=1.0）替代手动调参，不再对大盘特征单独降权
8. **市场环境对齐**：训练读 `market_context_cache.parquet`（`sync_market_context_file` 生成，支持 `pc_table_path` 注入预计算 PC），回测实时算，每次数据同步后刷新 parquet
9. **大盘特征不再强制降权**：历史 `mkt_contri=0.45` 规则已失效；当前默认等权（contri=1.0），让模型自主分配市场/个股特征权重
10. **PyBroker Warmup**：`start_date = 交易期起点 − warmup 个交易日`（用 zzqz_df.xlsx 交易日历），区间 < warmup 日数则不产生交易
11. **Live Signal**：只在 `run_backtest` 结束后从末个交易日输出，区间短/股票池小可能无输出
12. **模型加载**：`backtest.py` 默认加载旧模型，需 `backtest.reload_models()` 切换现役 `chip_accumulation_v6_g_pca1_z.pkl`，并通过 `backtest.load_magnitude_models()` 加载机会/风险幅度模型

## 缓存结构
- `stock_data_cache/stock_data.db`：元数据 + 复权因子
- `stock_data_cache/{symbol}.db`：单股 OHLCV（raw_ 前缀）
- `financial_reports_all.csv`：基本面缓存
- `zzqz_df.xlsx`：中证全指日线

## 输出文件
- `results/{timestamp}/`：回测结果
- `trades.xlsx` / `orders.xlsx`：交易记录
- `ultimate_trade_audit.xlsx`：入场/出场快照
- `global_strategy_audit.csv`：每日市场状态决策
