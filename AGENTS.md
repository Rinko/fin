# AGENTS.md — Fin 量化交易系统

## 项目概述
中国 A 股量化选股系统，基于 LightGBM 的机器学习选股 + 市场状态检测。

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
            co_compute.py (特征工程)
            is_market_ok.py (市场状态)
            local_data_cache.py (SQLite 缓存)
            stock_fetcher_bao.py (BaoStock API)
```

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
- 入场模型：29 特征（21 个股筹码 + 1 复合 + 7 大盘）— `chip_accumulation_v6_newfeat.pkl`
- 风控模型：32 特征（24 个股筹码 + 1 复合 + 7 大盘）— `chip_risk_model_v1_newfeat.pkl`
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
7. **feature_contri**：LightGBM 4.6.0 `feature_contri` 真实参数，`gain[i]=max(0,contri[i])*gain[i]`，大盘特征设 0.45
8. **市场环境对齐**：训练读 `market_context_cache.parquet`（`sync_market_context_file` 生成），回测实时算，每次数据同步后刷新 parquet
9. **大盘特征降权不可动**：MKT_FEATURES 的 `feature_contri=0.45` 是必要正则化，mkt_contri=1.0 实验回测 -19.3pp + 回撤恶化
10. **PyBroker Warmup**：`start_date = 交易期起点 − warmup 个交易日`（用 zzqz_df.xlsx 交易日历），区间 < warmup 日数则不产生交易
11. **Live Signal**：只在 `run_backtest` 结束后从末个交易日输出，区间短/股票池小可能无输出
12. **模型加载**：`backtest.py` 默认加载旧模型，需 `backtest.reload_models()` 切换现役 `chip_accumulation_v6_newfeat.pkl`

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
