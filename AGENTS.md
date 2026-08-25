# AGENTS.md — Fin 量化交易系统

> 📌 **业务操作速查 → [USAGE.md](USAGE.md)**（先读它） | 探索日志 → [EXPLORE_LOG.md](EXPLORE_LOG.md) | 更早归档 → ARCHIVE_SUMMARY.md

## 项目概述
中国 A 股量化选股系统。LightGBM 四模型架构（入场排名 + 风控排名 + 机会幅度 + 风险幅度）+ 五象限市场状态检测 + PyBroker 回测。

## 现役生产配置（2026-08-23 切换）

- **模型**：ALIGN 四件套（详见 [USAGE.md](USAGE.md)）
- **③ sizing 参数**：`OPPORT_HURDLE=0.02 / OPPORT_SIZING_COEFF=0.30 / clip[0.4,1.8]`（网格标定最优）
- **场景化 quota**：risk=0 / normal=2 / caution=3 / bottom=opp=5
- **floor**：全关；**仓位基准**：0.04；**初始资金口径**：1M
- 同口径成绩 vs 旧四件套：127.6% vs 111.6%，Sharpe 1.41 vs 1.27

## 统一入口

```text
run.py ── config.apply(line) ──┬─ prod/backtest → main.py(哨兵) → screen → backtest(PyBroker)
│                               │                   ├─ signal_engine / co_compute / is_market_ok
├─ signals → generate_signals   ├─ bench → simple_rank_benchmark（外接盘）
├─ audit → check_trades / entry / risk / magnitude / signal_level / exits
├─ daily  → 轻量每日信号（买入+卖出）
└─ train  → market_ml + train_magnitude_align（外接盘）
config.py = 唯一参数源（七业务线 profile；MANAGED_KEYS 全量盖章；其余模块禁止写 env）
```

## 常用命令
```bash
python run.py daily --holdings 持仓.csv   # 每日轻量信号（买入+卖出，分钟级）
python run.py prod --no-data              # 重路径生产校验（PyBroker 全链）
python run.py backtest                    # 组合回测（至 BASELINE_END）
python run.py signals                     # 全量候选导出
python run.py bench                       # 极简排名基准
python run.py audit entry|risk|magnitude|trades|exits|signal_level|scenario|challenger
python get_base_data.py --task all        # 全部数据同步
```

## 架构原则：co_compute 公共模块

- 所有训练/回测/探索脚本必须通过 `co_compute.compute_individual_indicators` + `apply_standardization` + `FeatureConfig` 生成特征
- 严禁手动读写 `market_context_cache.parquet`、monkeypatch 公共函数、内联重算已有逻辑
- 违反导致训练与推理口径漂移——是模型不可复现的主要根因

## 模型评估原则

1. **极简排名基准先行**：保留 ST/股价/流动性/is_profit_ok 过滤，剥离全部规则；每天按 ml_rank 取 top K，次日开盘买，持 20 天卖。通过后才进规则回测。
2. **LambdaRank 已验证不可行**：排序目标与策略可实现收益结构性错位。
3. **训练确定性**：LGBMRegressor 必须设 `deterministic=True, force_row_wise=True`；已验证双训逐位一致。

## 注意事项（硬性规范）

1. **PyBroker Logger Bug**：backtest.py 修补 PyBroker 源码拼写错误
2. **Numba JIT**：首次运行慢
3. **SQLite WAL**：并发读支持
4. **BaoStock 会话**：错误码自动重连
5. **特征双通道**：入场平滑版(use_smooth=True)，风控原始版(False)
6. **前瞻约束**：特征只用当日收盘，交易 buy_delay≥1
7. **feature_contri**：等权组合（contri=1.0），不单独降权大盘
8. **市场环境对齐**：每次数据同步后刷新 market_context_cache.parquet
9. **大盘不降权**：contri=1.0 等权
10. **PyBroker Warmup**：start = 交易期起点 − warmup 个交易日；区间 < warmup 则无交易
11. **Live Signal**：仅 run_backtest 结束后末日输出
12. **模型加载**：backtest.py 不做 import 期加载（哨兵置空）；由入口显式 reload_models() + load_magnitude_models()；直跑 main.py 被 RUN_LINE 哨兵拒绝
13. **复权口径**：默认前复权(qfq)。所有训练脚本的 adjust 参数必须为 'qfq'，与推理管线一致

## 关键文件

| 文件 | 用途 |
|---|---|
| backtest.py | PyBroker 回测框架 |
| signal_engine.py | 买入资格、卖出判断、目标仓位 |
| co_compute.py | 特征工程 + 截面标准化 |
| config.py | 唯一参数源（七业务线 profile） |
| run.py | 统一 CLI 入口 |
| generate_signals.py | 全量候选 CSV 导出 |
| exit_signal.py | 持仓离场信号工具 |
| check_invariants.py | 每日输出不变量监控 |
| market_ml.py | LightGBM 训练流水线 |

## 数据层

- SQLite `stock_data_cache/{symbol}.db` + 元数据 stock_data.db
- 基本面 financial_reports_all.csv
- 中证全指 zzqz_df.xlsx
- 大盘上下文 market_context_cache.parquet

## 存储

- 现役 pkl + 源码 → 项目根
- 审计 CSV / 归档模型 / 探索产物 → external_data/（软链接 2TB 外接盘）
- pkl 晋级时旧版归档至 external_data/models/snapshots/<日期>/
