# Fin 量化选股系统 · 业务使用手册

> 面向日常使用者。技术规范见 `AGENTS.md`，历史探索见 `EXPLORE_LOG.md`。

## 一、系统做什么

每个交易日收盘后，对全市场约 5000 只 A 股打分排序；结合大盘状态（底部/机会/谨慎/正常/风险五象限）决定当日买入名额与标的；持仓后由风控规则（风险排名恶化、幅度阈值、效率退出等）自动卖出，平均持有约 5~6 个交易日。熔断（risk）持续 ≥3 个交易日时，仍被模型看好的亏损持仓将触发防守性退出（熔断持续性持仓上限）。

> 五象限语义（2026-08-26 归位）：**opportunity** = 右侧进攻 + MA60 下方的修复型机会（每日名额 5）；**caution** 仅代表真防守（高位动能枯竭/急跌拦截，名额 1）；**normal** 常态（2）；**bottom** 抄底（5）；**risk** 熔断空仓（0）。

四个模型分工：

| 模型 | 职责 | 一句话 |
|---|---|---|
| 入场排名 | 选谁进候选池 | 预测未来 20 日相对强弱排名 |
| 风控排名 | 持仓预警 | 预测短期危险度，恶化即退出 |
| 机会幅度 | 买多重 | 预测超额收益幅度，正超门槛加仓 |
| 风险幅度 | 何时必须跑 | 预测最差单日跌幅，破 -5% 清仓 |

## 二、命令速查（唯一入口 run.py）

```bash
python run.py daily --holdings 我的持仓.csv   # ① 每日轻量信号：输出当日 买入名单+持仓卖出建议（分钟级，推荐）
python run.py prod --no-data        # ② 重路径生产校验（PyBroker 全链，最终权威；较慢）
python run.py backtest              # ② 组合回测（区间至 BASELINE_END）
python run.py signals               # ③ 导出全量候选 CSV（研究用）
python run.py bench                 # ④ 极简排名基准（模型比较专用）
python run.py audit entry|risk|magnitude|trades|signal_level   # ⑤ 审计
python run.py audit exits --holdings 持仓.csv --start 2026-01-01      # ⑥ 持仓离场信号
```

数据同步（另一入口）：
```bash
python get_base_data.py --task all                 # 全量同步
python get_base_data.py --task daily --date DATE   # 补某天
```

## 三、每日操作流程

1. 收盘后执行数据同步；
2. `run.py prod --no-data` —— 输出当日候选与信号（results 目录 + 终端摘要）；
3. 核对信号数是否正常（近期常态：日均 0~5 只，risk 日为 0 属正常）；
4. 每周一次 `run.py audit trades` 复盘成交质量。

## 四、各脚本功能与注意事项

| 脚本 | 功能 | 注意事项 |
|---|---|---|
| `run.py prod/backtest` | 生产信号 / 组合回测 | **不可直跑 main.py**（哨兵会拒绝）；参数由 config.py 统一管理 |
| `generate_signals`（经 signals） | 全量候选导出 | `--start` 需按 config 统一前推（WARMUP_TRADING_DAYS=600 交易日，日历精确回推），
  不足会因 EMA 路径依赖导致同日分数漂移甚至头部翻转；过短直接**零信号** |
| `simple_rank_benchmark`（经 bench） | 无规则纯排名对照 | 用于模型横向比较，成绩不代表策略收益 |
| `get_base_data.py` | BaoStock 数据同步 |
| `run.py daily` | 轻量每日信号：全市场打分→场景配额选买→持仓注入卖出规则链 | 五条卖出规则全量生效（含 Risk_Mag_Exit）；大盘场景默认 normal，可用 --scenario 覆盖 |
| `run.py audit exits` | 输入持仓 CSV(symbol,entry_date,entry_price[,shares]) 输出今日离场建议 | 大盘清仓类退出默认按 normal 场景，可用 --scenario 覆盖；Risk_Mag_Exit 当前降级跳过 | 更新后务必走「六、数据更新三查」 |
| `config.py` | 所有参数唯一定义处 | 修改默认值在这里；业务线差异看 PROFILES |

## 五、常见问题（FAQ）

**Q1 直跑 `python main.py` 报错"请通过统一入口"？**
设计如此。所有参数由 config.py 按业务线统一盖章，防止口径漂移。

**Q2 跑完没有任何信号？**
按顺序排查：① `--start` 是否满足 600 交易日统一预热；② 当期是否处于 risk/大盘关闭状态（属正常风控）；③ `run.py audit entry` 看模型审计是否异常。

**Q3 想临时调参（如回测区间）？**
命令行参数（--days/--start 等）直接传；环境类参数改 `config.py` 对应 PROFILES 或 DEFAULTS。

**Q4 模型文件在哪、怎么回滚？**
现役四件套在项目根目录；上一版在 `external_data/models/archive_prod4_20260823/`，
复制回根目录即完成回滚。

## 六、现役版本与成绩（截至 2026-08-21 同口径）

- 版本：ALIGN 四件套（2026-08-23 切换）
- 规则：quota risk=0 / normal=2 / caution=3 / bottom=opp=5；floor 关；仓位基准 4%
- 成绩 vs 旧版：**127.6% vs 111.6%｜Sharpe 1.41 vs 1.27｜Calmar 0.99 vs 0.94**
- 旧版回滚：见 Q4

## 七、数据更新后的三步必查

1. `run.py audit entry` 与 `run.py audit risk` 全量审计通过（无截断告警）；
2. 影子对比 jaccard 未突降（对比工具：`external_data/shadow/shadow_diff.py`）；
3. 短窗冒烟能正常出信号。
