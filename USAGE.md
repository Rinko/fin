# Fin 策略业务用法速查（2026-08-23 起）

## 每日运行
```bash
python run.py prod --no-data      # 生产信号（ALIGN 四件套 + 场景化 quota）
python run.py backtest            # 组合回测（区间至 BASELINE_END，默认 2026-08-17）
python run.py signals             # 全量候选导出（轻量口径）
python run.py audit trades|entry|risk|magnitude   # 审计
```

## 现役配置
- 模型：ALIGN 四件套 —— 入场(close→close20 GPR 秩回归) / 风控排名(原始通道) /
  机会幅度(超额收益%,原始值+hurdle 消费) / 风险幅度(未来5日最差单日%,阈值-0.05)
- 规则：场景化 quota risk=0 / normal=2 / caution=3 / bottom=opp=5；
  ml_rank_floor 全关；BASE_TARGET_SIZE=0.04；初始资金口径 1M
- 同口径基线（vs 生产旧四件套）：**127.6% vs 111.6%，Sharpe 1.41 vs 1.27，Calmar 0.99 vs 0.94**

## 硬性约束（违反即静默出错）
1. **必须经统一入口**：直跑 `main.py` 被 RUN_LINE 哨兵拒绝；参数只认 `config.py` 盖章值
2. **generate_signals / 影子跑法：`--start` 必须前推 ≥400 自然日** 预留指标 warmup，
   否则评估循环不执行、零信号（AGENTS 注意事项#10）
3. **幅度模型单位契约**：阈值型消费者(④)要百分比原始值；缩放型消费者(③)可归一化
4. **阈值标定必须全量流式预测**，禁止用排序头部子集样本估分位

## 数据更新后的三步必查
1. `run.py audit entry` / `run.py audit risk` 全量审计通过
2. 影子 jaccard 对比上次记录（突降 = 重训隐性换头部警报）
3. 短窗回测冒烟能正常出信号

## 回滚预案
旧四件套：`external_data/models/archive_prod4_20260823/` 四个 pkl 复制回 root 即还原。
