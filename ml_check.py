import os
import joblib
import pandas as pd
import numpy as np
import logging
import warnings
from scipy.stats import spearmanr

# 彻底静默 Pandas 警告
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

class AuditConfig:
    split_date = '2020-01-01'
    sample_frac = 0.2
    chunk_size = 300000
    top_k_ratio = [0.01, 0.05, 0.1, 0.2]

def run_comprehensive_audit(data_path='model_data.csv', model_path='accumulation_model_final.pkl'):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\n" + "="*120)
    print(f"{'QUANT MODEL & FEATURE AUDIT SYSTEM (PRO-V7 FINAL - OPTIMIZED)':^120}")
    print("="*120)
    
    # 1. 加载模型
    if not os.path.exists(model_path):
        logging.error(f"模型文件 {model_path} 不存在"); return
    
    pkg = joblib.load(model_path)
    model, features = pkg['model'], pkg['features']
    logging.info(f"模型加载成功，训练特征数: {len(features)}")

    # 2. 阶段 1: 特征质量审计
    logging.info("执行阶段 1: 全局特征质量审计...")
    all_cols = pd.read_csv(data_path, nrows=1).columns.tolist()
    essential_cols = ['date', 'target', 'target_val', 'symbol']
    use_cols = list(set(features + [c for c in essential_cols if c in all_cols]))
    
    # 抽样读取进行质量评估
    df_sample = pd.read_csv(data_path, usecols=use_cols).sample(frac=AuditConfig.sample_frac).replace([np.inf, -np.inf], np.nan)
    
    report = []
    for col in features:
        data = df_sample[col].dropna()
        report.append({
            'Feature': col, 'Mean': data.mean(), 'Std': data.std(),
            'NaN%': (df_sample[col].isna().sum() / len(df_sample)) * 100,
            'Unique_Cnt': data.nunique(),
            'Min': data.min(), 'Max': data.max()
        })
    print("\n" + "-"*120 + f"\n{'1. 特征质量全局审计 (抽样分析)':^120}\n" + "-"*120)
    print(pd.DataFrame(report).to_string(index=False, formatters={'Mean':'{:.4f}'.format, 'Std':'{:.4f}'.format, 'NaN%':'{:.2f}%'.format}))

    # 3. 阶段 2: IS vs OOS 表现
    logging.info("执行阶段 2: 预测表现审计...")
    eval_df = pd.read_csv(data_path, usecols=use_cols).replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['target'])
    eval_df = eval_df.reset_index(drop=True) # 【优化点：确保索引唯一，防范后续 loc 索引复制风险】
    
    eval_df['pred'] = model.predict(eval_df[features])
    eval_df['is_oos'] = np.where(eval_df['date'] < AuditConfig.split_date, 'In-Sample', 'Out-of-Sample')

    def calc_daily_ic(group):
        # 【优化点：不仅防范 pred 无波动，同样防范 target 无波动的情况】
        if len(group) < 15 or group['pred'].std() < 1e-8 or group['target'].std() < 1e-8: 
            return pd.Series({'RankIC': np.nan})
        ic, _ = spearmanr(group['pred'], group['target'])
        return pd.Series({'RankIC': ic})

    daily_ic = eval_df.groupby(['is_oos', 'date']).apply(calc_daily_ic, include_groups=False).reset_index()
    summary = daily_ic.groupby('is_oos')['RankIC'].agg([
        ('RankIC_Mean', 'mean'), ('RankIC_Std', 'std'),
        ('IC_IR', lambda x: x.mean() / (x.std() + 1e-9)),
        ('IC_WinRate', lambda x: (x > 0).mean())
    ]).reset_index()

    print("\n" + "-"*120 + f"\n{'2. IS (训练期) vs OOS (测试期) 核心表现对比':^120}\n" + "-"*120)
    print(summary.to_string(index=False, formatters={'RankIC_Mean':'{:.4f}'.format, 'RankIC_Std':'{:.4f}'.format, 'IC_IR':'{:.4f}'.format, 'IC_WinRate':'{:.2%}'.format}))

    # 4. 阶段 3: 分箱单调性 (修正时序偏误)
    print("\n" + "-"*120 + f"\n{'3. OOS 分箱单调性审计 (Decile Analysis - Corrected)':^120}\n" + "-"*120)
    oos_eval = eval_df[eval_df['is_oos'] == 'Out-of-Sample'].copy()
    if not oos_eval.empty:
        # 每日截面进行分箱
        oos_eval['bucket'] = oos_eval.groupby('date')['pred'].transform(
            lambda x: pd.qcut(x + np.random.uniform(0, 1e-12, len(x)), 10, labels=False, duplicates='drop')
        )
        # 【优化点：先算每日的截面均值，再对日期求均值，消除时序异方差和极端日期偏误】
        daily_b_stats = oos_eval.groupby(['date', 'bucket'])[['target', 'target_val']].mean().reset_index()
        b_stats = daily_b_stats.groupby('bucket')[['target', 'target_val']].mean().T
        
        print(b_stats.to_string(header=True))
        is_mono = b_stats.loc['target'].is_monotonic_increasing
        print(f"\n📈 OOS 表现是否单调递增: {is_mono} " + ("✅" if is_mono else "⚠️"))
        
        # 【补充指标：多空对冲收益与夏普比率】
        ls_series = daily_b_stats[daily_b_stats['bucket'] == 9].set_index('date')['target_val'] - \
                    daily_b_stats[daily_b_stats['bucket'] == 0].set_index('date')['target_val']
        ls_sharpe = (ls_series.mean() / (ls_series.std() + 1e-9)) * np.sqrt(242) # 假设年度交易日为242
        print(f"📊 OOS 多空组合 (Decile 9 - Decile 0) 年化夏普比率: {ls_sharpe:.2f}")

    # 5. 阶段 4: 极值决策命中率
    print("\n" + "-"*120 + f"\n{'4. 极值决策命中率与绝对质量审计 (OOS 段)':^120}\n" + "-"*120)
    if not oos_eval.empty:
        print(f"{'选股档位':<12} | {'对等排名命中率':<15} | {'平均实际目标值':<15} | {'平均原始收益(Raw)':<15} | {'Alpha 提升':<10}")
        print("-" * 120)
        for p in AuditConfig.top_k_ratio:
            def calc_top_metrics(group):
                k = max(1, int(len(group) * p))
                top_pre_idx = group['pred'].nlargest(k).index
                top_act_idx = group['target'].nlargest(k).index
                hit_rate = len(set(top_pre_idx) & set(top_act_idx)) / k
                return pd.Series({
                    'hit': hit_rate, 
                    'rank': group.loc[top_pre_idx, 'target'].mean(),
                    'raw': group.loc[top_pre_idx, 'target_val'].mean()
                })
            res = oos_eval.groupby('date').apply(calc_top_metrics, include_groups=False).mean()
            print(f"Top {p:>4.1%}      | {res['hit']:>15.2%} | {res['rank']:>15.4f} | {res['raw']:>15.4f} | {res['hit']/p:>10.2f}x")

    # 6. 阶段 5: 大盘崩溃日审计
    print("\n" + "-"*120 + f"\n{'5. 大盘极端日表现审计 (Market Crash Audit)':^120}\n" + "-"*120)
    mkt_median = oos_eval.groupby('date')['target_val'].median()
    crash_days = mkt_median[mkt_median < mkt_median.quantile(0.1)].index
    if not crash_days.empty:
        crash_data = oos_eval[oos_eval['date'].isin(crash_days)]
        crash_top = crash_data.groupby('date').apply(
            lambda x: x.loc[x['pred'].nlargest(max(1, int(len(x)*0.01))).index, 'target_val'].mean() if len(x) > 0 else np.nan, 
            include_groups=False
        ).mean()
        print(f"大盘崩溃日样本: {len(crash_days)} 天")
        print(f"崩溃日 Top 1% 选股平均收益: {crash_top:.4f} (大盘中位收益: {mkt_median[crash_days].mean():.4f})")
        print("结论: " + ("选股在崩溃日展现超额韧性" if crash_top > mkt_median[crash_days].mean() else "选股无法对抗大盘崩溃"))

    # 7. 阶段 6: 特征贡献与换仓稳定性
    print("\n" + "-"*120 + f"\n{'6. 模型特征贡献度与换仓稳定性':^120}\n" + "-"*120)
    # 【优化点：多模型特征重要性提取兼容性改造】
    if hasattr(model, 'feature_importances_'):
        feat_imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        print(feat_imp.to_string(index=False))
    elif hasattr(model, 'coef_'):
        feat_imp = pd.DataFrame({'Feature': features, 'Coef': model.coef_}).sort_values('Coef', key=abs, ascending=False)
        print(feat_imp.to_string(index=False))
    else:
        print("提示: 当前模型不支持直接提取特征重要性系数。")

    if 'symbol' in oos_eval.columns:
        daily_top = oos_eval.sort_values(['date', 'pred'], ascending=[True, False]).groupby('date')['symbol'].apply(lambda x: set(x.head(20)), include_groups=False)
        turnovers = []
        prev = None
        for curr in daily_top:
            if prev is not None: 
                # 【优化点：使用动态的持仓分母，防止不足 20 只股票时分母计算偏误】
                denom = min(len(prev), len(curr))
                turnovers.append(1 - len(prev & curr) / denom if denom > 0 else 0)
            prev = curr
        print(f"\nOOS 每日平均换仓率 (Top 20): {np.mean(turnovers):.2%}")

    print("\n" + "-"*120 + f"\n{'7. 特征计算逻辑一致性自检 (Cross-Check)':^120}\n" + "-"*120)

    # 1. 检查残差特征是否真正实现了“去偏”
    # ema_bias 和 res_bias 的相关性应该接近 0，如果很高，说明残差化失败
    res_corr = eval_df[['ema_bias_norm_z', 'res_bias_norm_z']].corr().iloc[0,1]
    print(f"Bias趋势项与残差项相关性: {res_corr:.4f} (预期 < 0.2, 实测: {'✅' if abs(res_corr) < 0.2 else '⚠️'})")

    # 2. 检查筹码分布的合理性
    # 理论上，价格离最高筹码峰(high90)越远，获利盘(profit)应该越小
    cost_profit_corr = eval_df['dist_to_high90_z'].corr(eval_df['ema_profit_z'])
    print(f"价格位置与获利盘负相关性: {cost_profit_corr:.4f} (预期 > 0, 实测: {'✅' if cost_profit_corr > 0 else '⚠️'})")

    # 3. 检查大盘因子是否覆盖
    print(f"大盘因子覆盖天数: {eval_df['date'].nunique()} / 预期交易日")

    """
    审计信号时效性：预测值对未来 N 天收益的相关性
    """
    print("\n" + "-"*120 + f"\n{'8. Alpha 衰减审计 (IC Decay Analysis)':^120}\n" + "-"*120)
    # 假设我们需要评估对未来 1-5 天的预测力
    # 注意：这需要数据中有未来几天的收益率，或者在脚本中通过 shift 处理
    decay_results = []
    for lag in range(1, 6):
        # 简化版：计算预测值与 target (T+1) 的滞后相关性
        # 实战中建议直接用 T+n 的真实 return 计算
        ic = eval_df.groupby('date').apply(
            lambda x: spearmanr(x['pred'], x['target'].shift(-lag+1))[0] if len(x)>20 else np.nan
        ).mean()
        decay_results.append({'Horizon': f'T+{lag}', 'RankIC': ic})
    
    print(pd.DataFrame(decay_results).to_string(index=False))
    
import os
import joblib
import pandas as pd
import numpy as np
import logging
from scipy.stats import spearmanr

def run_comprehensive_audit_v4(data_path='model_data.csv', model_path='chip_accumulation_v6.pkl'):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. 加载模型与数据
    if not os.path.exists(model_path):
        logging.error("模型文件不存在"); return
    pkg = joblib.load(model_path)
    model, features = pkg['model'], pkg['features']
    
    # 读取必要列
    essential_cols = ['date', 'symbol', 'close', 'amount', 'change_pct', 'target', 'target_val']
    use_cols = list(set(features + essential_cols))
    df = pd.read_csv(data_path, usecols=use_cols).replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['target'])
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. 生成预测值
    logging.info("正在生成模型预测值...")
    df['pred'] = model.predict(df[features])
    df['daily_ret'] = df['change_pct'] / 100 
    
    # 【核心修复】：在全局（按股票分组）进行 shift，而不是在 groupby('date') 内部
    logging.info("预计算信号衰减项...")
    df = df.sort_values(['symbol', 'date'])
    for gap in [5, 10, 20]:
        # Delay N 天的 IC，意味着今天的预测值要对齐 N 天后的 target_val
        df[f'target_val_lag_{gap}'] = df.groupby('symbol')['target_val'].shift(-gap)

    # 3. 筛选 OOS 段 (2020年至今)
    oos_df = df[df['date'] >= '2020-01-01'].copy()
    oos_df = oos_df.sort_values(['date', 'pred'], ascending=[True, False])

    # ==========================================================================================
    # A. 动态换仓率审计
    # ==========================================================================================
    print("\n" + "-"*120 + f"\n{'1. 动态换仓率与信号稳定性审计':^120}\n" + "-"*120)
    daily_top_list = oos_df.groupby('date')['symbol'].apply(lambda x: set(x.head(100)))
    turnovers = []
    prev_set = None
    for curr_set in daily_top_list:
        if prev_set is not None:
            turnovers.append(1 - len(prev_set & curr_set) / len(prev_set))
        prev_set = curr_set
    
    avg_turnover = np.mean(turnovers)
    print(f"OOS 每日平均换仓率 (Top 100): {avg_turnover:.2%}")

    # ==========================================================================================
    # B. Alpha 衰减审计 (修正报错后的逻辑)
    # ==========================================================================================
    print("\n" + "-"*120 + f"\n{'2. Alpha 衰减审计 (预测时效性)':^120}\n" + "-"*120)
    decay_report = []
    
    # Delay 0d
    ic_0 = oos_df.groupby('date').apply(lambda g: spearmanr(g['pred'], g['target_val'])[0] if len(g)>10 else np.nan, include_groups=False).mean()
    decay_report.append({'信号延迟': 'Delay 0d', 'RankIC': ic_0})
    
    # Delay N days
    for gap in [5, 10, 20]:
        col = f'target_val_lag_{gap}'
        ic_n = oos_df.groupby('date').apply(lambda g: spearmanr(g['pred'], g[col])[0] if g[col].notna().sum()>10 else np.nan, include_groups=False).mean()
        decay_report.append({'信号延迟': f'Delay {gap}d', 'RankIC': ic_n})
    
    print(pd.DataFrame(decay_report).to_string(index=False))

    # ==========================================================================================
    # C. 净值回撤审计 (20日持有期重叠模拟)
    # ==========================================================================================
    print("\n" + "-"*120 + f"\n{'3. 净值回撤审计 (20日持有期分仓模拟)':^120}\n" + "-"*120)
    
    # 建立矩阵提升计算速度
    pivot_daily_ret = oos_df.pivot(index='date', columns='symbol', values='daily_ret').fillna(0)
    signal_mask = oos_df.pivot(index='date', columns='symbol', values='pred').rank(axis=1, ascending=False) <= 20
    
    dates = pivot_daily_ret.index
    daily_strat_ret = []
    
    # 为了提速，将 mask 转换为 values
    mask_values = signal_mask.values
    ret_values = pivot_daily_ret.values
    
    for i in range(20, len(dates)):
        sub_rets = []
        for lag in range(20):
            # 找到那天选中的列索引
            idx = mask_values[i-lag]
            if idx.any():
                day_ret = ret_values[i, idx].mean()
                sub_rets.append(day_ret)
        daily_strat_ret.append(np.mean(sub_rets))
    
    strat_series = pd.Series(daily_strat_ret, index=dates[20:])
    net_strat_ret = strat_series - (avg_turnover * 2 * 0.0015) 
    
    cum_nav = (1 + net_strat_ret).cumprod()
    max_nav = cum_nav.cummax()
    dd = (cum_nav - max_nav) / max_nav
    
    print(f"OOS 最终累计收益率: {cum_nav.iloc[-1]-1:.2%}")
    print(f"OOS 最大回撤:       {dd.min():.2%}")
    print(f"年化夏普比率:       {(net_strat_ret.mean()*242)/(net_strat_ret.std()*np.sqrt(242)):.2f}")
    print(f"最长回撤天数:       {(dd < 0).astype(int).groupby((dd == 0).astype(int).cumsum()).sum().max()} 天")

    # ==========================================================================================
    # D. 盈亏持平点分析
    # ==========================================================================================
    print("\n" + "-"*120 + f"\n{'4. 盈亏持平点压力测试':^120}\n" + "-"*120)
    avg_m_ret = strat_series.mean()
    for bps in [5, 15, 30]:
        cost = avg_turnover * 2 * (bps / 10000)
        print(f"单边摩擦 {bps:>2} bps | 每日净损益: {avg_m_ret - cost:>8.4%} | 成本占比: {cost/avg_m_ret:>6.2%}")

    # ==========================================================================================
    # E. 特征泄露自检
    # ==========================================================================================
    print("\n" + "-"*120 + f"\n{'5. 特征泄露与容量审计':^120}\n" + "-"*120)
    leakage = []
    for f in features:
        c = oos_df[f].corr(oos_df['change_pct'])
        leakage.append({'Feature': f, 'Corr_Today': c})
    print(pd.DataFrame(leakage).sort_values('Corr_Today', ascending=False).head(5))

if __name__ == "__main__":
    # 执行审计
    run_comprehensive_audit(data_path='model_data.csv', model_path='chip_accumulation_v6.pkl')
    run_comprehensive_audit_v4()