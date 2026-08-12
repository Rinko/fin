import pandas as pd
import numpy as np
import os
import glob
import logging
from scipy.stats import spearmanr

def god_mode_auditor_v10():
    print("="*100)
    print(f"{' 👁️  God-Mode v10: 策略全维度逻辑与实战容量审计 ':^100}")
    print("="*100)

    # --- 1. 数据加载与对齐 ---
    all_results = glob.glob('results/*/ultimate_trade_audit.xlsx')
    if not all_results: raise FileNotFoundError("未找到归因表。")
    target_file = max(all_results, key=os.path.getmtime)
    
    print(f"[1/7] 正在加载数据源: {os.path.basename(target_file)}")
    df_t = pd.read_excel(target_file)
    df_d = pd.read_csv('debug_inference_results.csv', dtype={'symbol': str})

    # 基础格式化
    df_t['entry_date'] = pd.to_datetime(df_t['entry_date']).dt.normalize()
    df_t['exit_date'] = pd.to_datetime(df_t['exit_date']).dt.normalize()
    df_t['symbol'] = df_t['symbol'].astype(str).str.zfill(6)
    df_d['date'] = pd.to_datetime(df_d['date']).dt.normalize()
    df_d['symbol'] = df_d['symbol'].astype(str).str.zfill(6)

    print(">>> 正在还原底表中的未来收益与风险目标 (Target Reconstruction)...")
    df_d = df_d.sort_values(['symbol', 'date'])
    if not os.path.exists('global_strategy_audit.csv'):
        raise FileNotFoundError("缺失 global_strategy_audit.csv，无法执行维度 O/P/Q 审计")
    df_g = pd.read_csv('global_strategy_audit.csv')
    df_g['index'] = pd.to_datetime(df_g['index']).dt.normalize()
    # 提取关键列：日期、场景、市场状态
    mkt_context = df_g[['index', 'strat_primary_scenario', 'strat_is_market_ok']].copy()
    
    # 将场景信息合并到推理底板 df_d 中
    df_d = df_d.merge(mkt_context, left_on='date', right_on='index', how='left')
    
    def reconstruct_targets(group):
        close = group['close']
        daily_ret = close.pct_change(1)
        
        # 1. 还原 20日真实收益 (T+1 到 T+21)
        target_val = close.pct_change(20).shift(-21)
        
        # 2. 还原 GPR Target
        pos_rets = daily_ret.clip(lower=0)
        neg_rets = daily_ret.clip(upper=0).abs()
        f_pos_sum = pos_rets.rolling(20).sum().shift(-21)
        f_neg_sum = neg_rets.rolling(20).sum().shift(-21)
        gpr_target = f_pos_sum / (f_neg_sum + 0.0001)
        
        return pd.DataFrame({
            'target_val_real': target_val,
            'gpr_target_real': gpr_target
        }, index=group.index)

    # 向量化补全
    recon_df = df_d.groupby('symbol', group_keys=False).apply(reconstruct_targets)
    df_d = pd.concat([df_d, recon_df], axis=1)

    # 提取特征快照用于溯源
    feature_lookup = df_d[['date', 'symbol', 'ml_rank', 'risk_ml_rank', 'bias_20', 'profit_ratio', 'amount', 'close']].copy()

    # --- 2. 双塔评分与特征溯源 ---
    # 关联入场和离场时刻的快照
    df_t = df_t.merge(feature_lookup, left_on=['entry_date', 'symbol'], right_on=['date', 'symbol'], how='left')
    df_t.rename(columns={'ml_rank': 'entry_alpha_rank', 'risk_ml_rank': 'entry_risk_rank', 
                        'bias_20': 'entry_bias_val', 'profit_ratio': 'entry_profit_ratio', 'amount': 'entry_amount'}, inplace=True)
    df_t.drop(columns=['date'], inplace=True)
    
    df_t = df_t.merge(feature_lookup[['date', 'symbol', 'ml_rank', 'risk_ml_rank', 'close']], 
                      left_on=['exit_date', 'symbol'], right_on=['date', 'symbol'], how='left', suffixes=('', '_exit'))
    df_t.rename(columns={'ml_rank': 'exit_alpha_rank', 'risk_ml_rank': 'exit_risk_rank', 'close_exit': 'exit_close_price'}, inplace=True)
    df_t.drop(columns=['date'], inplace=True)

    # --- 3. 核心计算 (MAE/MFE/Decay/Sell-Fly) ---
    df_t['ret'] = df_t['return_pct'] / 100.0 if df_t['return_pct'].abs().mean() > 0.5 else df_t['return_pct']
    df_t['mae_pct'] = df_t['mae'] / df_t['entry']
    df_t['mfe_pct'] = df_t['mfe'] / df_t['entry']
    df_t['alpha_decay'] = df_t['exit_alpha_rank'] - df_t['entry_alpha_rank']
    df_t['risk_surge'] = df_t['entry_risk_rank'] - df_t['exit_risk_rank']

    # 【修正 KeyError】安全计算卖飞分析
    print(">>> 正在执行卖飞分析与价格矩阵运算...")
    price_matrix = df_d.pivot_table(index='date', columns='symbol', values='close')
    future_max_10d = price_matrix.rolling(window=10).max().shift(-10)
    
    def get_safe_future_max(row):
        sym, dt = row['symbol'], row['exit_date']
        if sym in future_max_10d.columns and dt in future_max_10d.index:
            return future_max_10d.at[dt, sym]
        return np.nan

    df_t['post_exit_max'] = df_t.apply(get_safe_future_max, axis=1)
    df_t['sell_fly_pct'] = (df_t['post_exit_max'] / (df_t['exit_close_price'] + 1e-9)) - 1.0
    # 价格异常处理：剔除涨幅超过 100% 的卖飞（通常是数据错误）
    df_t.loc[df_t['sell_fly_pct'] > 1.0, 'sell_fly_pct'] = np.nan
    df_t['is_sell_fly'] = df_t['sell_fly_pct'] > 0.10

    # --- 4. 生成报告 ---

    # --- 维度 A: 准入与执行 ---
    print("\n" + "-"*40 + " [维度 A: 准入质量与容量审计] " + "-"*40)
    df_t['is_quality_entry'] = (df_t['entry_alpha_rank'] <= 0.05) & (df_t['entry_risk_rank'] >= 0.05)
    print(f" - 优质入场占比 (Alpha顶尖+风险安全): {df_t['is_quality_entry'].mean()*100:.2f}%")
    avg_entry_amt = df_t['entry_amount'].mean() / 1e8
    print(f" - 入场标的平均成交额: {avg_entry_amt:.2f} 亿 (容量参考)")

    # --- 维度 B: 场景 x Bias 交叉期望 ---
    print("\n" + "-"*40 + " [维度 B: 场景 x Bias 交叉期望矩阵] " + "-"*40)
    df_t['bias_bin'] = pd.cut(df_t['entry_bias_val'], bins=[-np.inf, -0.05, 0, 0.05, np.inf], 
                              labels=['超跌(<-5%)', '回踩(-5~0)', '强势(0~5%)', '加速(>5%)'])
    pivot_bias = df_t.pivot_table(index='primary_scenario', columns='bias_bin', values='ret', aggfunc='mean', observed=False)
    print(pivot_bias.round(4))

    # --- 维度 C: 筹码获利盘分布 ---
    print("\n" + "-"*40 + " [维度 C: 大盘场景 x 获利盘比例 交叉期望矩阵] " + "-"*40)
    
    # 定义获利盘分组
    df_t['profit_group'] = pd.cut(df_t['entry_profit_ratio'], bins=[0, 0.2, 0.5, 0.8, 1.0], 
                                 labels=['冷门(<20%)', '平庸(20-50%)', '活跃(50-80%)', '疯狂(>80%)'])
    
    # 1. 收益率矩阵
    pivot_profit_ret = df_t.pivot_table(index='primary_scenario', 
                                       columns='profit_group', 
                                       values='ret', 
                                       aggfunc='mean', 
                                       observed=False).round(4)
    
    # 2. 交易笔数矩阵 (用于检查样本可靠性)
    pivot_profit_count = df_t.pivot_table(index='primary_scenario', 
                                        columns='profit_group', 
                                        values='ret', 
                                        aggfunc='count', 
                                        observed=False).fillna(0).astype(int)
    
    print("\n[平均收益率矩阵]:")
    print(pivot_profit_ret)
    skew = df_t['ret'].skew()
    print(f"收益分布偏度: {skew:.2f}")
    print("\n[交易样本量分布]:")
    print(pivot_profit_count)

    # 自动化诊断逻辑
    if 'risk' in pivot_profit_ret.index:
        risk_crazy_ret = pivot_profit_ret.loc['risk', '疯狂(>80%)']
        if risk_crazy_ret < -0.02:
            print(f"\n 💡 建议：在 RISK 场景下，'疯狂'组收益极差({risk_crazy_ret:.2%})，应禁止追高。")

    # --- 维度 D: 离场效能与卖飞分析 ---
    print("\n" + "-"*40 + " [维度 D: 离场效能与卖飞分析] " + "-"*40)
    reason_stats = df_t.groupby('sell_reason').agg({
        'alpha_decay': 'mean', 'risk_surge': 'mean', 'ret': 'mean', 'is_sell_fly': 'mean', 'sell_fly_pct': 'mean'
    }).round(4)
    print(reason_stats)

    # --- 维度 E: 费用与摩擦损耗 ---
    print("\n" + "-"*40 + " [维度 E: 费用与摩擦损耗审计] " + "-"*40)
    fee_per_trade = 0.0012 * 2 # 假设双边12bps
    df_t['gross_ret'] = df_t['ret']
    df_t['net_ret'] = df_t['ret'] - fee_per_trade
    fee_erosion = fee_per_trade / (df_t['ret'].abs().mean() + 1e-9)
    print(f" - 手续费对平均收益的侵蚀率: {fee_erosion:.2%}")
    print(f" - 扣费后单笔净胜率: {(df_t['net_ret'] > 0).mean()*100:.2f}%")

    # --- 维度 F: 双塔模型共线性审计 ---
    print("\n" + "-"*40 + " [维度 F: 双塔模型独立性审计] " + "-"*40)
    model_corr = df_t['entry_alpha_rank'].corr(df_t['entry_risk_rank'])
    print(f" - 买入模型与风险模型入场排名相关性: {model_corr:.4f}")
    if abs(model_corr) > 0.7:
        print(" ⚠️ 警告：两模型高度相关，风险模型可能只是 Alpha 的反向表达，缺乏独立防守价值。")

    # --- 维度 G: 极端交易画像 ---
    print("\n" + "-"*40 + " [维度 G: 暴利单 vs 大亏单画像] " + "-"*40)
    top_5 = df_t.nlargest(5, 'ret')[['symbol', 'entry_date', 'ret', 'entry_alpha_rank', 'entry_risk_rank', 'primary_scenario']]
    bot_5 = df_t.nsmallest(5, 'ret')[['symbol', 'entry_date', 'ret', 'entry_alpha_rank', 'entry_risk_rank', 'primary_scenario']]
    print("\n[Top 5 Profit Trades]:")
    print(top_5.to_string(index=False))
    print("\n[Bottom 5 Loss Trades]:")
    print(bot_5.to_string(index=False))

    print("\n" + "-"*40 + " [维度 J: 退出效率与排名滑坡审计] " + "-"*40)
    
    # 1. 利润留存率 (Profit Retention)
    # 逻辑：最终收益 / 持仓期最高收益。
    # 如果这个值很低（比如 < 30%），说明你总是从最高点坐电梯跌回来才卖。
    df_t['profit_retention'] = df_t['ret'] / (df_t['mfe_pct'] + 1e-9)
    print(f" - 平均利润留存率: {df_t[df_t['ret']>0]['profit_retention'].mean():.2%}")

    # 2. Alpha 滑坡敏感度
    # 统计：当 PnL 掉头向下时，Alpha 排名移动了多少？
    # 如果 PnL 掉了 5%，但 Alpha 排名只动了 1%，说明 Alpha 模型卖出钝化。
    df_t['rank_drift_per_loss'] = df_t['alpha_decay'] / (df_t['mfe_pct'] - df_t['ret'] + 1e-9)
    print(f" - 排名滑坡敏感度 (每1%回撤对应的排名位移): {df_t['rank_drift_per_loss'].mean():.4f}")

    # 3. 卖出延时诊断
    # 统计有多少比例的订单在 Alpha 掉出 10% 时其实是盈利的，但你留到了 30% 变亏损。
    too_late_exit = df_t[(df_t['alpha_decay'] > 0.20) & (df_t['ret'] < 0)]
    print(f" - 显著延迟卖出笔数 (Alpha腐烂>20%且亏损): {len(too_late_exit)} 笔")

    # 统计获利 > 5% 的订单中，Alpha 排名与后续收益的相关性
    winners = df_t[df_t['mfe_pct'] > 0.05].copy()
    corr = winners['exit_alpha_rank'].corr(winners['ret'])
    print(f"获利单中 Alpha 排名与最终收益相关性: {corr:.4f}")
    # 如果相关性很低（< 0.1），说明盈利后看 Alpha 排名卖出是完全无效的。

    # --- 维度 M: 准入阈值弹性审计 (修复版) ---
    print("\n" + "-"*40 + " [维度 M: 准入阈值弹性审计] " + "-"*40)
    # 分析：1%-3% 这部分被你“嫌弃”的样本，实际上表现如何？
    extra_pool = df_d[(df_d['ml_rank'] > 0.01) & (df_d['ml_rank'] <= 0.03)]
    
    if not extra_pool.empty:
        mean_ret = extra_pool['target_val_real'].mean()
        win_rate = (extra_pool['gpr_target_real'] > 1.0).mean()
        print(f" - 1%-3% 区间样本的平均 20 日收益: {mean_ret:.2%}")
        print(f" - 1%-3% 区间样本的胜率 (GPR>1.0): {win_rate:.2%}")
        
        if mean_ret > 0.01:
            print(" 💡 诊断：1%-3%区间仍有显著正收益。全改为0.01会丢失大量机会，建议维持弹性阈值。")
        else:
            print(" 💡 诊断：1%-3%区间收益平庸。收紧至0.01在逻辑上是成立的，指标下降可能是由于分散度不足。")

    # =====================================================================
    # --- 维度 O/P/Q: 阈值科学性决策支持审计 (基于全量底表) ---
    # =====================================================================
    print("\n" + "="*100)
    print(f"{' 📈 阈值科学性决策支持审计 (Threshold Decision Support) ':^100}")
    print("="*100)

    # 定义排名区间
    df_d['rank_group'] = pd.cut(df_d['ml_rank'], 
                                bins=[0, 0.01, 0.03, 0.05, 1.0], 
                                labels=['Top1%', '1-3%', '3-5%', '5%以下'])

    # 1. 维度 O: 场景 x 排名区间 收益矩阵
    pivot_rank_scene = df_d.pivot_table(index='strat_primary_scenario', 
                                       columns='rank_group', 
                                       values='target_val_real', 
                                       aggfunc='mean', 
                                       observed=False).round(4)
    print("\n[维度 O] 场景 x 排名区间 期望收益矩阵 (决策核心):")
    print(pivot_rank_scene)

    # 2. 维度 P: 标的可得性 (回答 Quota 能够被填满的概率)
    # 过滤掉风险极高的标的后，统计每日平均可买数量
    df_d['is_tradable'] = (df_d['risk_ml_rank'] > 0.01) 
    availability = df_d[df_d['is_tradable']].groupby(['date', 'rank_group'], observed=False).size().unstack().mean()
    print("\n[维度 P] 每日平均可选标的数量 (Risk > 0.01 过滤后):")
    print(availability.round(2))

    # 3. 维度 Q: 排名与风险抗性 (MAE 预测)
    # 使用还原的风险目标 risk_score_real (越负越险)
    if 'risk_score_real' in df_d.columns:
        mae_by_rank = df_d.groupby('rank_group', observed=False)['risk_score_real'].mean()
        print("\n[维度 Q] 排名区间 vs 预期最大跌幅 (Risk Score):")
        print(mae_by_rank.round(4))

    # --- 自动化阈值诊断 ---
    print("\n" + ">>> 自动生成的阈值优化建议 <<<")
    for scene in pivot_rank_scene.index:
        row = pivot_rank_scene.loc[scene]
        # 如果 1-3% 的收益仍然非常可观（>1.5%），则支持放宽
        if row['1-3%'] > 0.015:
            print(f" - [{scene:12}]: 1-3%组收益为 {row['1-3%']:.2%}, 建议放宽阈值至 0.03 以捕获更多利润。")
        elif row['Top1%'] > 0.01:
             print(f" - [{scene:12}]: 仅 Top 1% 有效, 建议收紧阈值至 0.01。")
        else:
             print(f" - [{scene:12}]: 该场景下全线低迷, 建议设为 0.005 或强制空仓。")

    # --- 最终诊断与建议 ---
    print("\n" + "🚀" * 15 + " God-Mode 策略诊断建议 " + "🚀" * 15)
    if df_t['is_quality_entry'].mean() < 0.05:
        print(" 💡 诊断：入场端风险限制过于苛刻（优质入场仅{:.1f}%），导致大量高收益 Alpha 被误杀。".format(df_t['is_quality_entry'].mean()*100))
        print("    建议：将入场时的风险准入阈值由 0.05 放宽至 0.01。")
    
    if df_t[df_t['sell_reason'] == 'Time_Capital_Efficiency']['sell_fly_pct'].mean() > 0.15:
        print(" 💡 诊断：‘时间效率止损’存在严重卖飞现象（平均卖飞{:.1f}%）。".format(df_t[df_t['sell_reason'] == 'Time_Capital_Efficiency']['sell_fly_pct'].mean()*100))
        print("    建议：将 bars_held 阈值从 10 天放宽至 15 天。")

    # 导出
    output_file = os.path.join(os.path.dirname(target_file), 'god_mode_audit_v10.xlsx')
    df_t.to_excel(output_file, index=False)
    print(f"\n[系统] 审计完成。详细报告已生成至: {output_file}")


# ==========================================================================================
# 审计维度 H: 买入资格漏斗分解 + 候选流动性画像 (策略执行层审计)
# 直击"模型 top 候选被 is_active 流动性门槛大量拦截"问题。
# 输入: debug_inference_results.csv (推理打分) + global_strategy_audit.csv (每日场景)
# ==========================================================================================
def funnel_and_liquidity_audit(
    debug_path='debug_inference_results.csv',
    audit_path='global_strategy_audit.csv',
):
    print("\n" + "="*100)
    print(f"{' 📊 买入资格漏斗分解 + 候选流动性画像 ':^100}")
    print("="*100)

    if not os.path.exists(debug_path):
        print(f"❌ 缺少 {debug_path}, 请先运行 backtest 生成推理文件。")
        return
    if not os.path.exists(audit_path):
        print(f"❌ 缺少 {audit_path}, 跳过。")
        return

    # 1. 场景映射 (audit)
    audit = pd.read_csv(audit_path, usecols=['index', 'strat_primary_scenario'])
    audit['dt'] = pd.to_datetime(audit['index'])
    scen_map = dict(zip(audit['dt'].dt.date, audit['strat_primary_scenario']))
    del audit

    # 2. 推理数据
    cols = ['date', 'symbol', 'ml_rank', 'bias_20', 'is_profit_ok',
            'amount_ma20', 'atr_ratio']
    df = pd.read_csv(debug_path, usecols=cols, dtype={'symbol': str})
    df['date'] = pd.to_datetime(df['date'])
    df['dt'] = df['date'].dt.date
    df['scenario'] = df['dt'].map(scen_map).fillna('normal')

    # 3. 场景相关参数 (与 backtest.py 保持一致)
    scenario_ml_thr = {'opportunity': 0.02, 'bottom': 0.03, 'normal': 0.01,
                       'caution': 0.01, 'risk': 0.01}
    df['ml_thr'] = df['scenario'].map(scenario_ml_thr)

    # 4. 每日流动性门槛 (横截面 30 分位 / 20 分位)
    df['liq_q30'] = df.groupby('date')['amount_ma20'].transform(lambda x: x.quantile(0.3))
    df['vol_q20'] = df.groupby('date')['atr_ratio'].transform(lambda x: x.quantile(0.2))

    # 5. bias_con 判定 (与 check_buy_eligibility_and_score 一致)
    def bias_ok(row):
        sc = row['scenario']
        if 'bottom' in sc:
            return row['bias_20'] < 0
        elif 'opportunity' in sc:
            return row['bias_20'] > -0.05
        elif 'normal' in sc:
            return row['bias_20'] > 0.05
        return True  # caution/risk 不设 bias 硬约束
    df['bias_ok'] = df.apply(bias_ok, axis=1)

    # 6. 漏斗各层
    n_all = len(df)
    l1 = df[df['ml_rank'] < df['ml_thr']]
    l2 = l1[l1['is_profit_ok']]
    l3 = l2[(l2['amount_ma20'] >= l2['liq_q30']) & (l2['atr_ratio'] >= l2['vol_q20'])]
    l4 = l3[l3['bias_ok']]

    def pct(x):
        return f"{len(x)} ({len(x)/n_all*100:.2f}%)"

    print(f"全市场: {n_all}")
    print(f"  层1 ml_rank<阈值: {pct(l1)}")
    print(f"  层2 +is_profit_ok: {pct(l2)}")
    print(f"  层3 +is_active(q30/q20): {pct(l3)}")
    print(f"  层4 +bias_con: {pct(l4)} (最终合格候选)")
    print(f"  is_active 层淘汰率: {(1-len(l3)/len(l2))*100:.1f}%" if len(l2) else "  is_active 层: N/A")

    # 7. 按场景分层的漏斗通过率
    print("\n" + "-"*90)
    print(f"{'按场景 is_active 通过率 (层2→层3)':^70}")
    for sc in ['opportunity', 'bottom', 'normal', 'caution', 'risk']:
        sub = l2[l2['scenario'] == sc]
        if len(sub) == 0:
            continue
        sub3 = l3[l3['scenario'] == sc]
        print(f"  {sc:12s}: 层2 {len(sub):6d} → 层3 {len(sub3):6d} | 通过率 {len(sub3)/len(sub)*100:5.1f}%")

    # 8. 候选流动性画像: 层2 vs 层3 vs 层4 的成交额分布
    print("\n" + "-"*90)
    print(f"{'候选流动性画像 (成交额中位数, 万元)':^70}")
    for name, sub in [('层2 (ml+profit_ok)', l2), ('层3 (过is_active)', l3),
                      ('层4 (最终合格)', l4)]:
        print(f"  {name:22s}: 中位数 {sub['amount_ma20'].median()/1e4:>12,.0f} | "
              f"均值 {sub['amount_ma20'].mean()/1e4:>12,.0f} | "
              f"P10 {sub['amount_ma20'].quantile(0.1)/1e4:>12,.0f}")

    # 9. 按月的 is_active 通过率时序 (检测模型流动性偏好漂移)
    print("\n" + "-"*90)
    print(f"{'逐月 is_active 通过率 (检测流动性偏好漂移)':^70}")
    l2['ym'] = l2['date'].dt.to_period('M')
    l3_ym = l3.set_index('date').index.to_period('M')
    l3_cnt = pd.Series(l3_ym).value_counts()
    monthly = []
    for ym, grp in l2.groupby('ym'):
        l3_c = int(l3_cnt.get(ym, 0))
        monthly.append({'月份': str(ym), '层2数': len(grp), '过is_active': l3_c,
                        '通过率': f"{l3_c/len(grp)*100:.1f}%" if len(grp) else "N/A"})
    mdf = pd.DataFrame(monthly)
    if not mdf.empty:
        print(mdf.to_string(index=False))


# ==========================================================================================
# 审计维度 I: 卖出后走势验证 (退出时机 vs 选股质量)
# 对已平仓交易按卖出原因分组, 统计卖出后 3/6/12/20 日走势。
# 判断"退出过早" (卖出后大涨) vs "模型选股失败" (卖出后继续跌)。
# 输入: ultimate_trade_audit.xlsx (交易归因) + debug_inference_results.csv (含收盘价)
# ==========================================================================================
def post_exit_audit(
    trades_path='results/walkforward_c/combined/ultimate_trade_audit.xlsx',
    debug_path='debug_inference_results.csv',
):
    print("\n" + "="*100)
    print(f"{' 🔍 卖出后走势验证 (退出时机 vs 选股质量) ':^100}")
    print("="*100)

    if not os.path.exists(trades_path):
        print(f"❌ 缺少 {trades_path}, 跳过。")
        return
    if not os.path.exists(debug_path):
        print(f"❌ 缺少 {debug_path}, 跳过。")
        return

    # 1. 加载交易
    trades = pd.read_excel(trades_path, dtype={'symbol': str})
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    required = ['symbol', 'exit_date', 'exit', 'sell_reason']
    missing = [c for c in required if c not in trades.columns]
    if missing:
        print(f"❌ trades 缺少列: {missing}")
        return

    # 2. 加载价格
    prices = pd.read_csv(debug_path, usecols=['date', 'symbol', 'close'], dtype={'symbol': str})
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values(['symbol', 'date'])

    # 3. 对每笔交易计算卖出后 N 日收益
    horizons = [3, 6, 12, 20]
    res_rows = []
    for _, tr in trades.iterrows():
        sym = str(tr['symbol'])
        sub = prices[prices['symbol'] == sym]
        if sub.empty:
            continue
        after = sub[sub['date'] > tr['exit_date']]
        if after.empty:
            continue
        row = {'symbol': sym, 'sell_reason': str(tr['sell_reason']),
               'exit_ret': tr['return_pct'] if 'return_pct' in trades.columns else np.nan}
        for n in horizons:
            if len(after) >= n:
                row[f'post_{n}d'] = after.iloc[n-1]['close'] / tr['exit'] - 1
            else:
                row[f'post_{n}d'] = np.nan
        res_rows.append(row)
    res = pd.DataFrame(res_rows)
    if res.empty:
        print("❌ 无匹配的卖出后价格数据。")
        return

    # 4. 按卖出原因汇总
    print(f"\n匹配交易数: {len(res)}")
    print("\n" + "-"*90)
    print(f"{'按卖出原因: 卖出后走势均值':^70}")
    for reason, grp in res.groupby('sell_reason'):
        if len(grp) < 3:
            continue
        line = f"  {reason:30s}: n={len(grp):4d} | "
        for n in horizons:
            v = grp[f'post_{n}d'].dropna()
            if len(v):
                line += f"{n}d={v.mean()*100:+5.2f}%({(v>0).mean()*100:3.0f}%) "
        print(line)

    # 5. 总体判断: 若主要卖因的卖出后收益为负, 说明模型选股失败而非退出过早
    print("\n" + "-"*90)
    main_reasons = res['sell_reason'].value_counts().head(3).index.tolist()
    for r in main_reasons:
        grp = res[res['sell_reason'] == r]
        v12 = grp['post_12d'].dropna()
        if len(v12):
            verdict = ("退出过早(应多持)" if v12.mean() > 0.02
                       else "模型选股失败(卖出后仍跌)" if v12.mean() < -0.02
                       else "中性(横盘)")
            print(f"  {r}: 卖出后 12d {v12.mean()*100:+.2f}% → {verdict}")


if __name__ == "__main__":
    god_mode_auditor_v10()
    funnel_and_liquidity_audit()
    post_exit_audit()