# audit/check_trades.py
# 交易归因与临场审计 (原 check_trades.py)
# 依赖: results/*/ultimate_trade_audit.xlsx + global_strategy_audit.csv
#       debug_inference_results.csv 可选 (DEBUG_INFERENCE=1), 缺失时自动降级部分维度
import glob
import logging
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DEFAULT_DEBUG = 'debug_inference_results.csv'
DEFAULT_AUDIT = 'global_strategy_audit.csv'


def find_latest_trades_file(roots='results'):
    files = glob.glob(f'{roots}/*/ultimate_trade_audit.xlsx')
    # combined 目录(多折滚动合并) 视为更完整的归因表
    combined = [f for f in files if '/combined/' in f]
    pool = combined if combined else files
    if not pool:
        raise FileNotFoundError("未找到 results/*/ultimate_trade_audit.xlsx，请先回测。")
    return max(pool, key=os.path.getmtime)


def _norm_symbol(series):
    return series.astype(str).str.extract(r'(\d{6})', expand=False).fillna(
        series.astype(str).str.replace('.SH', '').str.replace('.SZ', '').str.zfill(6))


# =============================================================================
# 主审计: God-Mode 全维度交易归因
# =============================================================================
def god_mode_auditor_v10(trades_path=None, debug_path=DEFAULT_DEBUG):
    print("=" * 100)
    print(f"{' 👁️  God-Mode v11: 策略全维度实战审计 ':^100}")
    print("=" * 100)

    trades_file = trades_path or find_latest_trades_file()
    print(f"[1/7] 归因表: {trades_file}")
    df_t = pd.read_excel(trades_file)
    df_t['symbol'] = _norm_symbol(df_t['symbol'])
    df_t['entry_date'] = pd.to_datetime(df_t['entry_date']).dt.normalize()
    df_t['exit_date'] = pd.to_datetime(df_t['exit_date']).dt.normalize()

    has_debug = os.path.exists(debug_path)
    if has_debug:
        print(">>> 使用 debug_inference_results.csv 还原未来收益、特征快照与价格矩阵...")
        df_d = pd.read_csv(debug_path, dtype={'symbol': str})
        df_d['date'] = pd.to_datetime(df_d['date']).dt.normalize()
        df_d['symbol'] = _norm_symbol(df_d['symbol'])
        df_d = df_d.sort_values(['symbol', 'date'])

        # 还原 20 日持有期 (T+1~T+21) 收益 / GPR / 前向最大回撤
        def reconstruct(group):
            close = group['close']
            daily_ret = close.pct_change(1)
            target_val = close.pct_change(20).shift(-21)
            pos = daily_ret.clip(lower=0); neg = daily_ret.clip(upper=0).abs()
            gpr = pos.rolling(20).sum().shift(-21) / (neg.rolling(20).sum().shift(-21) + 0.0001)
            # 前向 20 日最大回撤 (代理风险目标, 原 risk_score_real 从未计算 → 等效修复)
            start = close.shift(-1)
            fwd_min = close.shift(-2).rolling(20, min_periods=1).min().shift(-19)
            mdd20 = fwd_min / start - 1.0
            return pd.DataFrame({
                'target_val_real': target_val, 'gpr_target_real': gpr, 'fwd_mdd_20d': mdd20
            }, index=group.index)

        recon_df = df_d.groupby('symbol', group_keys=False).apply(reconstruct)
        df_d = pd.concat([df_d, recon_df], axis=1)
        feature_lookup = df_d[['date', 'symbol', 'ml_rank', 'risk_ml_rank', 'bias_20',
                               'profit_ratio', 'amount', 'close']].copy()

        df_t = df_t.merge(feature_lookup, left_on=['entry_date', 'symbol'],
                          right_on=['date', 'symbol'], how='left')
        df_t.rename(columns={'ml_rank': 'entry_alpha_rank', 'risk_ml_rank': 'entry_risk_rank',
                             'bias_20': 'entry_bias_val', 'profit_ratio': 'entry_profit_ratio',
                             'amount': 'entry_amount'}, inplace=True)
        df_t.drop(columns=['date'], inplace=True)
        df_t = df_t.merge(feature_lookup[['date', 'symbol', 'ml_rank', 'risk_ml_rank', 'close']],
                          left_on=['exit_date', 'symbol'], right_on=['date', 'symbol'],
                          how='left', suffixes=('', '_exit'))
        df_t.rename(columns={'ml_rank': 'exit_alpha_rank', 'risk_ml_rank': 'exit_risk_rank',
                             'close_exit': 'exit_close_price'}, inplace=True)
        df_t.drop(columns=['date'], inplace=True)
    else:
        print("⚠️ 无 debug_inference_results.csv (回测需 DEBUG_INFERENCE=1)，以下维度降级/跳过:")
        print("   - 入场/离场双塔排名 (feature snapshot)     → 用归因表自带 entry_ml_rank/exit_risk_ml_rank")
        print("   - 维度 C/D/F/J 中依赖排名差/卖飞的子项")
        print("   - 维度 M/O/P/Q (阈值科学性决策支持)")
        df_t['entry_alpha_rank'] = df_t.get('entry_ml_rank', np.nan)
        df_t['entry_risk_rank'] = np.nan
        df_t['exit_alpha_rank'] = np.nan
        df_t['exit_risk_rank'] = df_t.get('exit_risk_ml_rank', np.nan)
        df_t['entry_bias_val'] = df_t.get('entry_bias', np.nan)
        df_t['entry_profit_ratio'] = np.nan
        df_t['entry_amount'] = np.nan
        df_t['exit_close_price'] = df_t.get('exit', np.nan)

    # 全局场景 (维度 O/P/Q 需要)
    df_g = None
    if os.path.exists(DEFAULT_AUDIT):
        df_g = pd.read_csv(DEFAULT_AUDIT)
        df_g['dt'] = pd.to_datetime(df_g['index']).dt.normalize()
        mkt_context = df_g[['dt', 'strat_primary_scenario', 'strat_is_market_ok']].copy()

    # 核心派生量
    df_t['ret'] = df_t['return_pct'] / 100.0 if df_t['return_pct'].abs().mean() > 0.5 else df_t['return_pct']
    df_t['mae_pct'] = df_t['mae'] / df_t['entry'].abs()
    df_t['mfe_pct'] = df_t['mfe'] / df_t['entry']
    df_t['alpha_decay'] = df_t['exit_alpha_rank'] - df_t['entry_alpha_rank']
    df_t['risk_surge'] = df_t['entry_risk_rank'] - df_t['exit_risk_rank']

    # --- 卖飞分析 (需价格矩阵) ---
    if has_debug:
        print(">>> 卖飞分析 (post-exit 10日最高价)...")
        price_matrix = df_d.pivot_table(index='date', columns='symbol', values='close')
        future_max = price_matrix.rolling(10).max().shift(-10)
        pts = []
        for it in df_t[['symbol', 'exit_date']].itertuples():
            sym, dt = it.symbol, it.exit_date
            pts.append(future_max.at[dt, sym] if (sym in future_max.columns and dt in future_max.index) else np.nan)
        df_t['post_exit_max'] = pts
        df_t['sell_fly_pct'] = df_t['post_exit_max'] / (df_t['exit_close_price'] + 1e-9) - 1.0
        df_t.loc[df_t['sell_fly_pct'] > 1.0, 'sell_fly_pct'] = np.nan
        df_t['is_sell_fly'] = df_t['sell_fly_pct'] > 0.10
    else:
        df_t['sell_fly_pct'] = np.nan
        df_t['is_sell_fly'] = False

    # ============================ 维度输出 ============================
    # 维度 A. 准入质量
    print("\n" + "-" * 40 + " [维度 A: 准入质量审计] " + "-" * 40)
    if df_t['entry_risk_rank'].notna().any():
        df_t['is_quality_entry'] = (df_t['entry_alpha_rank'] <= 0.05) & (df_t['entry_risk_rank'] >= 0.05)
        print(f" - 优质入场占比 (Alpha前5% + 风控安全): {df_t['is_quality_entry'].mean()*100:.2f}%")
    else:
        df_t['is_quality_entry'] = df_t['entry_alpha_rank'] <= 0.05
        print(f" - Alpha 前5%入场占比 (无风控排名): {df_t['is_quality_entry'].mean()*100:.2f}%")
    if df_t['entry_amount'].notna().any():
        print(f" - 入场标的平均成交额: {df_t['entry_amount'].mean()/1e8:.2f} 亿 (容量参考)")

    # 维度 B. 场景 x 乖离 交叉期望
    print("\n" + "-" * 40 + " [维度 B: 场景 x 乖离交叉矩阵] " + "-" * 40)
    df_t['bias_bin'] = pd.cut(df_t['entry_bias_val'], bins=[-np.inf, -0.05, 0, 0.05, np.inf],
                              labels=['超跌(<-5%)', '回踩(-5~0)', '强势(0~5%)', '加速(>5%)'])
    if df_t['entry_bias_val'].notna().any():
        print(df_t.pivot_table(index='primary_scenario', columns='bias_bin', values='ret',
                               aggfunc='mean', observed=False).round(4))
    else:
        print("跳过 (无 bias 快照)")

    # 维度 C. 场景 x 获利盘 交叉
    print("\n" + "-" * 40 + " [维度 C: 场景 x 获利盘矩阵] " + "-" * 40)
    if df_t['entry_profit_ratio'].notna().any():
        df_t['profit_group'] = pd.cut(df_t['entry_profit_ratio'], bins=[0, 0.2, 0.5, 0.8, 1.0],
                                      labels=['冷门(<20%)', '平庸(20-50%)', '活跃(50-80%)', '疯狂(>80%)'])
        print("[平均收益率矩阵]:")
        print(df_t.pivot_table(index='primary_scenario', columns='profit_group', values='ret',
                               aggfunc='mean', observed=False).round(4))
        print("[交易样本量]:")
        print(df_t.pivot_table(index='primary_scenario', columns='profit_group', values='ret',
                               aggfunc='count', observed=False).fillna(0).astype(int))
        if 'risk' in df_t['primary_scenario'].values:
            risk_crazy = df_t[(df_t['primary_scenario'] == 'risk') & (df_t['profit_group'] == '疯狂(>80%)')]['ret']
            if not risk_crazy.empty and risk_crazy.mean() < -0.02:
                print(f" 💡 RISK 场景 '疯狂'组收益 {risk_crazy.mean():.2%}，应禁止追高。")
    else:
        print("跳过 (无获利盘快照)")

    # 维度 D. 卖出原因效能 + 卖飞
    print("\n" + "-" * 40 + " [维度 D: 离场效能与卖飞审计] " + "-" * 40)
    agg_cols = {c: 'mean' for c in ['alpha_decay', 'risk_surge', 'ret'] if c in df_t.columns}
    if df_t['sell_fly_pct'].notna().any():
        agg_cols.update({'is_sell_fly': 'mean', 'sell_fly_pct': 'mean'})
    print(df_t.groupby('sell_reason').agg(agg_cols).round(4))

    # 维度 E. 费用摩擦
    print("\n" + "-" * 40 + " [维度 E: 费用与摩擦损耗] " + "-" * 40)
    fee_per_trade = 0.0012 * 2
    df_t['net_ret'] = df_t['ret'] - fee_per_trade
    print(f" - 单笔双边费率: {fee_per_trade:.2%} | 平均毛收益: {df_t['ret'].mean():.2%}")
    print(f" - 扣费后净胜率: {(df_t['net_ret'] > 0).mean()*100:.2f}%")
    print(f" - 换手依赖成本侵蚀: {fee_per_trade/(df_t['ret'].abs().mean()+1e-9):.2%}")

    # 维度 F. 双塔共线性
    print("\n" + "-" * 40 + " [维度 F: 双塔模型独立性] " + "-" * 40)
    if df_t['entry_risk_rank'].notna().any():
        mc = df_t['entry_alpha_rank'].corr(df_t['entry_risk_rank'])
        print(f" - 买入/风控入场排名相关性: {mc:.4f}")
        if abs(mc) > 0.7:
            print(" ⚠️ 两模型高度相关，风控可能只是 Alpha 反向表达。")
    else:
        print("跳过 (无风控排名快照)")

    # 维度 G. 极端交易画像
    print("\n" + "-" * 40 + " [维度 G: 暴利 vs 大亏画像] " + "-" * 40)
    cols_show = ['symbol', 'entry_date', 'ret',
                 'entry_alpha_rank', 'entry_risk_rank', 'primary_scenario']
    if 'entry_ml_rank' in df_t.columns and 'entry_alpha_rank' not in df_t.columns:
        cols_show[3] = 'entry_ml_rank'
    if cols_show[-2] == 'entry_risk_rank' and not df_t[cols_show[-2]].notna().any():
        cols_show.remove('entry_risk_rank')
    print("\n[Top 5 Profit]:")
    print(df_t.nlargest(5, 'ret')[cols_show].to_string(index=False))
    print("\n[Bottom 5 Loss]:")
    print(df_t.nsmallest(5, 'ret')[cols_show].to_string(index=False))

    # 维度 J. 利润留存率 + 排名滑坡
    print("\n" + "-" * 40 + " [维度 J: 退出效率与排名滑坡] " + "-" * 40)
    df_t['profit_retention'] = df_t['ret'] / (df_t['mfe_pct'].abs() + 1e-9)
    pos = df_t[df_t['ret'] > 0]
    print(f" - 平均利润留存率 (盈利单): {pos['profit_retention'].mean():.2%}")
    if df_t['exit_alpha_rank'].notna().any():
        df_t['rank_drift_per_loss'] = df_t['alpha_decay'] / (df_t['mfe_pct'] - df_t['ret'] + 1e-9)
        print(f" - 排名滑坡敏感度 (每1%回撤对应排名位移): {df_t['rank_drift_per_loss'].mean():.4f}")
        late = df_t[(df_t['alpha_decay'] > 0.20) & (df_t['ret'] < 0)]
        print(f" - 显著延迟卖出笔数 (Alpha腐烂>20%且亏损): {len(late)}")
        winners = df_t[df_t['mfe_pct'] > 0.05]
        if len(winners) >= 10:
            c = winners['exit_alpha_rank'].corr(winners['ret'])
            print(f" - 盈利单中 Alpha 排名与最终收益相关: {c:.4f} "
                  + ("(排名卖出有效)" if abs(c) > 0.1 else "(盈利后看排名卖出几乎无效)"))
    else:
        print("跳过排名滑坡子项 (无 debug)")

    # ---------------- 需 debug 的深度维度 ----------------
    if has_debug and df_g is not None and 'strat_primary_scenario' in mkt_context.columns:
        df_d = df_d.merge(mkt_context, left_on='date', right_on='dt', how='left')
        print("\n" + "-" * 40 + " [维度 M: 准入阈值弹性] " + "-" * 40)
        extra = df_d[(df_d['ml_rank'] > 0.01) & (df_d['ml_rank'] <= 0.03)]
        if not extra.empty:
            mr = extra['target_val_real'].mean()
            wr = (extra['gpr_target_real'] > 1.0).mean()
            print(f" - 1%-3% 区间样本 20日平均收益: {mr:.2%} | 胜率(GPR>1): {wr:.2%}")
            print(f"   {'💡 建议维持弹性阈值(1%-3%仍有正收益)' if mr > 0.01 else '💡 收紧至0.01合理'}")

        print("\n" + "-" * 40 + " [维度 O: 场景 x 排名期望收益] " + "-" * 40)
        df_d['rank_group'] = pd.cut(df_d['ml_rank'], bins=[0, 0.01, 0.03, 0.05, 1.0],
                                    labels=['Top1%', '1-3%', '3-5%', '5%以下'])
        pivot_rank_scene = df_d.pivot_table(index='strat_primary_scenario', columns='rank_group',
                                            values='target_val_real', aggfunc='mean', observed=False).round(4)
        print(pivot_rank_scene)

        print("\n" + "-" * 40 + " [维度 P: 每日可选标的数] " + "-" * 40)
        df_d['is_tradable'] = df_d['risk_ml_rank'] > 0.01
        print(df_d[df_d['is_tradable']].groupby(['date', 'rank_group'], observed=False)
              .size().unstack().fillna(0).mean().round(2))

        print("\n" + "-" * 40 + " [维度 Q: 排名 vs 前向最大回撤] " + "-" * 40)
        print(df_d.groupby('rank_group', observed=False)['fwd_mdd_20d'].mean().round(4))

        print("\n" + "-" * 40 + " [自动阈值建议] " + "-" * 40)
        for scene in pivot_rank_scene.index:
            row = pivot_rank_scene.loc[scene]
            if row.get('1-3%', np.nan) > 0.015 and not np.isnan(row.get('1-3%', np.nan)):
                print(f" - [{scene:12}]: 1-3%组收益 {row['1-3%']:.2%} → 建议放宽至 0.03")
            elif row.get('Top1%', np.nan) > 0.01 and not np.isnan(row.get('Top1%', np.nan)):
                print(f" - [{scene:12}]: 仅 Top1% 有效 → 建议收紧至 0.01")
            else:
                print(f" - [{scene:12}]: 全线低迷 → 建议 0.005 或强制空仓")
    else:
        print("\n⚠️ 缺少 debug 或 global_strategy_audit，跳过维度 M/O/P/Q (阈值决策支持)。")

    # ---------------- 最终诊断 ----------------
    print("\n" + "🚀" * 15 + " God-Mode 诊断建议 " + "🚀" * 15)
    if df_t['is_quality_entry'].mean() < 0.05:
        print(f" 💡 入场端过于苛刻 (优质入场 {df_t['is_quality_entry'].mean()*100:.1f}%)，可考虑放宽风控准入。")
    # 卖出原因名随 backtest 演化: Time_Capital_Efficiency → Time_Efficiency_Exit
    time_exits = df_t[df_t['sell_reason'] == 'Time_Efficiency_Exit']
    if not time_exits.empty and time_exits['sell_fly_pct'].notna().any():
        sf = time_exits['sell_fly_pct'].mean()
        if sf > 0.15:
            print(f" 💡 ‘时间效率卖出’卖飞严重 (平均 {sf:.1%})，考虑拉长 bars_held。")

    out = os.path.join(os.path.dirname(trades_file), 'god_mode_audit_v11.xlsx')
    df_t.to_excel(out, index=False)
    print(f"\n[系统] 审计完成，明细已导出: {out}")


# =============================================================================
# 卖出后走势验证 (退出时机 vs 选股质量)
# =============================================================================
def post_exit_audit(trades_path=None, debug_path=DEFAULT_DEBUG):
    print("\n" + "=" * 100)
    print(f"{' 🔍 卖出后走势验证 (退出时机 vs 选股质量) ':^100}")
    print("=" * 100)

    trades_file = trades_path or find_latest_trades_file()
    if not os.path.exists(trades_file):
        print(f"❌ 缺少 {trades_file}"); return
    if not os.path.exists(debug_path):
        print(f"❌ 缺少 {debug_path} (DEBUG_INFERENCE=1 回测)，跳过。"); return

    trades = pd.read_excel(trades_file)
    trades['symbol'] = _norm_symbol(trades['symbol'])
    trades['exit_date'] = pd.to_datetime(trades['exit_date']).dt.normalize()
    miss = [c for c in ['symbol', 'exit_date', 'exit', 'sell_reason'] if c not in trades.columns]
    if miss:
        print(f"❌ trades 缺列: {miss}"); return

    prices = pd.read_csv(debug_path, usecols=['date', 'symbol', 'close'], dtype={'symbol': str})
    prices['symbol'] = _norm_symbol(prices['symbol'])
    prices['date'] = pd.to_datetime(prices['date']).dt.normalize()
    prices = prices.sort_values(['symbol', 'date'])

    horizons = [3, 6, 12, 20]
    res_rows = []
    for _, tr in trades.iterrows():
        sub = prices[(prices['symbol'] == tr['symbol']) & (prices['date'] > tr['exit_date'])]
        if sub.empty:
            continue
        row = {'symbol': tr['symbol'], 'sell_reason': str(tr['sell_reason']), 'exit_ret': tr['return_pct']}
        for n in horizons:
            if len(sub) >= n:
                row[f'post_{n}d'] = sub.iloc[n - 1]['close'] / tr['exit'] - 1
            else:
                row[f'post_{n}d'] = np.nan
        res_rows.append(row)
    res = pd.DataFrame(res_rows)
    if res.empty:
        print("❌ 无匹配的卖出后价格数据。"); return

    print(f"\n匹配交易数: {len(res)}")
    print("-" * 90)
    for reason, grp in res.groupby('sell_reason'):
        if len(grp) < 3:
            continue
        line = f"  {reason:30s}: n={len(grp):4d} | "
        for n in horizons:
            v = grp[f'post_{n}d'].dropna()
            if len(v):
                line += f"{n}d={v.mean()*100:+5.2f}%({(v>0).mean()*100:3.0f}%) "
        print(line)

    print("\n" + "-" * 90)
    for r in res['sell_reason'].value_counts().head(3).index:
        grp = res[res['sell_reason'] == r]
        v12 = grp['post_12d'].dropna()
        if len(v12):
            verdict = ("退出过早(应多持)" if v12.mean() > 0.02
                       else "模型选股失败(卖出后仍跌)" if v12.mean() < -0.02
                       else "中性(横盘)")
            print(f"  {r}: 卖出后 12d {v12.mean()*100:+.2f}% → {verdict}")


def run_trade_audit():
    _tp = os.environ.get('AUDIT_TRADES_PATH') or None
    god_mode_auditor_v10(trades_path=_tp)
    post_exit_audit(trades_path=_tp)


if __name__ == "__main__":
    run_trade_audit()