# audit/check_model_entry.py
# 入场模型综合审计 (原 ml_check.py)
# 模型: 现役 chip_accumulation_v6_newfeat.pkl (29 特征), 审计数据优先用模型绑定 data_file
# 可选 AUDIT_MAX_ROWS 环境变量: 只读前 N 行使快速冒烟测试
import os
import joblib
import pandas as pd
import numpy as np
import logging
import warnings
from scipy.stats import spearmanr

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

ENTRY_MODEL_DEFAULT = 'chip_accumulation_v6_newfeat.pkl'
FALLBACK_DATA = 'model_data.csv'
SPLIT_DATE = '2020-01-01'

class AuditConfig:
    split_date = SPLIT_DATE
    sample_frac = 0.08   # 特征质量抽查占比 (15GB 文件下降低全量 I/O)
    top_k_ratio = [0.01, 0.05, 0.1, 0.2]


def _log(msg):
    logging.info(msg)
    print(msg)


def _load_pkg(model_path):
    if not os.path.exists(model_path):
        logging.error(f"模型文件 {model_path} 不存在，请先训练或切换到现役 pkl。")
        return None, None
    return joblib.load(model_path), model_path


def _resolve_data_path(pkg, data_path):
    """优先使用模型绑定的审计数据文件，避免口径错配。
    新模型绑定的是 external_data/audit/*_data.csv (大文件, 经软链接可访问)。"""
    if 'data_file' in pkg and pkg.get('data_file') and os.path.exists(pkg['data_file']):
        return pkg['data_file'], True
    logging.warning(f"模型未绑定可用 data_file (绑定={pkg.get('data_file')}), 使用传入路径 {data_path}")
    return data_path, False


def _read_audit(data_path, usecols):
    nrows = os.environ.get('AUDIT_MAX_ROWS')
    kw = dict(usecols=usecols)
    if nrows:
        kw['nrows'] = int(nrows)
    return pd.read_csv(data_path, **kw)


def _safe_ic_mean(series):
    """groupby.apply 在极小样本下可能返回多形状对象, 统一coerce为标量防崩溃。"""
    try:
        vals = pd.to_numeric(series, errors='coerce')
    except Exception:
        vals = pd.to_numeric(pd.Series(series.ravel()), errors='coerce')
    return float(vals.mean(skipna=True))


def load_scenario_map():
    """从 global_strategy_audit.csv 提取 日期->场景 映射 (用于分场景 OOS IC)。"""
    path = 'global_strategy_audit.csv'
    if not os.path.exists(path):
        return None
    try:
        g = pd.read_csv(path, usecols=['index', 'strat_primary_scenario'])
        g['date'] = pd.to_datetime(g['index']).dt.normalize()
        return dict(zip(g['date'], g['strat_primary_scenario']))
    except Exception as e:
        logging.warning(f"加载场景映射失败: {e}")
        return None


def run_comprehensive_audit(model_path=ENTRY_MODEL_DEFAULT, data_path=FALLBACK_DATA):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("\n" + "=" * 120)
    print(f"{'入场模型 & 特征审计 (PRO-V8 现役模型)':^120}")
    print("=" * 120)

    pkg, _ = _load_pkg(model_path)
    if pkg is None:
        return
    model, features = pkg['model'], pkg['features']
    data_path, _ = _resolve_data_path(pkg, data_path)
    _log(f"模型: {model_path} | 特征数: {len(features)} | 数据: {data_path}")

    # ======================================================================
    # 1. 特征质量审计 (抽样)
    # ======================================================================
    _log("阶段 1: 特征质量抽样审计...")
    all_cols = pd.read_csv(data_path, nrows=1).columns.tolist()
    essential = ['date', 'target', 'target_val', 'symbol']
    use_cols = [c for c in set(features + essential) if c in all_cols]
    missing_feat = [f for f in features if f not in all_cols]
    if missing_feat:
        print(f"❌ FATAL: 模型有 {len(missing_feat)} 个特征不在审计数据中: {missing_feat[:5]}")
        return
    df_sample = _read_audit(data_path, use_cols).replace([np.inf, -np.inf], np.nan)
    n_sample = max(1000, int(len(df_sample) * AuditConfig.sample_frac))
    df_sample = df_sample.sample(n_sample, random_state=42)
    report = []
    for col in features:
        d = df_sample[col].dropna()
        report.append({'Feature': col, 'Mean': d.mean(), 'Std': d.std(),
                       'NaN%': (df_sample[col].isna().sum() / len(df_sample)) * 100,
                       'Min': d.min(), 'Max': d.max()})
    print("\n" + "-" * 120 + f"\n{'1. 特征质量全局审计 (抽样)':^120}\n" + "-" * 120)
    print(pd.DataFrame(report).to_string(index=False,
          formatters={'Mean': '{:.4f}'.format, 'Std': '{:.4f}'.format, 'NaN%': '{:.2f}%'.format}))

    # ======================================================================
    # 2. IS vs OOS 预测表现
    # ======================================================================
    _log("阶段 2: 预测表现审计 (全量读取)...")
    eval_df = _read_audit(data_path, use_cols).replace([np.inf, -np.inf], np.nan)
    eval_df = eval_df.dropna(subset=features + ['target'])
    eval_df = eval_df.reset_index(drop=True)
    if 'date' in eval_df.columns:
        eval_df['date_dt'] = pd.to_datetime(eval_df['date'], format='mixed', errors='coerce')
    else:
        eval_df['date_dt'] = pd.NaT

    _log("正在生成预测值...")
    eval_df['pred'] = model.predict(eval_df[features])
    eval_df['is_oos'] = np.where(eval_df['date_dt'] < pd.Timestamp(AuditConfig.split_date),
                                 'In-Sample', 'Out-of-Sample')

    def calc_daily_ic(group):
        if len(group) < 15 or group['pred'].std() < 1e-8 or group['target'].std() < 1e-8:
            return pd.Series({'RankIC': np.nan})
        return pd.Series({'RankIC': spearmanr(group['pred'], group['target'])[0]})

    daily_ic = eval_df.groupby(['is_oos', 'date_dt']).apply(calc_daily_ic, include_groups=False).reset_index()
    summary = daily_ic.groupby('is_oos')['RankIC'].agg([
        ('RankIC_Mean', 'mean'), ('RankIC_Std', 'std'),
        ('IC_IR', lambda x: x.mean() / (x.std() + 1e-9)),
        ('IC_WinRate', lambda x: (x > 0).mean())]).reset_index()
    print("\n" + "-" * 120 + f"\n{'2. IS (训练期) vs OOS (测试期) 核心表现对比':^120}\n" + "-" * 120)
    print(summary.to_string(index=False, formatters={'RankIC_Mean': '{:.4f}'.format,
          'RankIC_Std': '{:.4f}'.format, 'IC_IR': '{:.4f}'.format, 'IC_WinRate': '{:.2%}'.format}))

    # 后续阶段仅用 OOS, 提前释放 In-Sample 内存
    eval_df = eval_df[eval_df['is_oos'] == 'Out-of-Sample'].copy()
    if eval_df.empty:
        print("❌ OOS 段为空，检查 SPLIT_DATE/data 范围。")
        return

    # ======================================================================
    # 3. OOS 年度稳定性 (新增维度)
    # ======================================================================
    print("\n" + "-" * 120 + f"\n{'3. OOS 年度 RankIC 稳定性审计':^120}\n" + "-" * 120)
    eval_df['year'] = eval_df['date_dt'].dt.year
    yearly = eval_df.groupby('year').apply(
        lambda x: pd.Series({
            '日数': x['date_dt'].nunique(),
            'RankIC': x.groupby('date_dt').apply(
                lambda g: spearmanr(g['pred'], g['target'])[0] if (len(g) > 15 and g['pred'].std() > 1e-8) else np.nan,
                include_groups=False).mean(),
            '样本': len(x)}), include_groups=False).reset_index()
    print(yearly.to_string(index=False))
    yr_ic = yearly.dropna(subset=['RankIC'])['RankIC'] if 'RankIC' in yearly else []
    if not yr_ic.empty:
        bad = yearly[(yearly['RankIC'] < 0) & (yearly['日数'] >= 30)]
        if not bad.empty:
            print(f"⚠️ 下列年度 OOS RankIC 为负 (模型边际失效信号): {bad['year'].tolist()}")
        else:
            print("✅ 所有含 ≥30 交易日的年度 OOS RankIC 均为正，跨年稳定。")

    # ======================================================================
    # 4. OOS 分场景 RankIC (新增维度)
    # ======================================================================
    print("\n" + "-" * 120 + f"\n{'4. OOS 分市场场景 RankIC 审计':^120}\n" + "-" * 120)
    scen_map = load_scenario_map()
    if scen_map is None:
        print("⚠️ 无 global_strategy_audit.csv，跳过场景分层。")
    else:
        eval_df['scenario'] = eval_df['date_dt'].dt.normalize().map(scen_map).fillna('normal')
        scen_ic = eval_df.groupby('scenario').apply(
            lambda x: pd.Series({
                '日数': x['date_dt'].nunique(),
                'RankIC': x.groupby('date_dt').apply(
                    lambda g: spearmanr(g['pred'], g['target'])[0] if (len(g) > 15 and g['pred'].std() > 1e-8) else np.nan,
                    include_groups=False).mean(),
                '样本': len(x)}), include_groups=False).reset_index()
        print(scen_ic.to_string(index=False))
        neg = scen_ic[scen_ic['RankIC'] < 0]
        if not neg.empty:
            print(f"⚠️ 场景分层中以下场景 RankIC 为负: {neg['scenario'].tolist()}")

    # ======================================================================
    # 5. 分箱单调性 + 多空对冲 (T+20)
    # ======================================================================
    print("\n" + "-" * 120 + f"\n{'5. OOS 分箱单调性审计 (Decile)':^120}\n" + "-" * 120)
    oos = eval_df.copy()
    dec = oos.copy()
    dec['bucket'] = dec.groupby('date_dt')['pred'].transform(
        lambda x: pd.qcut(x + np.random.uniform(0, 1e-12, len(x)), 10, labels=False, duplicates='drop')
        if x.nunique() >= 10 else np.nan
    )
    dec = dec.dropna(subset=['bucket'])
    if dec.empty:
        print("⚠️ OOS 每日截面样本 <10 只，分箱/多空对冲审计跳过 (全量数据下应正常)。")
    else:
        daily_b = dec.groupby(['date_dt', 'bucket'])[['target', 'target_val']].mean().reset_index()
        b_stats = daily_b.groupby('bucket')[['target', 'target_val']].mean().T
        print(b_stats.to_string())
        if 'target' in b_stats.index:
            mono = b_stats.loc['target'].dropna().is_monotonic_increasing
            print(f"\n📈 OOS 分箱 target 是否单调递增: {mono} " + ("✅" if mono else "⚠️ 需检查 top-bin 是否被拉爆"))
        ls = daily_b[daily_b['bucket'] == 9].set_index('date_dt')['target_val'] - \
             daily_b[daily_b['bucket'] == 0].set_index('date_dt')['target_val']
        if len(ls):
            ls_sharpe = (ls.mean() / (ls.std() + 1e-9)) * np.sqrt(242)
            print(f"📊 OOS 多空组合 (Decile9-Decile0) T+20 年化夏普: {ls_sharpe:.2f}")
            hit9 = (ls > 0).mean()
            print(f"✅ Decile9 单日跑赢 Decile0 的比例: {hit9:.1%}")

    # ======================================================================
    # 6. 极值决策命中率
    # ======================================================================
    print("\n" + "-" * 120 + f"\n{'6. 极值决策命中率与绝对质量审计 (OOS)':^120}\n" + "-" * 120)
    print(f"{'选股档位':<12} | {'对等排名命中率':<15} | {'实际目标均值':<15} | {'实际收益':<15} | {'提升倍数'}")
    for p in AuditConfig.top_k_ratio:
        def calc(group):
            k = max(1, int(len(group) * p))
            top_pre = group['pred'].nlargest(k).index
            top_act = group['target'].nlargest(k).index
            return pd.Series({'hit': len(set(top_pre) & set(top_act)) / k,
                              'rank': group.loc[top_pre, 'target'].mean(),
                              'raw': group.loc[top_pre, 'target_val'].mean()})
        res = oos.groupby('date_dt').apply(calc, include_groups=False).mean()
        hit_s = res['hit'] if isinstance(res, pd.Series) and 'hit' in res.index else np.nan
        print(f"Top {p:>4.1%}      | {hit_s:>15.2%} | {res['rank'] if isinstance(res, pd.Series) and 'rank' in res.index else np.nan:>15.4f} | "
              f"{res['raw'] if isinstance(res, pd.Series) and 'raw' in res.index else np.nan:>15.4f} | {(hit_s/p if p else 0):>10.2f}x")

    # ======================================================================
    # 7. 大盘崩溃日审计
    # ======================================================================
    print("\n" + "-" * 120 + f"\n{'7. 大盘极端日表现审计 (Crash Days)':^120}\n" + "-" * 120)
    if 'target_val' in oos.columns:
        mkt_med = oos.groupby('date_dt')['target_val'].median()
        if len(mkt_med) >= 20:
            crash_days = mkt_med[mkt_med < mkt_med.quantile(0.1)].index
            crash_data = oos[oos['date_dt'].isin(crash_days)]
            if not crash_data.empty:
                crash_top = crash_data.groupby('date_dt').apply(
                    lambda x: x.loc[x['pred'].nlargest(max(1, int(len(x) * 0.01))).index, 'target_val'].mean()
                    if len(x) > 0 else np.nan, include_groups=False).mean()
                print(f"崩溃日样本: {len(crash_days)} 天 | 大盘中位收益: {mkt_med[crash_days].mean():.4f}")
                print(f"崩溃日 Top1% 平均收益: {crash_top:.4f} → "
                      + ("✅ 展现超额韧性" if crash_top > mkt_med[crash_days].mean() else "⚠️ 无法对抗大盘崩溃"))

    # ======================================================================
    # 8. 特征重要性 + 换仓稳定性 + 逻辑一致性自检 + Alpha 衰减 (原 phase 6-8)
    # ======================================================================
    print("\n" + "-" * 120 + f"\n{'8. 特征重要性与换仓稳定性':^120}\n" + "-" * 120)
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        print(imp.to_string(index=False))
    if 'symbol' in oos.columns:
        daily_top = oos.sort_values(['date_dt', 'pred'], ascending=[True, False]).groupby('date_dt')['symbol'].apply(
            lambda x: set(x.head(20)), include_groups=False)
        turnovers = []
        prev = None
        for cur in daily_top:
            if prev is not None:
                denom = min(len(prev), len(cur))
                turnovers.append(1 - len(prev & cur) / denom if denom > 0 else 0)
            prev = cur
        if turnovers:
            print(f"\nOOS 每日平均换仓率 (Top20): {np.mean(turnovers):.2%}")
        else:
            print("\nOOS 时序过短, 换仓率不可计算。")

    print("\n" + "-" * 120 + f"\n{'9. 特征计算逻辑一致性自检':^120}\n" + "-" * 120)
    if {'ema_bias_norm_z', 'res_bias_norm_z'}.issubset(oos.columns):
        rc = oos[['ema_bias_norm_z', 'res_bias_norm_z']].dropna().corr()
        rc = rc.iloc[0, 1] if len(rc) else np.nan
        print(f"Bias趋势项 vs 残差项相关: {rc:.4f} (期望 <0.2) {'✅' if not np.isnan(rc) and abs(rc) < 0.2 else '⚠️'}")
    if {'dist_to_high90_z', 'ema_profit_z'}.issubset(oos.columns):
        cp = oos['dist_to_high90_z'].corr(oos['ema_profit_z'])
        print(f"价格位置 vs 获利盘 相关: {cp:.4f} (期望>0, 价格越高获利盘越高) {'✅' if not np.isnan(cp) and cp > 0 else '⚠️'}")

    print("\n" + "-" * 120 + f"\n{'10. Alpha 衰减审计 (预测时效性)':^120}\n" + "-" * 120)
    if oos.empty:
        print("⚠️ 无 OOS 样本, 跳过 Alpha 衰减。")
    else:
        if 'close' in oos.columns:
            oos = oos.sort_values(['symbol', 'date_dt']).reset_index(drop=True)
            for lag in range(1, 6):
                oos[f'fwd_ret_{lag}d'] = oos.groupby('symbol')['close'].pct_change(lag).shift(-lag)
            cols_d = {f'fwd_ret_{lag}d': f'T+{lag}' for lag in range(1, 6)}
        else:
            for lag in range(1, 6):
                oos[f'target_fwd_{lag}d'] = oos.groupby('symbol')['target'].shift(-lag)
            cols_d = {f'target_fwd_{lag}d': f'T+{lag}' for lag in range(1, 6)}
        decay = []
        for col, h in cols_d.items():
            ic_tmp = oos.groupby('date_dt').apply(
                lambda x: spearmanr(x['pred'], x[col])[0]
                if (len(x) > 20 and x[col].notna().sum() > 10 and x[col].nunique() > 1) else np.nan,
                include_groups=False)
            decay.append({'Horizon': h, 'RankIC': _safe_ic_mean(ic_tmp)})
        print(pd.DataFrame(decay).to_string(index=False))
        r0, rn = decay[0]['RankIC'], decay[-1]['RankIC']
        if not np.isnan(r0) and not np.isnan(rn) and r0 < rn:
            print("⚠️ T+1 IC 低于 T+5，信号滞后程度需关注 (或 T+20 收益以中段兑现)。")


# ==========================================================================================
# 二. 流动性/容量视角: Top100 换手、重叠净值模拟、盈亏平衡摩擦 (原 v4)
# ==========================================================================================
def run_capacity_audit(model_path=ENTRY_MODEL_DEFAULT, data_path=FALLBACK_DATA):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("\n" + "=" * 120)
    print(f"{'入场模型 流动性/容量视角 (Top100 重叠净值模拟)':^120}")
    print("=" * 120)

    if not os.path.exists(model_path):
        logging.error("模型文件不存在"); return
    pkg = joblib.load(model_path)
    model, features = pkg['model'], pkg['features']
    data_path, _ = _resolve_data_path(pkg, data_path)
    _log(f"模型: {model_path} ({len(features)} 特征) | 数据: {data_path}")

    essential = ['date', 'symbol', 'close', 'amount', 'change_pct', 'target', 'target_val']
    all_cols = pd.read_csv(data_path, nrows=1).columns
    use_cols = [c for c in set(features + essential) if c in all_cols]
    missing = [f for f in features if f not in all_cols]
    if missing:
        print(f"❌ FATAL: 数据缺特征 {missing[:5]}"); return
    df = _read_audit(data_path, use_cols).replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['target'])
    df['date_dt'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    _log("正在生成预测值...")
    df['pred'] = model.predict(df[features])
    df['daily_ret'] = df['change_pct'] / 100

    df = df.sort_values(['symbol', 'date_dt'])
    for gap in [5, 10, 20]:
        df[f'target_val_lag_{gap}'] = df.groupby('symbol')['target_val'].shift(-gap)

    oos = df[df['date_dt'] >= pd.Timestamp('2020-01-01')].sort_values(['date_dt', 'pred'], ascending=[True, False])

    # A. 换仓率 Top100
    top = oos.groupby('date_dt')['symbol'].apply(lambda x: set(x.head(100)), include_groups=False)
    tos = []
    prev = None
    for cur in top:
        if prev is not None:
            tos.append(1 - len(prev & cur) / len(prev))
        prev = cur
    avg_to = np.mean(tos)
    print("\n" + "-" * 120 + f"\n{'1. 动态换仓率与信号稳定性 (Top100)':^120}\n" + "-" * 120)
    print(f"OOS 每日平均换仓率: {avg_to:.2%}")

    # B. Alpha 衰减 (Delay 0/5/10/20)
    print("\n" + "-" * 120 + f"\n{'2. Alpha 衰减审计 (Delay 0/5/10/20d)':^120}\n" + "-" * 120)
    report = []
    def ic0(g):
        return spearmanr(g['pred'], g['target_val'])[0] if (len(g) > 10 and g['pred'].std() > 0) else np.nan
    report.append({'信号延迟': 'Delay 0d', 'RankIC': oos.groupby('date_dt').apply(ic0, include_groups=False).mean()})
    for gap in [5, 10, 20]:
        col = f'target_val_lag_{gap}'
        report.append({'信号延迟': f'Delay {gap}d',
                       'RankIC': oos.groupby('date_dt').apply(
                           lambda g, c=col: spearmanr(g['pred'], g[c])[0] if g[c].notna().sum() > 10 else np.nan,
                           include_groups=False).mean()})
    print(pd.DataFrame(report).to_string(index=False))

    # C. 20日重叠净值模拟
    print("\n" + "-" * 120 + f"\n{'3. 净值回撤 (20日持有期分仓模拟, Top20)':^120}\n" + "-" * 120)
    pivot_ret = oos.pivot(index='date_dt', columns='symbol', values='daily_ret').fillna(0)
    mask = oos.pivot(index='date_dt', columns='symbol', values='pred').rank(axis=1, ascending=False) <= 20
    dates = pivot_ret.index
    daily_strat = []
    mv, rv = mask.values, pivot_ret.values
    rev = len(dates)
    for i in range(20, rev):
        sub = []
        for lag in range(20):
            idx = mv[i - lag]
            if idx.any():
                sub.append(rv[i, idx].mean())
        daily_strat.append(np.mean(sub) if sub else np.nan)
    strat = pd.Series(daily_strat, index=dates[20:]).dropna()
    net = strat - (avg_to * 2 * 0.0015)
    nav = (1 + net).cumprod()
    dd = (nav - nav.cummax()) / nav.cummax()
    print(f"OOS 累计收益: {nav.iloc[-1]-1:.2%} | 最大回撤: {dd.min():.2%} | "
          f"年化夏普: {(net.mean()*242)/(net.std()*np.sqrt(242)+1e-9):.2f}")

    # D. 盈亏平衡
    print("\n" + "-" * 120 + f"\n{'4. 盈亏持平点压力测试 (换手 Top100)':^120}\n" + "-" * 120)
    avg_m = strat.mean()
    for bps in [5, 15, 30]:
        cost = avg_to * 2 * (bps / 10000)
        print(f"单边摩擦 {bps:>2} bps | 每日净损益: {avg_m-cost:>8.4%} | 成本占比: {cost/avg_m:>6.2%}")

    # E. 特征泄露
    print("\n" + "-" * 120 + f"\n{'5. 特征泄露自检 (特征与当日涨跌相关)':^120}\n" + "-" * 120)
    leak = [{'Feature': f, 'Corr_Today': oos[f].corr(oos['change_pct'])} for f in features if f in oos.columns]
    print(pd.DataFrame(leak).sort_values('Corr_Today', ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    run_comprehensive_audit()
    run_capacity_audit()