# audit/check_scenario.py
# 大盘五象限场景标签科学审计 (Phase 1)
#
# 目的: 检验 is_market_ok.py 场景标签是否具备其语义主张的前瞻预测力,
#       并对关键硬编码参数做敏感性(tornado)分析, 为"滚动校准/动态替代"提供证据基线。
#
# 方法:
#   A. 事件研究: 以 T 日收盘打标(与回测 before_exec_fn 同一函数、同一口径),
#      度量 T+5/10/20/60 日前瞻收益(中证全指指数; --ew 时叠加等权域)。
#   B. 统计检验: 分场景 Welch t / Mann-Whitney U / 移动块自助置信区间 / 尾部概率。
#   C. 标签质量: 两两区分度、转移矩阵、翻转率、持续段长度、分年稳健性。
#   D. 基准对比: 五象限 eta^2 vs MA60 趋势过滤 / 20 日动量符号两个免费基准;
#      尾部捕获率(坏日子被 risk/caution 提前标记的比例)对比。
#   E. quota 一致性: 现役 BUY_QUOTA_* 隐含的场景偏好排序 vs 实测前瞻收益排序。
#   F. 参数 tornado (--tornado): 对 is_market_ok.py 内联字面量做定向文本替换后
#      重打标, 度量标签分布 TV 距离 / delta-eta2 / delta 尾部捕获 / delta 翻转率。
#      替换串缺失时立即报错(防源码漂移导致静默错标), 仅审计用途、不改生产链路。
#
# 数据口径说明: 广度表默认读 stock_data_cache/market_context_cache.parquet
#   (只读审计; 该文件由 get_base_data task_market 在每次数据同步后刷新)。
#   --source rebuild 时改用 co_compute.calculate_high_low_stats 从个股库现场
#   重算(慢, 分钟级), 结果不写回 parquet。
#
# 用法:
#   python run.py audit scenario                    # 默认 2021-01-01 至今
#   python run.py audit scenario --start 2019-01-01 --ew
#   python run.py audit scenario --tornado          # 参数敏感性
#   python run.py audit scenario --fixed            # 固定阈值版对照
#
# HOLDOUT 纪律: 2025-09-01 起的区间为终审专用, 迭代期评估一律 --end 2025-08-31;
# 终审只许对最终幸存者各执行一次。
import os
import re
import sys
import types
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

import is_market_ok

PARQUET_PATH = os.path.join('stock_data_cache', 'market_context_cache.parquet')
ZZQZ_PATH = 'zzqz_df.xlsx'
HORIZONS = (5, 10, 20, 60)
SCENARIOS = ['risk', 'bottom', 'opportunity', 'caution', 'normal']
QUOTA_DEFAULTS = {'bottom': 5, 'opportunity': 5, 'normal': 2, 'caution': 3, 'risk': 0}
WARMUP_DAYS = 120          # 打标最少历史(交易日), 规避冷启动 normal 兜底污染
BAD_TAIL = -0.05           # "坏日子"定义: fwd20 <= -5%
BOOT_N = 500               # 块自助重采样次数
BOOT_BLOCK = 10            # 块长度(交易日)


# =========================================================================
# 0. 工具函数
# =========================================================================
def prepare_zzqz():
    """复刻 backtest.py 导入期对 zzqz_df 的派生列构造 (唯一允许的复制点)。"""
    z = pd.read_excel(ZZQZ_PATH).rename(columns={
        '日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low',
        '收盘': 'close', '成交量': 'volume', '成交额': 'amount',
        '振幅': 'amplitude', '涨跌幅': 'change_pct', '涨跌额': 'change', '换手率': 'turnover'})
    z['date'] = pd.to_datetime(z['date'], format='%Y-%m-%d')
    z.set_index('date', inplace=True)
    z = z.sort_index()
    z['vol_ma5'] = z['volume'].rolling(5).mean()
    z['vol_ma20'] = z['volume'].rolling(20).mean()
    z['vol_ma60'] = z['volume'].rolling(60).mean()
    z['close_ma5'] = z['close'].rolling(5).mean()
    z['close_ma10'] = z['close'].rolling(10).mean()
    z['close_ma20'] = z['close'].rolling(20).mean()
    z['close_ma60'] = z['close'].rolling(60).mean()
    z['close_q30_w20'] = z['close'].rolling(20, min_periods=5).quantile(0.3)
    z['close_q30_w60'] = z['close'].rolling(60, min_periods=15).quantile(0.3)
    z['volume_q15_w120'] = z['volume'].rolling(120, min_periods=30).quantile(0.15)
    z['close_max_w20'] = z['close'].rolling(20, min_periods=5).max()
    return z


def load_breadth(source='parquet'):
    if source == 'parquet':
        b = pd.read_parquet(PARQUET_PATH)
    else:
        from local_data_cache import LocalDataCache
        from screen import basic_screen
        import co_compute
        cache_dir = './stock_data_cache'
        symbols = basic_screen(cache_dir=cache_dir)
        print(f"[rebuild] 中证全指股票池 {len(symbols)} 只, 逐库读取中...")
        ldc = LocalDataCache(cache_dir=cache_dir)
        parts = []
        for i, s in enumerate(symbols):
            try:
                df = ldc.get_stock_data(s, '1990-01-01', '2100-01-01', adjust='hfq', mode=2)
                if df is None or df.empty:
                    continue
                df = df.copy()
                df['symbol'] = s
                parts.append(df[['date', 'symbol', 'close', 'amount']])
            except Exception:
                continue
            if i % 1000 == 0:
                print(f"[rebuild] 已读取 {i}/{len(symbols)}")
        full = pd.concat(parts, ignore_index=True)
        full['date'] = pd.to_datetime(full['date'])
        b = co_compute.calculate_high_low_stats(full)
    b['date'] = pd.to_datetime(b['date'])
    return b.set_index('date').sort_index()


def label_days(eval_dates, zzqz_df, breadth_df, judge_fn, total_stocks=None,
               use_dynamic=True):
    """对 eval_dates 逐日调用场景判定 (输入严格 <= T 收盘, 与回测同口径)。"""
    rows = []
    for d in eval_dates:
        r = judge_fn(d, zzqz_df, breadth_df, total_stocks=total_stocks,
                     use_dynamic_threshold=use_dynamic)
        rows.append({'date': d, 'primary_scenario': r['primary_scenario'],
                     'position_multiplier': r['position_multiplier'],
                     'is_market_ok': bool(r['is_market_ok']),
                     'decision_reason': r['decision_reason']})
    lab = pd.DataFrame(rows).set_index('date')

    # 前瞻收益 (指数): fwd_k = close_{T+k}/close_T - 1
    close = zzqz_df['close'].reindex(lab.index)
    prev20 = zzqz_df['close'].shift(20).reindex(lab.index)
    ma60 = zzqz_df['close_ma60'].reindex(lab.index)
    for k in HORIZONS:
        lab[f'fwd{k}'] = close.shift(-k) / close - 1.0
    lab['ma60_up'] = (close >= ma60).astype(int)
    lab['mom20_pos'] = (close / prev20 - 1 > 0).astype(int)
    return lab


def attach_ew_forward(lab, ew_ret):
    """等权域前瞻收益: 日度截面等权平均收益 -> 前向 k 日复合。"""
    lp = np.log1p(ew_ret.reindex(lab.index).fillna(0.0))
    for k in HORIZONS:
        lab[f'ewfwd{k}'] = np.expm1(lp.rolling(k).sum().shift(-k))
    return lab


def build_ew_return_series(cache_path=None):
    """全市场日度等权平均收益 (中证全指股票池, hfq 口径与 parquet 一致)。
    cache_path 提供时读写缓存, 避免重复分钟级全库扫描。"""
    if cache_path and os.path.exists(cache_path):
        ew = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        print(f"[ew] 使用缓存: {cache_path} ({len(ew)} 天)")
        return ew
    from local_data_cache import LocalDataCache
    from screen import basic_screen
    cache_dir = './stock_data_cache'
    symbols = basic_screen(cache_dir=cache_dir)
    print(f"[ew] 中证全指股票池 {len(symbols)} 只, 计算等权日收益...")
    ldc = LocalDataCache(cache_dir=cache_dir)
    s_sum, s_cnt = {}, {}
    for i, sym in enumerate(symbols):
        try:
            df = ldc.get_stock_data(sym, '1990-01-01', '2100-01-01', adjust='hfq', mode=2)
            if df is None or len(df) < 2:
                continue
            r = df['close'].pct_change()
            g = r.groupby(df['date']).agg(['sum', 'count'])
            for dt, row in g.iterrows():
                dt = pd.Timestamp(dt).normalize()
                s_sum[dt] = s_sum.get(dt, 0.0) + float(row['sum'])
                s_cnt[dt] = s_cnt.get(dt, 0) + int(row['count'])
        except Exception:
            continue
        if i % 1000 == 0:
            print(f"[ew] 已读取 {i}/{len(symbols)}")
    ew = pd.Series({k: s_sum[k] / max(s_cnt[k], 1) for k in s_sum}).sort_index()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ew.to_frame('ew_ret').to_csv(cache_path)
        print(f"[ew] 已缓存: {cache_path}")
    return ew


# =========================================================================
# 1.5 选股质量交叉验证 (纸面回放回测实际买入名单)
# =========================================================================
def pick_quality_by_scenario(audit_csv='global_strategy_audit.csv',
                             ew_cache=None, fwd_horizon=20):
    """用 global_strategy_audit.csv 的 strat_top_x_buys 做前向收益回放,
    检验 quota 排序在'选股质量'层面是否成立 (而非仅指数层面)。"""
    banner("I. 选股质量交叉验证 (回测实际 picks 的 T+%d 前向收益 vs 全域等权基线)" % fwd_horizon)
    if not os.path.exists(audit_csv):
        print(f"[跳过] 缺少 {audit_csv}")
        return None
    audit = pd.read_csv(audit_csv)
    audit['dt'] = pd.to_datetime(audit['index']).dt.normalize()
    audit = audit.dropna(subset=['strat_primary_scenario', 'strat_top_x_buys'])
    audit = audit[audit['strat_top_x_buys'].astype(str).str.contains(r'\d{6}', na=False)]
    if audit.empty:
        print("[跳过] 审计 CSV 无有效买入记录")
        return None

    day_picks = {}
    all_syms = set()
    for _, row in audit.iterrows():
        syms = sorted(set(re.findall(r'\d{6}', str(row['strat_top_x_buys']))))
        if syms:
            day_picks[row['dt']] = syms
            all_syms.update(syms)
    print(f"[picks] {len(day_picks)} 个买入日 | {len(all_syms)} 只不同标的")

    from local_data_cache import LocalDataCache
    ldc = LocalDataCache(cache_dir='./stock_data_cache')
    closes = {}
    for i, sym in enumerate(sorted(all_syms)):
        try:
            df = ldc.get_stock_data(sym, '1990-01-01', '2100-01-01', adjust='hfq', mode=2)
            if df is not None and len(df):
                s = df.set_index(pd.to_datetime(df['date']))['close']
                closes[sym] = s[~s.index.duplicated()]
        except Exception:
            continue
        if i % 500 == 0:
            print(f"[picks] 已读取 {i}/{len(all_syms)}")

    wide = pd.DataFrame(closes).sort_index()
    pick_fwd = (wide.shift(-fwd_horizon) / wide - 1.0).mean(axis=1)  # 当日 picks 等权 fwd

    # 全域等权基线 (缓存复用)
    ew_ret = build_ew_return_series(cache_path=ew_cache)
    lp = np.log1p(ew_ret)
    ew_fwd = np.expm1(lp.rolling(fwd_horizon).sum().shift(-fwd_horizon))

    rows = []
    for dt, syms in day_picks.items():
        if dt not in pick_fwd.index or pd.isna(pick_fwd.loc[dt]):
            continue
        rows.append({'date': dt, 'scenario': str(audit.loc[audit['dt'] == dt, 'strat_primary_scenario'].iloc[0]),
                     'n_picks': len(syms), 'pick_fwd': pick_fwd.loc[dt],
                     'ew_fwd': ew_fwd.reindex([dt]).iloc[0]})
    pq = pd.DataFrame(rows)
    pq['excess'] = pq['pick_fwd'] - pq['ew_fwd']

    print(f"\n{'场景':12s} {'天数':>5s} {'picks数':>7s} {'pick_fwd':>9s} {'ew_fwd':>8s} {'选股超额':>8s} "
          f"{'胜过基线':>8s}")
    for sc in SCENARIOS:
        g = pq[pq['scenario'] == sc].dropna(subset=['pick_fwd', 'ew_fwd'])
        if g.empty:
            continue
        print(f"{sc:12s} {len(g):5d} {g['n_picks'].sum():7d} {g['pick_fwd'].mean() * 100:+8.2f}% "
              f"{g['ew_fwd'].mean() * 100:+7.2f}% {g['excess'].mean() * 100:+8.2f}% "
              f"{fmt_pct((g['excess'] > 0).mean()):>8s}")
    print("\n解读: 若某场景 quota 高但选股超额持续为负, 说明该象限的'放行更多名额'缺乏选股质量支撑;")
    print("      注意 picks 是 PyBroker buy_delay=1 次日开盘成交的代理, 与真实成交有滑点差异。")
    return pq


def block_bootstrap_ci(x, n_boot=BOOT_N, block=BOOT_BLOCK, seed=7):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < block * 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_boot, nb))
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = (starts[b][:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        means[b] = x[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi


def eta_squared(y, groups):
    df = pd.DataFrame({'y': y, 'g': groups}).dropna()
    if df['g'].nunique() < 2:
        return np.nan
    grand = df['y'].mean()
    ss_tot = ((df['y'] - grand) ** 2).sum()
    if ss_tot <= 0:
        return np.nan
    ss_bet = sum(len(g) * (g['y'].mean() - grand) ** 2 for _, g in df.groupby('g'))
    return ss_bet / ss_tot


def fmt_pct(v, digits=1):
    return "n/a" if pd.isna(v) else f"{v * 100:.{digits}f}%"


def tv_distance(p_counts, q_counts):
    keys = set(p_counts) | set(q_counts)
    p = np.array([p_counts.get(k, 0) for k in keys], float)
    q = np.array([q_counts.get(k, 0) for k in keys], float)
    p, q = p / p.sum(), q / q.sum()
    return 0.5 * float(np.abs(p - q).sum())


def banner(title):
    print("\n" + "=" * 100)
    print(f"{title:^100}")
    print("=" * 100)


# =========================================================================
# 1. 分场景事件研究
# =========================================================================
def event_study_table(lab, ew=False):
    base20 = lab['fwd20'].dropna()
    rows = []
    for sc in SCENARIOS:
        sub = lab[lab['primary_scenario'] == sc]
        rest = lab[lab['primary_scenario'] != sc]
        f20 = sub['fwd20'].dropna()
        r20 = rest['fwd20'].dropna()
        row = {'场景': sc, '天数': len(sub), '占比': len(sub) / max(len(lab), 1)}
        for k in HORIZONS:
            v = sub[f'fwd{k}'].dropna()
            row[f'mean{k}d'] = v.mean() if len(v) else np.nan
        row['win20d'] = (f20 > 0).mean() if len(f20) else np.nan
        row['excess20'] = f20.mean() - base20.mean() if len(f20) else np.nan
        ok = len(f20) > 5 and len(r20) > 5
        row['p_t'] = stats.ttest_ind(f20, r20, equal_var=False).pvalue if ok else np.nan
        row['p_mwu'] = stats.mannwhitneyu(f20, r20, alternative='two-sided').pvalue if ok else np.nan
        lo, hi = block_bootstrap_ci(f20)
        row['ci20_lo'], row['ci20_hi'] = lo, hi
        row['tail_bad'] = (f20 <= BAD_TAIL).mean() if len(f20) else np.nan
        row['tail_good'] = (f20 >= 0.05).mean() if len(f20) else np.nan
        if ew and 'ewfwd20' in lab.columns:
            e20 = sub['ewfwd20'].dropna()
            row['ew_mean20'] = e20.mean() if len(e20) else np.nan
            row['ew_excess20'] = e20.mean() - lab['ewfwd20'].mean() if len(e20) else np.nan
        rows.append(row)
    ev = pd.DataFrame(rows)

    banner("A. 分场景前瞻收益事件研究 (T+20 为主判据)")
    show = ev[['场景', '天数', '占比', 'mean5d', 'mean10d', 'mean20d', 'mean60d',
               'win20d', 'excess20', 'ci20_lo', 'ci20_hi', 'p_t', 'p_mwu',
               'tail_bad', 'tail_good']].copy()
    for c in ['mean5d', 'mean10d', 'mean20d', 'mean60d', 'excess20', 'ci20_lo', 'ci20_hi']:
        show[c] = show[c].map(lambda v: "n/a" if pd.isna(v) else f"{v * 100:+.2f}%")
    for c in ['占比', 'win20d', 'tail_bad', 'tail_good']:
        show[c] = show[c].map(fmt_pct)
    for c in ['p_t', 'p_mwu']:
        show[c] = show[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    print(show.to_string(index=False))
    if ew and 'ew_mean20' in ev.columns:
        ewshow = ev[['场景', 'ew_mean20', 'ew_excess20']].copy()
        for c in ['ew_mean20', 'ew_excess20']:
            ewshow[c] = ewshow[c].map(lambda v: "n/a" if pd.isna(v) else f"{v * 100:+.2f}%")
        print("\n[等权域对照 (ewfwd20)]")
        print(ewshow.to_string(index=False))
    base_rate = fmt_pct((base20 <= BAD_TAIL).mean())
    print(f"\n解读: bottom/opportunity 的 mean20d 应显著>0 且 excess20>0; risk 的 tail_bad 应数倍于"
          f"\n无条件基率({base_rate}); CI 不含 0 或 p<0.05 才支持该象限存在价值。")
    return ev


def pairwise_separation(lab):
    banner("B. 两两区分度 (fwd20 Mann-Whitney p 值矩阵, p>=0.05 表示不可区分)")
    cols = {}
    for sc in SCENARIOS:
        v = lab.loc[lab['primary_scenario'] == sc, 'fwd20'].dropna()
        if len(v):
            cols[sc] = v
    mat = pd.DataFrame(index=list(cols), columns=list(cols), dtype=float)
    for a in cols:
        for b in cols:
            if a != b:
                mat.loc[a, b] = stats.mannwhitneyu(cols[a], cols[b], alternative='two-sided').pvalue
    print(mat.map(lambda v: "  -  " if pd.isna(v) else f"{v:.3f}").to_string())
    weak = [(a, b) for i, a in enumerate(mat.index) for b in mat.index[i + 1:]
            if pd.notna(mat.loc[a, b]) and mat.loc[a, b] >= 0.05]
    if weak:
        print(f"\n[警告] 不可区分对: {weak}")
        print("       这些象限在收益维度上缺乏存在依据, 建议合并或改用其他目标变量重新定义。")


def stability_report(lab):
    banner("C. 标签稳定性 (抖动 = 交易成本放大器)")
    seq = lab['primary_scenario']
    trans = pd.crosstab(seq.shift(1), seq, normalize='index').reindex(index=SCENARIOS, columns=SCENARIOS)
    print("[状态转移概率 (行->列)]:")
    print(trans.map(lambda v: "  -  " if pd.isna(v) else f"{v:.2f}").to_string())
    flip = (seq != seq.shift(1)).iloc[1:].mean()
    print(f"\n总体翻转率: {fmt_pct(flip)} (平均每 ~{1 / max(flip, 1e-9):.1f} 个交易日换一次场景)")
    run_id = (seq != seq.shift()).cumsum()
    runs = seq.groupby(run_id).agg(['first', 'size'])
    print("[各场景持续段统计]:")
    for sc in SCENARIOS:
        seg = runs[runs['first'] == sc]['size']
        if len(seg):
            print(f"  {sc:12s}: 段数 {len(seg):4d} | 中位 {seg.median():4.0f} 天 | "
                  f"P25 {seg.quantile(.25):4.0f} | P75 {seg.quantile(.75):4.0f} | 最长 {seg.max():4d}")
    short = runs[(runs['size'] <= 3)]
    print(f"  <=3 天短段占比: {fmt_pct(len(short) / max(len(runs), 1))} — 过高说明阈值切在分布密集区")


def yearly_breakdown(lab):
    banner("D. 分年稳健性 (防全样本叙事)")
    rows = []
    for y, g in lab.groupby(lab.index.year):
        top = g['primary_scenario'].value_counts(normalize=True)
        f20 = g['fwd20'].dropna()
        rows.append({'年份': y, '天数': len(g),
                     '主场景': f"{top.index[0]}({top.iloc[0] * 100:.0f}%)",
                     'risk+caution占比': fmt_pct(g['primary_scenario'].isin(['risk', 'caution']).mean()),
                     'mean20d': "n/a" if not len(f20) else f"{f20.mean() * 100:+.2f}%",
                     'win20d': "n/a" if not len(f20) else fmt_pct((f20 > 0).mean()),
                     'eta2_当年': f"{eta_squared(g['fwd20'], g['primary_scenario']):.3f}"})
    print(pd.DataFrame(rows).to_string(index=False))
    eta_years = [float(str(r['eta2_当年'])) for r in rows]
    if eta_years:
        neg = sum(1 for v in eta_years if v <= 0.005)
        print(f"\n[检查] 当年 eta^2<=0.005 (标签几乎不解释收益) 的年份数: {neg}/{len(eta_years)}")


def baseline_comparison(lab):
    banner("E. 基准对比: 五象限标签 vs 免费基准 (fwd20)")
    y = lab['fwd20']
    eta5 = eta_squared(y, lab['primary_scenario'])
    eta_ma = eta_squared(y, lab['ma60_up'])
    eta_mom = eta_squared(y, lab['mom20_pos'])
    print(f"  五象限标签   eta^2 = {eta5:.4f}")
    print(f"  MA60 二分    eta^2 = {eta_ma:.4f}")
    print(f"  mom20 二分   eta^2 = {eta_mom:.4f}")

    bad = y <= BAD_TAIL
    flags = {
        'risk(仅)': lab['primary_scenario'].eq('risk'),
        'risk|caution': lab['primary_scenario'].isin(['risk', 'caution']),
        'MA60下方': lab['ma60_up'] == 0,
        'mom20<0': lab['mom20_pos'] == 0,
    }
    n_bad = int(bad.sum())
    print(f"\n[尾部捕获对比] 坏日子定义 fwd20<={BAD_TAIL:.0%}, 全期 {n_bad} 天 "
          f"({fmt_pct(bad.mean())})")
    print(f"  {'标记规则':14s} {'捕获率(recall)':>14s} {'精确率(precision)':>18s}")
    for name, f in flags.items():
        if f.sum() == 0:
            continue
        recall = bad[f].sum() / max(n_bad, 1)
        prec = bad[f].sum() / max(int(f.sum()), 1)
        print(f"  {name:14s} {fmt_pct(recall):>14s} {fmt_pct(prec):>18s}")
    verdict = "胜过两个基准" if (pd.notna(eta5) and eta5 > max(eta_ma, eta_mom)) else "未胜过免费基准, 决策树存在过拟合嫌疑"
    print(f"\n结论: 五象限 eta^2 {verdict}")


def quota_consistency(lab):
    banner("F. quota 排序一致性 (现役 BUY_QUOTA_* 隐含偏好 vs 实测前瞻收益)")
    quota = {sc: int(os.environ.get(f'BUY_QUOTA_{sc.upper()}', QUOTA_DEFAULTS[sc])) for sc in SCENARIOS}
    mean20 = lab.groupby('primary_scenario')['fwd20'].mean()
    tbl = pd.DataFrame({'quota': pd.Series(quota), 'mean_fwd20': mean20}).reindex(SCENARIOS)
    tbl['mean_fwd20'] = tbl['mean_fwd20'].map(lambda v: "n/a" if pd.isna(v) else f"{v * 100:+.2f}%")
    print(tbl.to_string())
    exp_order = [sc for sc, _ in sorted(quota.items(), key=lambda kv: -kv[1]) if quota[sc] > 0]
    act_order = list(mean20.dropna().sort_values(ascending=False).index)
    common = [s for s in exp_order if s in act_order]
    act_c = [s for s in act_order if s in common]
    inversions = sum(1 for i in range(len(common)) for j in range(i + 1, len(common))
                     if act_c.index(common[i]) > act_c.index(common[j]))
    print(f"\nquota 隐含排序: {' > '.join(exp_order)}")
    print(f"实测收益排序 : {' > '.join(act_order)}")
    print(f"排序逆序对数 : {inversions}/{len(common) * (len(common) - 1) // 2}"
          + ("  — 存在逆序, 建议用 audit_signal_level 的场景分层复核后调整 quota" if inversions else ""))


# =========================================================================
# 2. 参数 tornado (定向文本替换 + 重打标; 替换串缺失即报错防漂移)
# =========================================================================
_KNIFE = "(zzqz_row['close'] / zzqz_row['close_prev'] - 1.0) < -0.015"
_SURGE = "available_breadth['low20'].iloc[-2] * 1.25"
_MELT = "max(breadth_row.get('low20_ratio_q80', 0.18), 0.18)"
_SPARK = "avg_high_5d * 1.1"
_QUIET = "- 1) < 0.005"

# (变体名, 原文锚点串, 替换后整串); 锚点在源码中必须存在, 否则立即报错防漂移
TORNADO_SPECS = [
    ('飞刀单日大跌线=-0.010', _KNIFE, _KNIFE.replace('< -0.015', '< -0.010')),
    ('飞刀单日大跌线=-0.025', _KNIFE, _KNIFE.replace('< -0.015', '< -0.025')),
    ('踩踏激增系数=1.15', _SURGE, _SURGE.replace('* 1.25', '* 1.15')),
    ('踩踏激增系数=1.50', _SURGE, _SURGE.replace('* 1.25', '* 1.50')),
    ('熔断底线=0.14', _MELT, "max(breadth_row.get('low20_ratio_q80', 0.14), 0.14)"),
    ('熔断底线=0.22', _MELT, "max(breadth_row.get('low20_ratio_q80', 0.22), 0.22)"),
    ('点火放大系数=1.05', _SPARK, _SPARK.replace('avg_high_5d * 1.1', 'avg_high_5d * 1.05')),
    ('点火放大系数=1.20', _SPARK, _SPARK.replace('avg_high_5d * 1.1', 'avg_high_5d * 1.20')),
    ('隐性走强平静带=0.003', _QUIET, _QUIET.replace('< 0.005', '< 0.003')),
    ('隐性走强平静带=0.010', _QUIET, _QUIET.replace('< 0.005', '< 0.010')),
]


def load_variant_module(src_text):
    mod = types.ModuleType('is_market_ok_variant')
    mod.__file__ = is_market_ok.__file__
    exec(compile(src_text, mod.__file__, 'exec'), mod.__dict__)
    return mod


def run_tornado(eval_dates, zzqz_df, breadth_df, base_lab, use_dynamic=True,
                total_stocks=None):
    with open(is_market_ok.__file__, encoding='utf-8') as f:
        src = f.read()

    def metrics(l):
        dist = l['primary_scenario'].value_counts().to_dict()
        flip = (l['primary_scenario'] != l['primary_scenario'].shift()).iloc[1:].mean()
        bad = l['fwd20'] <= BAD_TAIL
        cap = (bad & l['primary_scenario'].isin(['risk', 'caution'])).sum() / max(int(bad.sum()), 1)
        return tv_distance(dist, base_dist), eta_squared(l['fwd20'], l['primary_scenario']), cap, flip

    base_bad = base_lab['fwd20'] <= BAD_TAIL
    base_dist = base_lab['primary_scenario'].value_counts().to_dict()
    b_flip = (base_lab['primary_scenario'] != base_lab['primary_scenario'].shift()).iloc[1:].mean()
    b_e2 = eta_squared(base_lab['fwd20'], base_lab['primary_scenario'])
    b_cap = (base_bad & base_lab['primary_scenario'].isin(['risk', 'caution'])).sum() / \
        max(int(base_bad.sum()), 1)

    rows = [{'参数变体': 'BASELINE', 'TV距离': 0.0, 'd_eta2': 0.0, 'd_尾部捕获': 0.0, 'd_翻转率': 0.0}]
    for name, old, new in TORNADO_SPECS:
        if src.count(old) == 0:
            raise SystemExit(f"[tornado] 源码漂移: 找不到锚点串 -> {old!r}\n请更新 TORNADO_SPECS。")
        mod = load_variant_module(src.replace(old, new))
        lab_v = label_days(eval_dates, zzqz_df, breadth_df,
                           mod.scenario_based_market_judgment,
                           total_stocks=total_stocks, use_dynamic=use_dynamic)
        tv, e2, cap, flip = metrics(lab_v)
        rows.append({'参数变体': name, 'TV距离': tv, 'd_eta2': e2 - b_e2,
                     'd_尾部捕获': cap - b_cap, 'd_翻转率': flip - b_flip})
        print(f"  [tornado] {name}: TV={tv:.3f} dEta2={e2 - b_e2:+.4f} "
              f"dCap={cap - b_cap:+.3f} dFlip={flip - b_flip:+.3f}")
        del mod, lab_v

    tor = pd.DataFrame(rows)
    banner("G. 参数敏感性 tornado (绝对值越大 = 该参数越脆弱/越值得滚动校准)")
    t = tor.iloc[1:].copy() if len(tor) > 1 else tor
    t['_mag'] = t[['TV距离', 'd_eta2', 'd_尾部捕获', 'd_翻转率']].abs().max(axis=1)
    t = t.sort_values('_mag', ascending=False)
    for _, r in t.iterrows():
        bar = '#' * max(1, int(r['_mag'] * 60))
        print(f"  {r['参数变体']:24s} |{bar:<30s}| TV={r['TV距离']:.3f} "
              f"dEta2={r['d_eta2']:+.4f} dCap={r['d_尾部捕获']:+.3f} dFlip={r['d_翻转率']:+.3f}")
    print("\n解读: TV 距离大 => 标签分布随参数剧变(阈值贴在密度高的区域);")
    print("      d_eta2/dCap 平坦 => 参数是噪声拟合, 可安全动态化; 断崖 => 脆弱, 优先滚动校准。")
    return tor


# =========================================================================
# 3. 主流程
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description='大盘五象限场景标签科学审计')
    ap.add_argument('--start', default='2021-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--source', choices=['parquet', 'rebuild'], default='parquet')
    ap.add_argument('--ew', action='store_true', help='叠加全市场等权域前瞻收益(慢)')
    ap.add_argument('--fixed', action='store_true', help='用固定阈值版打标做对照')
    ap.add_argument('--tornado', action='store_true', help='参数敏感性分析')
    ap.add_argument('--picks', default=None,
                    help='选股质量交叉验证, 传入 global_strategy_audit.csv 路径启用')
    ap.add_argument('--out', default=os.path.join('external_data', 'scenario_audit'))
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now()
    start = pd.Timestamp(args.start)
    use_dynamic = not args.fixed

    print(f"[scenario_audit] 区间 {start.date()} ~ {end.date()} | "
          f"阈值模式={'固定' if args.fixed else '动态v5'} | 广度源={args.source}")

    zzqz_df = prepare_zzqz()
    breadth_df = load_breadth(args.source)
    common = sorted(set(zzqz_df.index).intersection(breadth_df.index))
    common = [d for d in common if len([x for x in common if x <= d]) >= WARMUP_DAYS]
    eval_dates = [d for d in common if start <= d <= end]
    if len(eval_dates) < 100:
        raise SystemExit(f"[scenario_audit] 有效评估日不足: {len(eval_dates)}")

    total_stocks = None if use_dynamic else int(breadth_df['is_valid'].median())
    judge = is_market_ok.scenario_based_market_judgment
    print(f"[scenario_audit] 打标中... ({len(eval_dates)} 个交易日)")
    lab = label_days(eval_dates, zzqz_df, breadth_df, judge,
                     total_stocks=total_stocks, use_dynamic=use_dynamic)

    if args.ew:
        ew = build_ew_return_series(
            cache_path=os.path.join(args.out, 'ew_daily_returns.csv'))
        lab = attach_ew_forward(lab, ew)

    ev = event_study_table(lab, ew=args.ew)
    pairwise_separation(lab)
    stability_report(lab)
    yearly_breakdown(lab)
    baseline_comparison(lab)
    quota_consistency(lab)

    pq = None
    if args.picks:
        pq = pick_quality_by_scenario(args.picks,
                                      ew_cache=os.path.join(args.out, 'ew_daily_returns.csv'))

    tor = None
    if args.tornado:
        tor = run_tornado(eval_dates, zzqz_df, breadth_df, lab,
                          use_dynamic=use_dynamic, total_stocks=total_stocks)

    # 触发原因频次 (每场景 top5)
    banner("H. 触发原因频次 (每场景 Top5)")
    for sc in SCENARIOS:
        sub = lab[lab['primary_scenario'] == sc]['decision_reason']
        if len(sub) == 0:
            continue
        top5 = sub.value_counts().head(5)
        items = ' | '.join(f"{k.split('】')[-1]}:{v}" for k, v in top5.items())
        print(f"  {sc:12s}(n={len(sub):4d}): {items}")

    # 落盘
    os.makedirs(args.out, exist_ok=True)
    tag = 'fixed' if args.fixed else 'dynamic'
    p1 = os.path.join(args.out, f'labels_{tag}.csv')
    p2 = os.path.join(args.out, f'event_study_{tag}.csv')
    lab.to_csv(p1, encoding='utf-8-sig')
    ev.to_csv(p2, index=False, encoding='utf-8-sig')
    print(f"\n[scenario_audit] 明细已保存: {p1}")
    print(f"[scenario_audit] 汇总已保存: {p2}")
    if tor is not None:
        p3 = os.path.join(args.out, f'tornado_{tag}.csv')
        tor.to_csv(p3, index=False, encoding='utf-8-sig')
        print(f"[scenario_audit] tornado 已保存: {p3}")
    if pq is not None:
        p4 = os.path.join(args.out, f'picks_quality_{tag}.csv')
        pq.to_csv(p4, index=False, encoding='utf-8-sig')
        print(f"[scenario_audit] 选股质量已保存: {p4}")


if __name__ == '__main__':
    main()
