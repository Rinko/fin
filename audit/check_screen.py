# audit/check_screen.py
# 海选漏斗审计 (替代原 check_backtest.py)
# 说明: 原 check_backtest.py 依赖已不再产出的 individual_stocks_audit_filtered.csv, 已废弃。
# 本脚本改用 debug_inference_results.csv (DEBUG_INFERENCE=1 生成的全市场推理快照) +
# global_strategy_audit.csv (每日场景/门槛/top_x 决策), 与 backtest.py 的筛选手法对标。
import os
import re
import numpy as np
import pandas as pd

DEFAULT_DEBUG = 'debug_inference_results.csv'
DEFAULT_AUDIT = 'global_strategy_audit.csv'

# 与 backtest.py before_exec_fn 中 buy_quota 保持一致
QUOTA_BY_SCENE = {'bottom': 5, 'opportunity': 3, 'normal': 1, 'caution': 2, 'risk': 2}


def _normalize_symbols(board):
    if 'symbol' in board.columns:
        board['symbol'] = board['symbol'].astype(str).str.extract(r'(\d{6})', expand=False).fillna(board['symbol'])


def _load_audit(audit_path):
    audit = pd.read_csv(audit_path)
    # 注意: 不要 set_index 后按 index 做 merge —— pandas 2.3.0 对"曾是 index 的 datetime 列"
    # 重循环 merge 会静默失配 (0 匹配), 因此保留 dt 为普通列, 统一 right_on='dt'。
    audit['dt'] = pd.to_datetime(audit['index']).dt.normalize()
    return audit


def _scenario_row(audit, col):
    return audit[col].to_dict()


# =========================================================================
# 1. 漏斗分解 + 候选流动性画像 (与 backtest.py check_buy_eligibility_and_score 对标)
# =========================================================================
def funnel_and_liquidity_audit(debug_path=DEFAULT_DEBUG, audit_path=DEFAULT_AUDIT):
    print("\n" + "=" * 100)
    print(f"{'📊 海选漏斗分解 + 每日可得性 (对回测逻辑)':^100}")
    print("=" * 100)

    if not os.path.exists(debug_path) or not os.path.exists(audit_path):
        print(f"❌ 缺少 {debug_path} 或 {audit_path}，请先回测生成。")
        print("  >> 回测: DEBUG_INFERENCE=1 ... (或直接主流程 run_backtest)")
        return

    audit = _load_audit(audit_path)
    col_ok = [c for c in ['strat_daily_ml_threshold', 'strat_money_supply_signal',
                          'strat_congestion_too_high', 'strat_is_market_ok'] if c in audit.columns]

    avail_cols = pd.read_csv(debug_path, nrows=1).columns
    cols = [c for c in ['date', 'symbol', 'ml_rank', 'bias_20', 'is_profit_ok', 'amount_ma20', 'atr_ratio'] if c in avail_cols]
    df = pd.read_csv(debug_path, usecols=cols, dtype={'symbol': str})
    _normalize_symbols(df)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.merge(audit[['dt', 'strat_primary_scenario'] + col_ok], left_on='date', right_on='dt',
                  how='left', validate='many_to_one')

    # 场景阈值: 优先取回测真实值, 缺失回退到硬编码 (与 backtest.py scenario_map 一致)
    if 'strat_daily_ml_threshold' in df.columns:
        df['ml_thr'] = df['strat_daily_ml_threshold'].fillna(
            df['strat_primary_scenario'].map({'opportunity': 0.02, 'bottom': 0.03,
                                              'normal': 0.01, 'caution': 0.01, 'risk': 0.01}).fillna(0.01)
        )
    else:
        df['ml_thr'] = df['strat_primary_scenario'].map(
            {'opportunity': 0.02, 'bottom': 0.03, 'normal': 0.01, 'caution': 0.01, 'risk': 0.01}).fillna(0.01)

    # 每日流动性门槛 (与 backtest.py GLOBAL_SCREEN_THRESHOLDS: amount q30 / atr q20)
    df['liq_q30'] = df.groupby('date')['amount_ma20'].transform(lambda x: x.quantile(0.3))
    df['vol_q20'] = df.groupby('date')['atr_ratio'].transform(lambda x: x.quantile(0.2))

    # bias_con 判定 (与 backtest.py 一致)
    def bias_ok(row):
        sc = str(row['strat_primary_scenario'])
        if 'bottom' in sc:
            return row['bias_20'] < 0
        if 'opportunity' in sc:
            return row['bias_20'] > -0.05
        if 'normal' in sc:
            return row['bias_20'] > 0.05
        return True  # caution/risk 不设 bias 硬约束

    df['bias_ok'] = df.apply(bias_ok, axis=1)
    df['congestion_ok'] = ~df['strat_congestion_too_high'].fillna(False).astype(bool)
    df['market_ok'] = df['strat_is_market_ok'].fillna(False).astype(bool)
    df['money_ok'] = df['strat_money_supply_signal'].fillna(1.0) >= 0.3

    n_all = len(df)
    l1 = df[df['ml_rank'] < df['ml_thr']]
    l2 = l1[l1['is_profit_ok']]
    l3 = l2[(l2['amount_ma20'] >= l2['liq_q30']) & (l2['atr_ratio'] >= l2['vol_q20'])]
    l4 = l3[l3['bias_ok']]
    l5 = l4[l4['congestion_ok']]
    l6 = l5[l5['market_ok'] & l5['money_ok']]  # 最终合格候选 (对应 is_eligible)

    def pct(x, base=n_all):
        return f"{len(x):,} ({len(x)/base*100:.2f}%)"

    print(f"全市场推理样本: {n_all:,}")
    print(f"  层1 ml_rank < 场景门槛:          {pct(l1)}")
    print(f"  层2 + is_profit_ok(基本面):      {pct(l2)}")
    print(f"  层3 + is_active(流动性q30/q20):  {pct(l3)}")
    print(f"  层4 + bias_con(位置约束):        {pct(l4)}")
    print(f"  层5 + 非拥挤 (congestion_ok):    {pct(l5)}")
    print(f"  层6 + 大盘过线 & 货币过线:        {pct(l6)} (最终合格候选)")
    if len(l2):
        print(f"  is_active 层淘汰率: {(1-len(l3)/len(l2))*100:.1f}%   移动/拥挤层合计淘汰率: {(1-len(l6)/len(l4))*100:.1f}%")

    # 按场景通过率 (l2 → l6)
    print("\n" + "-" * 90)
    print(f"{'按场景漏斗通过率 (层2 → 层6)':^70}")
    for sc in ['opportunity', 'bottom', 'normal', 'caution', 'risk']:
        sub = l2[l2['strat_primary_scenario'] == sc]
        if len(sub) == 0:
            continue
        sub6 = l6[l6['strat_primary_scenario'] == sc]
        print(f"  {sc:12s}: 层2 {len(sub):6,} → 层6 {len(sub6):6,} | 通过率 {len(sub6)/len(sub)*100:5.1f}%")

    # 候选流动性画像
    print("\n" + "-" * 90)
    print(f"{'候选流动性画像 (amount_ma20, 万元)':^70}")
    for name, sub in [('层2 (ml+profit)', l2), ('层3 (过is_active)', l3), ('层6 (最终合格)', l6)]:
        if len(sub) == 0:
            continue
        print(f"  {name:22s}: 中位数 {sub['amount_ma20'].median()/1e4:>12,.0f} | "
              f"均值 {sub['amount_ma20'].mean()/1e4:>12,.0f} | "
              f"P10 {sub['amount_ma20'].quantile(0.1)/1e4:>12,.0f}")

    # 逐月通过率漂移
    print("\n" + "-" * 90)
    print(f"{'逐月最终通过率 (检测模型/流动性偏好漂移)':^70}")
    l2['ym'] = l2['date'].dt.to_period('M')
    l6_cnt = pd.Series(l6.set_index('date').index.to_period('M')).value_counts()
    rows = []
    for ym, grp in l2.groupby('ym'):
        c = int(l6_cnt.get(ym, 0))
        rows.append({'月份': str(ym), '层2': len(grp), '过全层': c, '通过率': f"{c/len(grp)*100:.1f}%" if len(grp) else "N/A"})
    mdf = pd.DataFrame(rows)
    if not mdf.empty:
        print(mdf.to_string(index=False))


# =========================================================================
# 2. 每日买入额度 (quota) 覆盖与空仓日审计
# =========================================================================
def quota_coverage_audit(audit_path=DEFAULT_AUDIT, debug_path=DEFAULT_DEBUG):
    print("\n" + "=" * 100)
    print(f"{'📈 每日买入额度 (quota) 覆盖与空仓占比审计':^100}")
    print("=" * 100)

    if not os.path.exists(audit_path):
        print("❌ 缺少 global_strategy_audit.csv")
        return

    audit = _load_audit(audit_path)

    # 场景出现天数
    scene_days = audit['strat_primary_scenario'].value_counts()
    print("\n[市场场景分布 (交易日)]:")
    print(scene_days.to_string())

    # market_ok 天占比
    if 'strat_is_market_ok' in audit.columns:
        ok = audit['strat_is_market_ok'].astype(bool)
        print(f"\n大盘过线 (is_market_ok=True) 天数: {ok.sum()} / {len(ok)} ({ok.mean()*100:.1f}%)")

    # top_x_buys 覆盖: 每天是否买满 quota (5/3/1/2)
    if 'strat_top_x_buys' not in audit.columns:
        print("❌ 无 strat_top_x_buys 列，跳过 quota 覆盖审计。")
        return

    def quota(sc):
        return QUOTA_BY_SCENE.get(sc, None)

    rows = []
    for _, row in audit.iterrows():
        sc = str(row['strat_primary_scenario'])
        q = quota(sc)
        if q is None:
            continue
        buys = row['strat_top_x_buys']
        n_buy = 0
        if isinstance(buys, str) and buys.strip():
            n_buy = len(set(re.findall(r'\d{6}', buys)))
        rows.append({'date': row['dt'], 'scenario': sc, 'quota': q, 'n_buy': n_buy})
    coverage = pd.DataFrame(rows)
    if coverage.empty:
        print("❌ 无有效每日决策记录。")
        return

    hyp = coverage['n_buy'] >= coverage['quota']
    print(f"\n总决策日: {len(coverage)}")
    print(f"买满 quota 天数: {hyp.sum()} ({hyp.mean()*100:.1f}%)")
    print(f"空仓日 (n_buy=0) : {(coverage['n_buy']==0).sum()} ({(coverage['n_buy']==0).mean()*100:.1f}%)")
    print("\n[按场景 quota 填满率]:")
    print(coverage.groupby('scenario').apply(lambda g: pd.Series({
        '天数': len(g),
        '平均买入': f"{g['n_buy'].mean():.2f}",
        '买满率': f"{((g['n_buy'] >= g['quota']).mean()*100):.1f}%",
        '空仓率': f"{((g['n_buy']==0).mean()*100):.1f}%"
    }), include_groups=False).to_string())

    # 连续空仓天数 (决策枯竭期) — 需 reconstruct 通过 debug 才能判定是缺候选还是禁入
    deficit = coverage[coverage['n_buy'] < coverage['quota']]
    if not deficit.empty:
        print(f"\n[额度缺口] {len(deficit)} 天未买满 quota (共 {deficit['quota'].sum()-deficit['n_buy'].sum()} 个空位)")
        print(f"  首个未买满: {deficit['date'].min().date()} | 最近未买满: {deficit['date'].max().date()}")


# =========================================================================
# 3. 选股对齐审计 (复刻原 check_backtest.py Section E / C)
# =========================================================================
def topx_alignment_audit(debug_path=DEFAULT_DEBUG, audit_path=DEFAULT_AUDIT):
    print("\n" + "=" * 100)
    print(f"{'🎯 最终买入席位与漏斗一致性验证 (漏网之鱼探测)':^100}")
    print("=" * 100)

    if not os.path.exists(audit_path) or not os.path.exists(debug_path):
        print("❌ 需 global_strategy_audit.csv + debug_inference_results.csv (DEBUG_INFERENCE=1 生成)。")
        return

    audit = _load_audit(audit_path)
    cols = ['date', 'symbol', 'ml_rank', 'bias_20', 'is_profit_ok', 'amount_ma20', 'atr_ratio']
    df = pd.read_csv(debug_path, usecols=cols, dtype={'symbol': str})
    _normalize_symbols(df)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.merge(audit[['dt', 'strat_primary_scenario', 'strat_daily_ml_threshold',
                         'strat_congestion_too_high', 'strat_is_market_ok',
                         'strat_money_supply_signal']],
                  left_on='date', right_on='dt', how='left')

    df['ml_thr'] = df['strat_daily_ml_threshold'].fillna(
        df['strat_primary_scenario'].map({'opportunity': 0.02, 'bottom': 0.03,
                                          'normal': 0.01, 'caution': 0.01, 'risk': 0.01}).fillna(0.01))
    df['liq_q30'] = df.groupby('date')['amount_ma20'].transform(lambda x: x.quantile(0.3))
    df['vol_q20'] = df.groupby('date')['atr_ratio'].transform(lambda x: x.quantile(0.2))

    def bias_ok(row):
        sc = str(row['strat_primary_scenario'])
        if 'bottom' in sc:
            return row['bias_20'] < 0
        if 'opportunity' in sc:
            return row['bias_20'] > -0.05
        if 'normal' in sc:
            return row['bias_20'] > 0.05
        return True

    df['bias_ok'] = df.apply(bias_ok, axis=1)
    elig = df[(df['ml_rank'] < df['ml_thr']) & df['is_profit_ok'] &
              (df['amount_ma20'] >= df['liq_q30']) & (df['atr_ratio'] >= df['vol_q20']) &
              df['bias_ok'] & (~df['strat_congestion_too_high'].fillna(False)) &
              df['strat_is_market_ok'].fillna(False) &
              (df['strat_money_supply_signal'].fillna(1.0) >= 0.3)]

    def quota(sc):
        if 'bottom' in sc:
            return 5
        if 'opportunity' in sc:
            return 3
        if 'normal' in sc:
            return 1
        return 2

    mismatch = 0
    checked = 0
    extra_buys = []  # top_x 中出现但不在 eligible 中的 (应为空)
    missed_top = []  # eligible top 中未被买入的 (quota 上限外可忽略)
    sample = []

    for _, row in audit.iterrows():
        buys = row.get('strat_top_x_buys', np.nan)
        if not isinstance(buys, str) or buys == 'nan' or not buys:
            continue
        buy_set = set()
        for tok in buys.split('|'):
            m = re.search(r'\d{6}', tok)
            if m and m.group(0) != 'nan':
                buy_set.add(m.group(0))
        if not buy_set:
            continue
        day = elig[elig['date'] == row['dt']]
        if day.empty:
            extra_buys.append((row['dt'].date(), buy_set, '无候选'))
            continue
        q = quota(str(row['strat_primary_scenario']))
        top_n = day.nsmallest(q, 'ml_rank')
        top_set = set(top_n['symbol'])
        mismatch += (buy_set - top_set) != set()
        checked += 1
        if buy_set != top_set and len(sample) < 5:
            sample.append((row['dt'].date(), '|'.join(sorted(buy_set)), '|'.join(sorted(top_set))))
        extra = buy_set - top_set
        if extra and len(extra_buys) < 5:
            extra_buys.append((row['dt'].date(), extra, f"已选: {sorted(top_set)[:3]}"))

    print(f"\n对齐检查天数: {checked}")
    print(f"不一致天数: {mismatch}")
    if sample:
        for dt, b, t in sample:
            print(f"  ⚠️ {dt} 买入 {b} | 顶分合格 {t}")
    if extra_buys:
        print("\n[买入但不是 top-quota 合格者/或无候选日期] (前 5):")
        for dt, b, note in extra_buys:
            print(f"  ⚠️ {dt} 买入 {sorted(b)} — {note}")
    else:
        print("✅ 买入席位 = 各日 eligible 中 ml_rank 最高的 quota 只，选股对齐通过 (无漏网之鱼)。")

    # 漏选分析: eligible 大于 quota 却只买部分的日子 (容量提示)
    over = elig.groupby('date').size()
    tight_days = over[over >= 2].index
    print(f"\n eligible 池≥2 只的天数: {len(tight_days)} (quota 约束下存在选股取舍，容量健康)")


def run_screen_audit():
    print("=" * 100)
    print(f"{' 🧭 海选漏斗与准入层审计 (Debug Inference + Global Audit) ':^100}")
    print("=" * 100)
    funnel_and_liquidity_audit()
    quota_coverage_audit()
    topx_alignment_audit()


if __name__ == "__main__":
    run_screen_audit()