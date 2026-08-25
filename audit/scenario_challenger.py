# audit/scenario_challenger.py
# 五象限场景标签挑战者 (Phase 3 champion-challenger)
#
# 思路: 用监督学习替代手工决策树 —— 以 market_context_cache.parquet 的时点特征
#   (广度/动量/拥挤度/大盘因子) 预测 T+20 前瞻收益的两个二元头:
#     p_bad = P(fwd20 <= -5%)   (风险头)
#     p_up  = P(fwd20 > 0)      (上行头)
#   走前(expanding)按季重训, 训练窗与测试窗之间留 21 个交易日 embargo 防泄漏。
#   场景映射只用因果(过去分布)分位点作切点, 优先级与现行决策树一致,
#   对外接口与 is_market_ok.scenario_based_market_judgment 完全同构。
#
# 评估: 与现役标签在同一套指标下对比 (eta^2 / 翻转率 / 持续段 / 尾部捕获 /
#   分场景前瞻收益显著性), 输出并排报告; 标签落盘供回测影子验证。
#
# 用法:
#   python run.py audit challenger                 # 默认 2021-01-01 至今
#   python run.py audit challenger --start 2019-01-01
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

from audit.check_scenario import (
    SCENARIOS, BAD_TAIL, HORIZONS, PARQUET_PATH, ZZQZ_PATH,
    prepare_zzqz, load_breadth, label_days, banner, fmt_pct, eta_squared,
    tv_distance, is_market_ok)

MULTIPLIER = {'risk': 0.0, 'bottom': 1.2, 'opportunity': 0.8,
              'caution': 0.7, 'normal': 0.5}
MIN_TRAIN = 756            # 最少训练样本(交易日, 约 3 年)
EMBARGO = 21               # 训练/测试隔离带(>= 前瞻窗口)
RETRAIN_FREQ = 'Q'         # 按季重训
PAST_Q_MIN = 250           # 因果分位切点的最少历史

FEATURES_BREADTH = [
    'up_ratio', 'high10_ratio', 'high20_ratio', 'high60_ratio',
    'low10_ratio', 'low20_ratio', 'low60_ratio',
    'high_v', 'low_v', 'low_a_smooth', 'high_a_smooth',
    'high_ratio', 'low_ratio',
    'congestion', 'congestion_ma20', 'congestion_bias',
    'mkt_trend', 'mkt_vol', 'mkt_liq', 'mkt_bias',
    'mkt_position', 'mkt_ret_20', 'mkt_ret_60',
]
LGB_PARAMS = dict(objective='binary', n_estimators=250, learning_rate=0.05,
                  num_leaves=15, min_child_samples=50, colsample_bytree=0.8,
                  reg_lambda=5.0, random_state=42, n_jobs=4, verbose=-1,
                  deterministic=True, force_row_wise=True)


def build_features(breadth_df, zzqz_df):
    """时点特征矩阵: parquet 广度/因子列 + zzqz 量价衍生列 (全部 <=T 信息)。"""
    feat_cols = [c for c in FEATURES_BREADTH if c in breadth_df.columns]
    X = breadth_df[feat_cols].astype(float).copy()
    close = zzqz_df['close']
    vol = zzqz_df['volume']
    derived = pd.DataFrame({
        'zz_bias10': close / close.rolling(10).mean() - 1.0,
        'zz_bias60': close / close.rolling(60).mean() - 1.0,
        'zz_mom20': close.pct_change(20),
        'zz_ret5': close.pct_change(5),
        'zz_vol_ratio': vol.rolling(5).mean() / vol.rolling(20).mean() - 1.0,
        'zz_range20': close.rolling(20).max() / close.rolling(20).min() - 1.0,
    }).reindex(X.index)
    X = pd.concat([X, derived], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def walkforward_predict(X, y_bad, y_up, eval_dates):
    """季度走前预测 p_bad / p_up (仅用 <t-EMBARGO 信息训练)。"""
    cal = X.index
    pos = {d: i for i, d in enumerate(cal)}
    preds = pd.DataFrame(index=cal, columns=['p_bad', 'p_up'], dtype=float)

    groups = {}
    for i, d in enumerate(eval_dates):
        groups.setdefault((d.year, d.quarter), []).append(d)

    n_fit = 0
    for key in sorted(groups):
        test_dates = groups[key]
        t0 = test_dates[0]
        end_pos = pos[t0] - EMBARGO
        if end_pos < MIN_TRAIN:
            continue
        train_idx = slice(0, end_pos + 1)
        Xtr = X.iloc[train_idx]
        ok = Xtr.notna().all(axis=1) & y_bad.notna().reindex(Xtr.index, fill_value=False)
        Xtr = Xtr[ok]
        if len(Xtr) < MIN_TRAIN:
            continue
        for col, y in (('p_bad', y_bad), ('p_up', y_up)):
            ytr = y.reindex(Xtr.index).dropna().astype(int)
            m = lgb.LGBMClassifier(**LGB_PARAMS)
            m.fit(Xtr.loc[ytr.index], ytr)
            Xt = X.loc[[d for d in test_dates if d in pos]]
            ok_t = Xt.notna().all(axis=1)
            p = pd.Series(np.nan, index=Xt.index)
            if ok_t.any():
                p[ok_t] = m.predict_proba(Xt[ok_t])[:, 1]
            preds.loc[p.index, col] = p
        n_fit += 1
    print(f"[challenger] 完成走前拟合 {n_fit} 个季度 x 2 头")
    return preds.dropna(how='all')


def causal_quantile(s, q):
    """过去分布的 expanding 分位点 (shift(1) 保证不含当日)。"""
    return s.expanding(min_periods=PAST_Q_MIN).quantile(q).shift(1)


def map_scenarios(preds, zzqz_df):
    """因果切点 -> 五象限; 决策树优先级与现役一致。"""
    pb, pu = preds['p_bad'], preds['p_up']
    pb_hi = causal_quantile(pb, 0.90)
    pb_mid = causal_quantile(pb, 0.70)
    pu_hi = causal_quantile(pu, 0.80)
    ma60 = zzqz_df['close_ma60'].reindex(preds.index)
    close = zzqz_df['close'].reindex(preds.index)
    above_ma = (close >= ma60).fillna(False)

    out = []
    cur = None
    run = 0
    for d in preds.index:
        b, u = pb.get(d), pu.get(d)
        bh, bm, uh = pb_hi.get(d), pb_mid.get(d), pu_hi.get(d)
        if any(pd.isna(v) for v in (b, u, bh, bm, uh)):
            sc = 'normal'
            reason = '切点预热期默认'
        elif b >= bh:
            sc, reason = 'risk', f'p_bad={b:.3f}>=pastQ90({bh:.3f})'
        elif u >= uh and not above_ma.loc[d]:
            sc, reason = 'bottom', f'p_up={u:.3f}>=pastQ80({uh:.3f}) 且 MA60 下方(左侧修复)'
        elif u >= uh:
            sc, reason = 'opportunity', f'p_up={u:.3f}>=pastQ80({uh:.3f}) 且 MA60 上方(右侧)'
        elif b >= bm or not above_ma.loc[d]:
            sc = 'caution'
            reason = (f'p_bad={b:.3f}>=pastQ70({bm:.3f})' if b >= bm else 'MA60 下方')
        else:
            sc, reason = 'normal', '常态'
        mult = MULTIPLIER[sc]
        out.append({'date': d, 'primary_scenario': sc,
                    'position_multiplier': mult,
                    'is_market_ok': sc != 'risk',
                    'decision_reason': f'[challenger] {reason}'})
    lab = pd.DataFrame(out).set_index('date')

    close_al = zzqz_df['close'].reindex(lab.index)
    prev20 = zzqz_df['close'].shift(20).reindex(lab.index)
    ma60_al = zzqz_df['close_ma60'].reindex(lab.index)
    for k in HORIZONS:
        lab[f'fwd{k}'] = close_al.shift(-k) / close_al - 1.0
    lab['ma60_up'] = (close_al >= ma60_al).astype(int)
    lab['mom20_pos'] = (close_al / prev20 - 1 > 0).astype(int)
    return lab


# =========================================================================
# 对比评估
# =========================================================================
def _flip_rate(seq):
    return (seq != seq.shift(1)).iloc[1:].mean()


def _short_run_share(seq):
    run_id = (seq != seq.shift()).cumsum()
    runs = seq.groupby(run_id).size()
    return (runs <= 3).mean()


def _tail_stats(lab):
    bad = lab['fwd20'] <= BAD_TAIL
    flag = lab['primary_scenario'].isin(['risk', 'caution'])
    recall = (bad & flag).sum() / max(int(bad.sum()), 1)
    prec = (bad & flag).sum() / max(int(flag.sum()), 1)
    return recall, prec


def _sc_mean_p(lab, sc):
    a = lab.loc[lab['primary_scenario'] == sc, 'fwd20'].dropna()
    r = lab.loc[lab['primary_scenario'] != sc, 'fwd20'].dropna()
    if len(a) < 5 or len(r) < 5:
        return np.nan, np.nan
    return a.mean(), stats.mannwhitneyu(a, r, alternative='two-sided').pvalue


def compare(inc, cha):
    rows = []
    def add(name, v1, v2, fmt="{:.3f}"):
        rows.append({'指标': name,
                     '现役决策树': "n/a" if pd.isna(v1) else fmt.format(v1),
                     '挑战者LGBM': "n/a" if pd.isna(v2) else fmt.format(v2)})

    add('eta^2(fwd20)', eta_squared(inc['fwd20'], inc['primary_scenario']),
        eta_squared(cha['fwd20'], cha['primary_scenario']), "{:.4f}")
    add('翻转率', _flip_rate(inc['primary_scenario']), _flip_rate(cha['primary_scenario']), "{:.1%}")
    add('<=3天短段占比', _short_run_share(inc['primary_scenario']),
        _short_run_share(cha['primary_scenario']), "{:.1%}")
    r1, p1 = _tail_stats(inc)
    r2, p2 = _tail_stats(cha)
    add('尾部捕获 recall(risk|caution)', r1, r2, "{:.1%}")
    add('尾部捕获 precision(risk|caution)', p1, p2, "{:.1%}")
    for sc in ['bottom', 'opportunity', 'risk']:
        m1, _ = _sc_mean_p(inc, sc)
        m2, _ = _sc_mean_p(cha, sc)
        add(f'{sc} mean_fwd20', m1, m2, "{:+.2%}")
    _, pm1 = _sc_mean_p(inc, 'bottom')
    _, pm2 = _sc_mean_p(cha, 'bottom')
    add('bottom MWU p 值(vs 其余)', pm1, pm2, "{:.3f}")

    dist1 = inc['primary_scenario'].value_counts(normalize=True).to_dict()
    dist2 = cha['primary_scenario'].value_counts(normalize=True).to_dict()
    agree = (inc['primary_scenario'].reindex(cha.index)
             == cha['primary_scenario']).mean()

    banner("Champion-Challenger 并排对比 (同区间同口径)")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n场景分布 TV 距离: {tv_distance(dist1, dist2):.3f} | 两标签逐日一致率: {fmt_pct(agree)}")
    print("\n解读: 挑战者若在 eta^2/尾部捕获上占优且翻转率不升高, 即具备影子替换资格;")
    print("      正式切换前应将其接入 backtest 影子跑一轮完整组合回测再定。")
    for name, lab in (('现役', inc), ('挑战者', cha)):
        cnt = lab['primary_scenario'].value_counts()
        dist_s = ' '.join(f"{s}:{cnt.get(s, 0)}" for s in SCENARIOS)
        print(f"  [{name}] 分布: {dist_s}")


def main():
    ap = argparse.ArgumentParser(description='五象限场景标签挑战者对比')
    ap.add_argument('--start', default='2021-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--out', default=os.path.join('external_data', 'scenario_audit'))
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now()
    start = pd.Timestamp(args.start)
    print(f"[challenger] 区间 {start.date()} ~ {end.date()}")

    zzqz_df = prepare_zzqz()
    breadth_df = load_breadth('parquet')

    X = build_features(breadth_df, zzqz_df)
    fwd20_full = zzqz_df['close'].shift(-20) / zzqz_df['close'] - 1.0
    y_bad = (fwd20_full <= BAD_TAIL).astype(float)
    y_bad[fwd20_full.isna()] = np.nan
    y_up = (fwd20_full > 0).astype(float)
    y_up[fwd20_full.isna()] = np.nan

    common = sorted(set(zzqz_df.index).intersection(breadth_df.index).intersection(X.dropna().index))
    common = [d for i, d in enumerate(common) if i >= 120]
    eval_dates = [d for d in common if start <= d <= end]
    if len(eval_dates) < 100:
        raise SystemExit(f"[challenger] 有效评估日不足: {len(eval_dates)}")

    preds = walkforward_predict(X, y_bad, y_up, eval_dates)
    cha = map_scenarios(preds, zzqz_df)
    cha = cha[cha.index.isin(eval_dates)]

    judge = is_market_ok.scenario_based_market_judgment
    print("[challenger] 现役基线打标中...")
    inc = label_days(eval_dates, zzqz_df, breadth_df, judge,
                     total_stocks=None, use_dynamic=True)

    compare(inc, cha)

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, 'labels_challenger.csv')
    cha.to_csv(p, encoding='utf-8-sig')
    print(f"\n[challenger] 挑战者标签已保存: {p}")


if __name__ == '__main__':
    main()
