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


def walkforward_predict(X, y_bad, y_up, eval_dates, mono=None):
    """季度走前预测 p_bad / p_up (仅用 <t-EMBARGO 信息训练)。
    mono: {'p_bad': 约束列表|None, 'p_up': ...} 与 X 列对齐的单调约束。"""
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
            params = dict(LGB_PARAMS)
            if mono and mono.get(col):
                assert len(mono[col]) == X.shape[1], '单调约束长度须等于特征数'
                params['monotone_constraints'] = list(mono[col])
            m = lgb.LGBMClassifier(**params)
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


def compare(labels_map):
    """labels_map: {显示名: labels_df}; 列动态扩展。"""
    names = list(labels_map)
    rows = []
    def add(name, vals, fmt="{:.3f}"):
        rows.append({'指标': name,
                     **{n: ("n/a" if pd.isna(v) else fmt.format(v)) for n, v in zip(names, vals)}})

    labs = list(labels_map.values())
    add('eta^2(fwd20)', [eta_squared(l['fwd20'], l['primary_scenario']) for l in labs], "{:.4f}")
    add('翻转率', [_flip_rate(l['primary_scenario']) for l in labs], "{:.1%}")
    add('<=3天短段占比', [_short_run_share(l['primary_scenario']) for l in labs], "{:.1%}")
    tails = [_tail_stats(l) for l in labs]
    add('尾部捕获 recall(risk|caution)', [t[0] for t in tails], "{:.1%}")
    add('尾部捕获 precision(risk|caution)', [t[1] for t in tails], "{:.1%}")
    for sc in ['bottom', 'opportunity', 'risk']:
        add(f'{sc} mean_fwd20', [_sc_mean_p(l, sc)[0] for l in labs], "{:+.2%}")
    add('bottom MWU p 值(vs 其余)', [_sc_mean_p(l, 'bottom')[1] for l in labs], "{:.3f}")

    banner("Champion-Challenger 并排对比 (同区间同口径)")
    print(pd.DataFrame(rows).to_string(index=False))
    base_name, base_lab = names[0], labs[0]
    for n, l in zip(names[1:], labs[1:]):
        d1 = base_lab['primary_scenario'].value_counts(normalize=True).to_dict()
        d2 = l['primary_scenario'].value_counts(normalize=True).to_dict()
        agree = (base_lab['primary_scenario'].reindex(l.index) == l['primary_scenario']).mean()
        print(f"  [{n}] vs {base_name}: 分布TV={tv_distance(d1, d2):.3f} | 逐日一致率={fmt_pct(agree)}")
    print("\n解读: 挑战者若在 eta^2/尾部捕获上占优且翻转率不升高, 即具备影子替换资格;")
    print("      正式切换前应将其接入 backtest 影子跑一轮完整组合回测再定。")
    for name, lab in labels_map.items():
        cnt = lab['primary_scenario'].value_counts()
        dist_s = ' '.join(f"{s}:{cnt.get(s, 0)}" for s in SCENARIOS)
        print(f"  [{name}] 分布: {dist_s}")


# =========================================================================
# 业务不变量验收 (标签层, 对所有标签器对称适用)
# =========================================================================
def biz_invariant_checks(lab, breadth_df, zzqz_df):
    """按业务经验对'标签'做不变量测试 (与模型内部无关):
    B1 拥挤尖峰: congestion > 过去500日Q99(与回测熔断同口径)的日子,
       被标 risk|caution 的比例应显著高于常态日 —— '拥挤过高=不健康'。
    B2 动量中性: mom20 极端五分位(高/低)日的场景分布不应被单一场景吸收
       —— '动量强单独不构成方向决断'。
    B3 踩踏广度: low20_ratio > 过去Q90 的日子, 不应集中标注 bottom/opportunity。
    返回 {测试名: (数值描述, PASS/FAIL/WARN)}。"""
    res = {}
    idx = lab.index

    # --- B1 拥挤尖峰 ---
    cong = breadth_df['congestion'].reindex(idx)
    thr99 = breadth_df['congestion'].rolling(500, min_periods=250).quantile(0.99).shift(1).reindex(idx)
    spike = (cong > thr99).fillna(False)
    flag = lab['primary_scenario'].isin(['risk', 'caution'])
    p_flag_spike = flag[spike].mean() if spike.sum() else np.nan
    p_flag_base = flag.mean()
    if spike.sum() >= 10:
        z = (p_flag_spike - p_flag_base) / np.sqrt(p_flag_base * (1 - p_flag_base) / spike.sum())
        res['B1 拥挤尖峰->防御标签'] = (
            f"P(防|尖峰)={fmt_pct(p_flag_spike)} vs 基率={fmt_pct(p_flag_base)} "
            f"(n={int(spike.sum())}, z={z:+.1f})",
            'PASS' if z > 1.645 else 'FAIL')
    else:
        res['B1 拥挤尖峰->防御标签'] = (f"尖峰样本不足(n={int(spike.sum())})", 'WARN')

    # --- B2 动量充分性 (业务口径: 动量'强'单独不构成方向决断。
    #     检验"充分性": 在无广度参与(up_ratio<0.5)的日子里, 高动量日的进攻判定
    #     占比不应显著高于其他日子 —— 即动量在缺乏独立证据时不得额外放行进攻;
    #     同时输出高动日场景分布供参考(现役树曾 76% 单边押注 opportunity)。---
    mom = zzqz_df['close'].pct_change(20).reindex(idx)
    mom_rank = mom.expanding(min_periods=250).rank(pct=True)
    top = (mom_rank >= 0.8).fillna(False)
    upr_all = breadth_df['up_ratio'].reindex(idx)
    nopart = (upr_all < 0.5).fillna(True)
    overall = lab['primary_scenario'].value_counts(normalize=True)
    sub_top = lab.loc[top, 'primary_scenario']
    info = ''
    if len(sub_top) >= 30:
        d = sub_top.value_counts(normalize=True).reindex(overall.index).fillna(0)
        info = f"[参考] mom高五分位分布: opp={fmt_pct(d.get('opportunity', 0), 0)} risk={fmt_pct(d.get('risk', 0), 0)}"
    cellA = top & nopart          # 高动量 & 无参与
    cellB = (~top) & nopart       # 非高动量 & 无参与
    opp = lab['primary_scenario'].eq('opportunity')
    nA, nB = int(cellA.sum()), int(cellB.sum())
    if nA >= 20 and nB >= 20:
        pA, pB = opp[cellA].mean(), opp[cellB].mean()
        p_pool = (opp[cellA].sum() + opp[cellB].sum()) / (nA + nB)
        z = (pA - pB) / np.sqrt(max(p_pool * (1 - p_pool) * (1 / nA + 1 / nB), 1e-12))
        res['B2 动量强->不定方向'] = (
            f"P(进|高动&无参与)={fmt_pct(pA)}(n={nA}) vs P(进|其余&无参与)={fmt_pct(pB)}(n={nB}) "
            f"z={z:+.1f} {info}",
            'PASS' if z < 1.645 else 'FAIL')
    else:
        res['B2 动量强->不定方向'] = (f"样本不足(A={nA},B={nB}) {info}", 'WARN')

    # --- B3 踩踏广度 ---
    l20r = breadth_df['low20_ratio'].reindex(idx)
    q90 = breadth_df['low20_ratio'].rolling(120, min_periods=30).quantile(0.9).shift(1).reindex(idx)
    stampede = (l20r > q90).fillna(False)
    bull_lab = lab['primary_scenario'].isin(['bottom', 'opportunity'])
    if stampede.sum() >= 10:
        p_bull_stamp = bull_lab[stampede].mean()
        p_bull_base = bull_lab.mean()
        res['B3 踩踏广度->禁进攻标签'] = (
            f"P(进|踩踏)={fmt_pct(p_bull_stamp)} vs 基率={fmt_pct(p_bull_base)}",
            'PASS' if p_bull_stamp <= p_bull_base * 1.1 else 'FAIL')
    else:
        res['B3 踩踏广度->禁进攻标签'] = (f"样本不足(n={int(stampede.sum())})", 'WARN')
    return res


def print_biz_checks(checks_map):
    banner("业务不变量验收 (标签层; 判据来自业务经验, 对各标签器对称)")
    names = list(checks_map)
    tests = list(next(iter(checks_map.values())).keys())
    hdr = f"{'测试':26s} " + " ".join(f"{n:>22s}" for n in names)
    print(hdr)
    for t in tests:
        cells = []
        for n in names:
            desc, verdict = checks_map[n][t]
            cells.append(f"[{verdict}] {desc:>16s}")
        print(f"{t:26s} " + " ".join(f"{c:>24s}" for c in cells))
    print("\n说明: B1/B3 为单向假设检验(业务方向明确); B2 为中性检验(动量单独不定方向)。")


def build_hinge_features(X):
    """铰链特征改造: 业务效应是分段形(危险区在'过高段')而非全局单调。
    以因果锚点(过去250日中位数/分位, shift1 防泄漏)切出铰链段,
    仅对铰链段施加单调约束 —— '业务定边界与方向, 段内数据说话'。"""
    Xh = X.copy()

    def hinge(s, base, direction):
        # direction=+1: 取超出锚点的上段; -1: 取低于锚点的下段
        return ((s - base) if direction > 0 else (base - s)).clip(lower=0)

    cong = Xh['congestion']
    base_c = cong.shift(1).rolling(250, min_periods=60).median()
    Xh['cong_excess'] = hinge(cong, base_c, +1)   # 拥挤过高段
    Xh['cong_gap'] = hinge(cong, base_c, -1)      # 拥挤崩塌段
    c20 = Xh['congestion_ma20']
    b20 = c20.shift(1).rolling(250, min_periods=60).median()
    Xh['cong20_excess'] = hinge(c20, b20, +1)
    Xh['cong20_gap'] = hinge(c20, b20, -1)
    mom = Xh['zz_mom20']
    q80 = mom.shift(1).rolling(250, min_periods=60).quantile(0.8)
    q20 = mom.shift(1).rolling(250, min_periods=60).quantile(0.2)
    Xh['mom_hi'] = hinge(mom, q80, +1)            # 动量极端上段 (约束置0: 单独不定方向)
    Xh['mom_lo'] = hinge(mom, q20, -1)

    Xh = Xh.drop(columns=['congestion', 'congestion_ma20', 'zz_mom20'])
    BAD_POS = ['cong_excess', 'cong_gap', 'cong20_excess', 'cong20_gap',
               'low20_ratio', 'mkt_vol', 'zz_range20']
    mono_bad = [1 if c in BAD_POS else 0 for c in Xh.columns]
    mono_up = [-v for v in mono_bad]
    return Xh, mono_bad, mono_up


# =========================================================================
# 解释性审计: 模型预测核心是否可被业务逻辑解释
# =========================================================================
FEATURE_GLOSSARY = {
    'up_ratio': '普涨率(上涨家数占比)', 'high10_ratio': '创10日新高占比',
    'high20_ratio': '创20日新高占比', 'high60_ratio': '创60日新高占比',
    'low10_ratio': '创10日新低占比', 'low20_ratio': '创20日新低占比(踩踏广度)',
    'low60_ratio': '创60日新低占比', 'high_v': '新高家数一阶变化(动能)',
    'low_v': '新低家数一阶变化', 'low_a_smooth': '新低加速度(平滑)',
    'high_a_smooth': '新高加速度(平滑)', 'high_ratio': '新高强度(vs 5日均线)',
    'low_ratio': '新低强度(vs 5日均线)', 'congestion': '成交额Top5%集中度(拥挤度)',
    'congestion_ma20': '拥挤度20日均线', 'congestion_bias': '拥挤度偏离',
    'mkt_trend': '大盘趋势因子(252日分位)', 'mkt_vol': '大盘波动因子(252日分位)',
    'mkt_liq': '大盘流动性因子(252日分位)', 'mkt_bias': '大盘乖离因子(252日分位)',
    'mkt_position': '大盘位置因子(252日分位)', 'mkt_ret_20': '大盘近20日收益',
    'mkt_ret_60': '大盘近60日收益',
    'zz_bias10': '指数乖离MA10', 'zz_bias60': '指数乖离MA60',
    'zz_mom20': '指数20日动量', 'zz_ret5': '指数5日收益',
    'zz_vol_ratio': '量能5/20比', 'zz_range20': '20日振幅区间',
}


def explain_heads(X, y_bad, y_up, end_date, top_n=8):
    """内训全历史(仅用于解读, 不作绩效主张), 回答'模型在看什么':
    gain 重要性 + TreeSHAP 贡献占比 + 方向(Spearman) + 分位响应曲线。"""
    train = X[X.index <= pd.Timestamp(end_date)]
    ok = train.notna().all(axis=1)
    train = train[ok]
    feats = list(train.columns)

    for head_name, y in (('p_bad (风险头: fwd20<=-5%)', y_bad), ('p_up (上行头: fwd20>0)', y_up)):
        yt = y.reindex(train.index).dropna().astype(int)
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(train.loc[yt.index], yt)
        booster = m.booster_
        logit = booster.predict(train.loc[yt.index].values, pred_contrib=True)
        contrib = np.abs(logit[:, :-1]).mean(axis=0)
        contrib_share = contrib / max(contrib.sum(), 1e-9)
        gain = booster.feature_importance(importance_type='gain')
        gain_share = gain / max(gain.sum(), 1e-9)
        proba = m.predict_proba(train.loc[yt.index])[:, 1]

        rows = []
        for i in np.argsort(-contrib_share)[:top_n]:
            f = feats[i]
            rho = stats.spearmanr(train.loc[yt.index, f], logit[:, i]).statistic
            # 分位响应曲线: 特征五分位 -> 平均预测概率
            qb = pd.qcut(train.loc[yt.index, f].to_numpy(), 5, labels=False, duplicates='drop')
            curve = pd.Series(np.asarray(proba)).groupby(qb).mean().values
            rows.append({'特征': f, '业务含义': FEATURE_GLOSSARY.get(f, '?'),
                         'gain%': f"{gain_share[i] * 100:.1f}",
                         'SHAP%': f"{contrib_share[i] * 100:.1f}",
                         '方向': '+' if rho > 0 else '-',
                         'Q1->Q5响应': ' '.join(f"{v * 100:.0f}%" for v in curve)})
        banner(f"解释性审计 — {head_name} (截至 {pd.Timestamp(end_date).date()}, 内训仅作解读)")
        print(pd.DataFrame(rows).to_string(index=False))
        print("  方向列 = 该特征与预测logit的Spearman符号; Q1->Q5 = 特征从低到高五分位的平均预测概率\n")
    print("[升级选项] 单调约束版: 将业务先验写为硬约束(如 low20_ratio↑=>p_bad↑, high20_ratio↑=>p_up↑,\n"
          "      congestion↑=>p_bad↑), 使模型'不可违反业务常识'成为构造性保证而非事后检验。")


def apply_business_gates(lab, preds, breadth_df, zzqz_df):
    """纯模型 + 业务公理门控 (映射层安全底线, 预测核心不动):
    G1 拥挤尖峰(cong>过去500日Q99) => 强制防御, 按 p_bad 相对分位选 risk/caution
    G2 opportunity 需广度参与确认(up_ratio>=0.5) —— 动量不单独决断
    G3 踩踏广度(low20_ratio>过去120日Q90) => 禁攻(bottom/opportunity -> caution)
    返回 (gated_lab, 门控触发统计)。"""
    idx = lab.index
    out = lab.copy()
    cong = breadth_df['congestion'].reindex(idx)
    thr99 = breadth_df['congestion'].rolling(500, min_periods=250).quantile(0.99).shift(1).reindex(idx)
    g1 = (cong > thr99).fillna(False)
    l20r = breadth_df['low20_ratio'].reindex(idx)
    q90 = breadth_df['low20_ratio'].rolling(120, min_periods=30).quantile(0.9).shift(1).reindex(idx)
    g3 = (l20r > q90).fillna(False)
    upr = breadth_df['up_ratio'].reindex(idx)
    pb = preds['p_bad'].reindex(idx)
    pb_q70 = pb.expanding(min_periods=PAST_Q_MIN).quantile(0.70).shift(1).reindex(idx)
    above_ma = (zzqz_df['close'] >= zzqz_df['close_ma60']).reindex(idx).fillna(False)

    hits = {'G1拥挤防御': 0, 'G2参与不足': 0, 'G3踩踏禁攻': 0}
    for d in idx:
        s = out.at[d, 'primary_scenario']
        new, tags = s, []
        if bool(g3.at[d]) and s in ('bottom', 'opportunity'):
            new, _ = 'caution', tags.append('G3踩踏禁攻')
        if bool(g1.at[d]):
            want = 'risk' if (pd.notna(pb_q70.at[d]) and pb.at[d] >= pb_q70.at[d]) else 'caution'
            if new != want:
                new = want
                tags.append('G1拥挤防御')
        if new == 'opportunity' and not (pd.notna(upr.at[d]) and upr.at[d] >= 0.5):
            new = 'normal' if bool(above_ma.at[d]) else 'caution'
            tags.append('G2参与不足')
        for t in tags:
            hits[t] += 1
        if new != s:
            out.at[d, 'primary_scenario'] = new
            out.at[d, 'position_multiplier'] = MULTIPLIER[new]
            out.at[d, 'is_market_ok'] = new != 'risk'
            base_reason = str(out.at[d, 'decision_reason']).replace('[shadow] ', '')
            out.at[d, 'decision_reason'] = f"{base_reason}+{'/'.join(tags)}"
    print(f"[gates] 门控触发: {hits}")
    return out


def main():
    ap = argparse.ArgumentParser(description='五象限场景标签挑战者对比')
    ap.add_argument('--start', default='2021-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--explain', action='store_true', help='输出驱动因子解释性审计')
    ap.add_argument('--hinge', action='store_true',
                    help='追加铰链特征+分段单调约束变体 (业务定边界, 段内数据说话)')
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
    gate = apply_business_gates(cha.copy(), preds, breadth_df, zzqz_df)

    judge = is_market_ok.scenario_based_market_judgment
    print("[challenger] 现役基线打标中...")
    inc = label_days(eval_dates, zzqz_df, breadth_df, judge,
                     total_stocks=None, use_dynamic=True)

    labels_map = {'现役决策树': inc, '自由LGBM': cha, '门控LGBM(参照)': gate}
    out_extra = {'labels_challenger_gated.csv': gate}

    if args.hinge:
        Xh, mono_bad, mono_up = build_hinge_features(X)
        print("[challenger] 铰链约束变体走前拟合中...")
        preds_h = walkforward_predict(Xh, y_bad, y_up, eval_dates,
                                      mono={'p_bad': mono_bad, 'p_up': mono_up})
        chh = map_scenarios(preds_h, zzqz_df)
        chh = chh[chh.index.isin(eval_dates)]
        labels_map['铰链约束LGBM'] = chh
        out_extra['labels_challenger_hinge.csv'] = chh

    compare(labels_map)

    checks_map = {n: biz_invariant_checks(l, breadth_df, zzqz_df)
                  for n, l in labels_map.items()}
    print_biz_checks(checks_map)

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, 'labels_challenger.csv')
    cha.to_csv(p, encoding='utf-8-sig')
    print(f"\n[challenger] 挑战者标签已保存: {p}")
    for fname, lab_df in out_extra.items():
        pf = os.path.join(args.out, fname)
        lab_df.to_csv(pf, encoding='utf-8-sig')
        print(f"[challenger] 变体标签已保存: {pf}")

    if args.explain:
        explain_heads(X, y_bad, y_up, end_date=eval_dates[-1])


if __name__ == '__main__':
    main()
