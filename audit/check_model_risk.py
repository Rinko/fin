# audit/check_model_risk.py
# 风控模型 + 入场/风控协同审计 (原 ml_check_sell.py)
# 现役: chip_accumulation_v6_newfeat.pkl (29 特征) + chip_risk_model_v1_newfeat.pkl (32 特征)
# 注意: 风控模型绑定审计数据 chip_risk_model_v1_newfeat_data.csv 若缺失, 将降级用入场审计数据
#       做风控打分 (32 个风控特征均在入场数据中存在), 此时风控 IC/分箱维度自动跳过。
import os
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

BUY_MODEL_DEFAULT = 'chip_accumulation_v6_newfeat.pkl'
RISK_MODEL_DEFAULT = 'chip_risk_model_v1_newfeat.pkl'
BUY_DATA_FALLBACK = 'model_data.csv'
RISK_DATA_FALLBACK = 'model_risk_data.csv'
OOS_START = '2020-01-01'


def _log(msg):
    logging.info(msg)
    print(msg)


def _read(data_path, usecols):
    # 全量读取，禁止截断（口径统一原则）
    return pd.read_csv(data_path, usecols=usecols)


def run_synergy_audit(buy_model_path=BUY_MODEL_DEFAULT, risk_model_path=RISK_MODEL_DEFAULT,
                      buy_data_path=BUY_DATA_FALLBACK, risk_data_path=RISK_DATA_FALLBACK):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("\n" + "=" * 120)
    print(f"{'入场 x 风控 协同与避雷效果审计 (现役 newfeat)':^120}")
    print("=" * 120)

    # ==========================================================================
    # 1. 加载模型与预测
    # ==========================================================================
    _log("加载模型并生成预测分...")
    buy_pkg = joblib.load(buy_model_path)
    b_model, b_features = buy_pkg['model'], buy_pkg['features']

    buy_df = None
    if 'data_file' in buy_pkg and buy_pkg['data_file'] and os.path.exists(buy_pkg['data_file']):
        buy_data_path = buy_pkg['data_file']
        _log(f"入场模型绑定数据: {buy_data_path}")
    if not os.path.exists(buy_data_path):
        logging.error(f"入场审计数据不存在: {buy_data_path}"); return
    entry_cols = pd.read_csv(buy_data_path, nrows=1).columns

    # --- 风控数据解析: 缺绑定文件则降级 ---
    risk_pkg = joblib.load(risk_model_path)
    r_model, r_features = risk_pkg['model'], risk_pkg['features']
    risk_data_path = risk_pkg.get('data_file', risk_data_path)
    risk_data_missing = not (risk_data_path and os.path.exists(risk_data_path))
    if risk_data_missing:
        risk_avail = [f for f in r_features if f in entry_cols]
        if len(risk_avail) == len(r_features):
            print("⚠️ WARN: 风控审计数据缺失, 降级用入场审计数据做风控打分 (风控 IC/分箱维度跳过)。")
            risk_data_path = buy_data_path
        else:
            logging.error(f"风控特征 {len(r_features)-len(risk_avail)} 个不在入场数据中, 无法降级。"); return
    else:
        _log(f"风控绑定数据: {risk_data_path}")

    # 读入场数据
    buy_need = ['date', 'symbol', 'target', 'target_val', 'close', 'change_pct'] + list(b_features)
    buy_need = [c for c in set(buy_need) if c in entry_cols]
    buy_df = _read(buy_data_path, buy_need)
    buy_df['date'] = pd.to_datetime(buy_df['date'], format='mixed', errors='coerce')
    buy_df['pred'] = b_model.predict(buy_df[b_features])
    buy_df = buy_df[buy_df['date'] >= pd.Timestamp(OOS_START)].copy()

    # 读风控数据 (若降级则用同一个 buy_df)
    risk_cols = pd.read_csv(risk_data_path, nrows=1).columns
    risk_need = ['date', 'symbol'] + list(r_features)
    if not risk_data_missing and 'risk_score' in risk_cols:
        risk_need.append('risk_score')
    risk_need = [c for c in set(risk_need) if c in risk_cols]
    risk_df = _read(risk_data_path, risk_need)
    risk_df['date'] = pd.to_datetime(risk_df['date'], format='mixed', errors='coerce')
    risk_df['pred_risk'] = r_model.predict(risk_df[r_features])
    risk_df = risk_df[risk_df['date'] >= pd.Timestamp(OOS_START)].copy()

    if buy_df.empty or risk_df.empty:
        print("❌ OOS 段为空，跳过。")
        return

    # 合并
    combined = buy_df[['date', 'symbol', 'pred', 'target', 'target_val']].merge(
        risk_df[['date', 'symbol', 'pred_risk'] + (['risk_score'] if 'risk_score' in risk_df.columns else [])],
        on=['date', 'symbol'], how='inner')
    has_risk_target = 'risk_score' in combined.columns
    del buy_df, risk_df

    print(f"合并样本: {len(combined):,} | 日期: {combined['date'].min().date()} ~ {combined['date'].max().date()} | "
          f"风控目标可用: {has_risk_target}")

    # ==========================================================================
    # 2. 特征重要性
    # ==========================================================================
    print("\n" + "=" * 120 + f"\n{'1. 特征重要性 (入场 vs 风控)':^120}\n" + "=" * 120)
    def imp(model, fts):
        s = pd.DataFrame({'Feature': fts, 'Importance': model.feature_importances_})
        s['Pct'] = s['Importance'] / s['Importance'].sum()
        return s.sort_values('Importance', ascending=False).head(10)
    print("[入场模型 Top10]:")
    print(imp(b_model, b_features).to_string(index=False))
    print("[风控模型 Top10]:")
    print(imp(r_model, r_features).to_string(index=False))

    # ==========================================================================
    # 3. 核心 RankIC (进场 / 风控)
    # ==========================================================================
    print("\n" + "=" * 120 + f"\n{'2. 核心预测表现 (OOS RankIC)':^120}\n" + "=" * 120)
    def ic_by_day(df, p, t):
        return df.groupby('date').apply(
            lambda g: spearmanr(g[p], g[t])[0] if (len(g) > 20 and g[p].std() > 1e-8 and g[t].std() > 1e-8) else np.nan,
            include_groups=False).mean()
    buy_ic = ic_by_day(combined, 'pred', 'target')
    print(f"入场 (Alpha) OOS RankIC: {buy_ic:.4f}")
    if has_risk_target:
        risk_ic = ic_by_day(combined, 'pred_risk', 'risk_score')
        print(f"风控 (Risk)  OOS RankIC: {risk_ic:.4f} (预期为负: 越负越险)")
    else:
        print("风控 RankIC: 跳过 (无 risk_score 目标)")

    # ==========================================================================
    # 4. 分箱单调性
    # ==========================================================================
    print("\n" + "=" * 120 + f"\n{'3. 分箱单调性审计 (Decile)':^120}\n" + "=" * 120)
    def decile_stats(df, p, v):
        df = df.copy()
        df['bin'] = df.groupby('date')[p].transform(
            lambda x: pd.qcut(x + np.random.uniform(0, 1e-12, len(x)), 10, labels=False, duplicates='drop')
            if x.nunique() >= 10 else np.nan)
        df = df.dropna(subset=['bin'])
        d = df[v].groupby([df['date'], df['bin']]).mean().to_frame(name='v').reset_index()
        return d.groupby('bin')['v'].mean().T
    print("入场分箱 (target 20日收益):")
    print(decile_stats(combined, 'pred', 'target_val').round(4))
    if has_risk_target:
        print("风控分箱 (risk_score, 越负越险):")
        print(decile_stats(combined, 'pred_risk', 'risk_score').round(4))

    # ==========================================================================
    # 5. 协同策略 (剔除风险最高前10%) — 与回测 risk_ml_rank 同口径 (ascending=True)
    # ==========================================================================
    print("\n" + "=" * 120 + f"\n{'4. 协同策略与避雷效果 (Top20)':^120}\n" + "=" * 120)
    risk_rank = combined.groupby('date')['pred_risk'].rank(pct=True, ascending=True)  # 越小越险
    raw_top = combined.sort_values(['date', 'pred'], ascending=[True, False]).groupby('date').head(20)
    raw_ret = raw_top.groupby('date')['target_val'].mean()
    safe_top = combined[risk_rank > 0.10].sort_values(['date', 'pred'], ascending=[True, False]).groupby('date').head(20)
    safe_ret = safe_top.groupby('date')['target_val'].mean()

    def stats(rets):
        ann = rets.mean() * 242
        std = rets.std() * np.sqrt(242)
        nav = (1 + rets).cumprod()
        return ann, ann / (std + 1e-9), ((nav - nav.cummax()) / nav.cummax()).min()
    r, f = stats(raw_ret), stats(safe_ret)
    print(f"{'指标':<12} | {'原始策略':<18} | {'风控过滤':<18} | {'提升'}")
    print("-" * 80)
    print(f"{'年化收益':<12} | {r[0]:>18.2%} | {f[0]:>18.2%} | {f[0]-r[0]:>+8.2%}")
    print(f"{'年化夏普':<12} | {r[1]:>18.4f} | {f[1]:>18.4f} | {f[1]/r[1]-1 if r[1]>0 else 0:>+8.2%}")
    print(f"{'最大回撤':<12} | {r[2]:>18.2%} | {f[2]:>18.2%} | {abs(f[2])-abs(r[2]):>+8.2%}")

    # ==========================================================================
    # 6. 崩溃日避雷效果
    # ==========================================================================
    print("\n" + "-" * 80 + f"\n{'5. 大盘崩溃日避雷效果':^80}\n" + "-" * 80)
    mkt_med = combined.groupby('date')['target_val'].median()
    crash = mkt_med[mkt_med < mkt_med.quantile(0.1)].index
    if not crash.empty:
        c_raw = raw_ret[raw_ret.index.isin(crash)]
        c_safe = safe_ret[safe_ret.index.isin(crash)]
        print(f"崩溃日 {len(crash)} 天 | 原始 Top20 日均: {c_raw.mean():.4f} | 风控过滤后: {c_safe.mean():.4f} "
              f"({(c_safe.mean()-c_raw.mean())*100:+.2f}%/日)" + (" ✅避雷有效" if c_safe.mean() > c_raw.mean() else " ⚠️过滤无效/误杀盈利"))

    # ==========================================================================
    # 7. 年度稳定性
    # ==========================================================================
    print("\n" + "=" * 120 + f"\n{'6. 年度 RankIC 稳定性':^120}\n" + "=" * 120)
    combined['year'] = combined['date'].dt.year
    rows = {}
    for yr, g in combined.groupby('year'):
        if g['date'].nunique() < 30:
            continue
        rows[yr] = {'Alpha_IC': g.groupby('date').apply(
            lambda d: spearmanr(d['pred'], d['target'])[0] if len(d) > 20 else np.nan, include_groups=False).mean()}
        if has_risk_target:
            rows[yr]['Risk_IC'] = g.groupby('date').apply(
                lambda d: spearmanr(d['pred_risk'], d['risk_score'])[0] if len(d) > 20 else np.nan,
                include_groups=False).mean()
    print(pd.DataFrame.from_dict(rows, orient='index').round(4))

    # ==========================================================================
    # 8. 分场景协同 (新增)
    # ==========================================================================
    print("\n" + "=" * 120 + f"\n{'7. 分市场场景: 风控过滤是否误杀盈利 (新增)':^120}\n" + "=" * 120)
    try:
        g = pd.read_csv('global_strategy_audit.csv', usecols=['index', 'strat_primary_scenario'])
        smap = dict(zip(pd.to_datetime(g['index']).dt.normalize(), g['strat_primary_scenario']))
        raw_top = raw_top.copy(); raw_top['scenario'] = raw_top['date'].dt.normalize().map(smap).fillna('normal')
        safe_top = safe_top.copy(); safe_top['scenario'] = safe_top['date'].dt.normalize().map(smap).fillna('normal')
        rr = raw_top.groupby('scenario')['target_val'].mean()
        sr = safe_top.groupby('scenario')['target_val'].mean()
        tbl = pd.DataFrame({'原始Top20': rr, '风控过滤后': sr, 'Δ': sr - rr}).round(4)
        print(tbl.to_string())
        neg = tbl[tbl['Δ'] < 0]
        if not neg.empty:
            print(f"⚠️ 场景 {neg.index.tolist()} 风控过滤拖累收益 (误杀可盈利 alpha)，请结合风控 IC 判定是否过杀。")
    except Exception as e:
        print(f"⚠️ 分场景协同跳过: {e}")

    print("\n" + "-" * 80)
    if risk_data_missing:
        print("ℹ️ 提示: chip_risk_model_v1_newfeat_data.csv 缺失, 仅暴露拓扑近似。完整风控审计请重训时指定 audit_dir: ")
        print("   market_ml 训练风控模型时传 audit_dir 参数 (写入 external_data/audit/ 并绑定 data_file)。")


if __name__ == "__main__":
    _mp = os.environ.get('AUDIT_MODEL_PATH') or None
    if _mp and os.path.exists(_mp):
        _dp = os.environ.get('AUDIT_DATA_PATH') or (
            os.path.splitext(_mp)[0]+'_data.csv'
            if os.path.exists(os.path.splitext(_mp)[0]+'_data.csv') else FALLBACK_DATA)
    else:
        _mp, _dp = None, None
    if _mp is None:
        _kw = {}
    else:
        _rdp = os.environ.get('AUDIT_DATA_PATH') or (
            os.path.splitext(_mp)[0]+'_data.csv'
            if os.path.exists(os.path.splitext(_mp)[0]+'_data.csv') else None)
        _kw = dict(risk_model_path=_mp)
        if _rdp: _kw['risk_data_path'] = _rdp
    run_synergy_audit(**_kw)
