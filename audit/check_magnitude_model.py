# audit/check_magnitude_model.py
# 幅度模型 (opport_mag / risk_mag) 全面审计
# 用法: python audit/check_magnitude_model.py [模型路径] [target_col]
#       python audit/check_magnitude_model.py external_data/models/chip_opport_magnitude.pkl target_val
#       python audit/check_magnitude_model.py external_data/models/chip_risk_magnitude.pkl risk_score
# 
# 检查维度:
#   1. 预测分布与 MSE 压缩检测 (核心 — 区分度是否够做阈值判断)
#   2. 日 Z-score 可用性 (CV/分位分布)
#   3. OOS RankIC + ICIR + 年度稳定性
#   4. 分场景 IC (市场状态切换时预测是否退化)
#   5. 全量特征重要性 (末位低重要性标记删除候选)
#   6. 预测尺度校验 (pred_std / true_std 检测过度压缩)
#   7. 日 Z-score 后区分度 + 阈值可用性
import os, sys, joblib, logging, numpy as np, pandas as pd
from scipy.stats import spearmanr

DEFAULT_MODEL = os.path.join('external_data', 'models', 'chip_opport_magnitude.pkl')
DEFAULT_TARGET = 'target_val'
OOS_START = '2020-01-01'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _log(msg):
    logging.info(msg)
    print(msg)


def _load_scenario_map():
    """读取 global_strategy_audit.csv 构建 date→scenario 映射"""
    path = 'global_strategy_audit.csv'
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, usecols=['index', 'strat_primary_scenario'])
    df = df.rename(columns={'index': 'date', 'strat_primary_scenario': 'primary_scenario'})
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    return dict(zip(df['date'], df['primary_scenario']))


def _safe_ic_mean(s):
    """groupby apply 可能返回 float (单组) 或 Series"""
    if isinstance(s, (float, np.floating, int, np.integer)):
        return float(s)
    v = s.dropna()
    if v.empty or isinstance(v, (float, np.floating)):
        return float(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan
    return float(v.mean())


def run(model_path=None, target_col=None):
    model_path = model_path or DEFAULT_MODEL
    target_col = target_col or DEFAULT_TARGET
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        return

    pkg = joblib.load(model_path)
    features = pkg['features']
    model = pkg['model']
    actual_target = pkg.get('target_col', target_col)
    _log(f"模型: {model_path}  ({len(features)} 特征, target={actual_target})")

    # 读 OOS 数据
    data_file = pkg.get('data_file')
    if not data_file or not os.path.exists(data_file):
        data_file = 'external_data/audit/chip_accumulation_v6_newfeat_data.csv'
    _log(f"审计数据: {data_file}")

    nrows = None  # 已禁用截断：全量流式（chunksize）
    usecols = ['date'] + features
    # target_col only added if it exists in data (risk_score typically doesn't)
    target_in_data = target_col and target_col not in features
    kw = {'usecols': usecols, 'chunksize': 800000}
    if nrows:
        kw['nrows'] = nrows
        kw.pop('chunksize')

    chunks = []
    for ch in pd.read_csv(data_file, **kw):
        ch['date'] = pd.to_datetime(ch['date'], format='mixed')
        ch = ch[(ch['date'] >= OOS_START) & (ch['date'] <= '2020-12-31')]
        if len(ch) == 0:
            continue
        ch['pred'] = model.predict(ch[features])
        # target_col from pkg or arg; if in separate CSV column, include it
        cols = ['date', 'pred']
        if target_col and target_col in ch.columns:
            cols.append(target_col)
        chunks.append(ch[cols])

    df = pd.concat(chunks, axis=0).reset_index(drop=True)
    _log(f"OOS 样本数: {len(df):,}  (2020-01-01 ~ 2020-12-31)")

    p = df['pred'].values
    scenario_map = _load_scenario_map()
    if scenario_map:
        df['scenario'] = df['date'].map(scenario_map).fillna('unknown')

    # ========================================================================
    # 1. 预测分布与 MSE 压缩检测
    # ========================================================================
    print(f"\n{'='*80}")
    print("1. 预测分布与 MSE 压缩检测")
    print(f"{'='*80}")
    print(f"  mean={p.mean():.4f}  std={p.std():.4f}  min={p.min():.4f}  max={p.max():.4f}")
    print(f"  p1={np.percentile(p,1):.4f}  p10={np.percentile(p,10):.4f}  p25={np.percentile(p,25):.4f}")
    print(f"  p50={np.percentile(p,50):.4f}  p75={np.percentile(p,75):.4f}  p90={np.percentile(p,90):.4f}")
    print(f"  p99={np.percentile(p,99):.4f}")
    spread = np.percentile(p, 99) - np.percentile(p, 1)
    if spread < 0.05 or p.std() < 0.01:
        status = "⚠️ 严重 MSE 压缩"
        print(f"  {status} (spread={spread:.4f} std={p.std():.4f}) — 绝对值不可靠, 必须日 Z-score")
    elif spread < 0.1 or p.std() < 0.02:
        status = "⚡ MSE 压缩"
        print(f"  {status} (spread={spread:.4f} std={p.std():.4f}) — 建议日 Z-score 后使用")
    else:
        status = "✅ 区分度正常"
        print(f"  {status} (spread={spread:.4f} std={p.std():.4f})")

    # ========================================================================
    # 2. 日 Z-score 可用性
    # ========================================================================
    print(f"\n{'='*80}")
    print("2. 日 Z-score 可用性")
    print(f"{'='*80}")
    grp = df.groupby('date')['pred']
    df['pred_z'] = ((df['pred'] - grp.transform('mean')) / (grp.transform('std') + 1e-9)).clip(-5, 5)
    pz = df['pred_z'].dropna().values
    print(f"  z-score 后: mean≈0 std={pz.std():.2f}")
    for sigma, label in [(1.5, '警戒'), (2.0, '卖出'), (2.5, '强卖')]:
        pct_above = (pz > sigma).mean()
        pct_below = (pz < -sigma).mean()
        print(f"  |z| > {sigma}σ (>1σ={((pz>sigma)|(pz<-sigma)).mean():.1%})  "
              f"买入侧={pct_below:.1%}  卖出侧={pct_above:.1%}")
    if pz.std() < 0.5:
        print(f"  ⚡ z-score 后区分度仍低 (std={pz.std():.2f}) — 模型预测能力极弱")
    elif pz.std() > 1.5:
        print(f"  ⚡ z-score 后分布过宽 (std={pz.std():.2f}) — 参数需调 clip")

    # Skip IC if target not in data
    has_target = target_col in df.columns if target_col else False
    if not has_target:
        print(f"\n⏭️ 跳过 IC 分析 ({target_col} 不在审计数据中)")
    else:
        ic_daily = df.groupby('date').apply(
            lambda x: spearmanr(x['pred'], x[target_col])[0]
            if len(x) > 15 and x['pred'].std() > 1e-9 and x[target_col].std() > 1e-9
            else np.nan, include_groups=False
        )
        ic_daily = ic_daily.apply(_safe_ic_mean)

        print(f"\n{'='*80}")
        print("3. OOS RankIC + 年度稳定性")
        print(f"{'='*80}")
        print(f"  RankIC={ic_daily.mean():.4f}  ICIR={ic_daily.mean()/(ic_daily.std()+1e-9):.3f}  天数={ic_daily.notna().sum()}")

        # 年度
        ic_yearly = df.copy()
        ic_yearly['year'] = ic_yearly['date'].dt.year
        ic_yearly['ic'] = ic_yearly['date'].map(ic_daily)
        yearly = ic_yearly.groupby('year')['ic'].agg(['mean', 'count']).dropna()
        print(f"\n  年度 RankIC:")
        for yr, row in yearly.iterrows():
            flag = " ⚠️" if row['mean'] < 0 else ""
            print(f"    {int(yr)}: {row['mean']:+.4f}  (n={int(row['count']):.0f}){flag}")

        # 场景
        if scenario_map:
            ic_yearly['scenario'] = ic_yearly['date'].map(scenario_map).fillna('unknown')
            scen = ic_yearly.groupby('scenario')['ic'].agg(['mean', 'count'])
            print(f"\n  分场景 IC:")
            for s, row in scen.iterrows():
                flag = " ⚠️ 退化" if row['mean'] < -0.01 else ""
                print(f"    {s:<12s} {row['mean']:+.4f}  (n={int(row['count']):.0f}){flag}")

    # ========================================================================
    # 4. 预测尺度校验
    # ========================================================================
    if target_col in df.columns:
        print(f"\n{'='*80}")
        print("4. 预测尺度校验 (pred vs true)")
        print(f"{'='*80}")
        true_mean = df[target_col].mean()
        true_std = df[target_col].std()
        pred_mean = df['pred'].mean()
        pred_std = df['pred'].std()
        scale_ratio = pred_std / (true_std + 1e-9)
        bias = (pred_mean - true_mean) / (abs(true_mean) + 1e-9)
        print(f"  真值: mean={true_mean:.4f} std={true_std:.4f}")
        print(f"  预测: mean={pred_mean:.4f} std={pred_std:.4f}")
        print(f"  σ比 (pred/true): {scale_ratio:.4f}  ({'✅ 正常' if 0.1<scale_ratio<10 else '⚠️ 尺度严重不匹配'})")
        print(f"  偏差 (pred-true)/|true|: {bias*100:.1f}%  ({'✅ 无偏' if abs(bias)<0.2 else '⚠️ 有偏'})")
        if scale_ratio < 0.1:
            print(f"  ⚠️ 预测 σ 严重小于真值 σ — MSE 损失过度压缩, 必须日 Z-score")

    # ========================================================================
    # 5. 全量特征重要性 (末位标记)
    # ========================================================================
    print(f"\n{'='*80}")
    print("5. 全量特征重要性")
    print(f"{'='*80}")
    imp = np.maximum(model.feature_importances_, 0)
    total = imp.sum()
    order = np.argsort(imp)[::-1]
    cum = 0
    low_features = []
    for rank, idx in enumerate(order):
        pct = imp[idx] / (total + 1e-9)
        cum += pct
        flag = ""
        if pct < 0.001:  # < 0.1%
            flag = " ⚡ 极低 — 删除候选"
            low_features.append(features[idx])
        elif pct < 0.005:  # < 0.5%
            flag = " — 低"
        print(f"  {rank+1:>2d}. {features[idx]:<28s} {pct:>6.1%}  (累计 {cum:.1%}){flag}")
    if low_features:
        print(f"\n  💡 建议删除 ({len(low_features)} 个): {', '.join(low_features)}")
    else:
        print(f"\n  ✅ 所有特征重要性 > 0.1%, 无需删减")

    # ========================================================================
    # 6. 综合判定
    # ========================================================================
    print(f"\n{'='*80}")
    print("6. 综合判定")
    print(f"{'='*80}")
    issues = []
    if status.startswith("⚠️"):
        issues.append("MSE 压缩 — 必须日 Z-score 后使用")
    elif status.startswith("⚡"):
        issues.append("MSE 压缩 — 建议日 Z-score")
    if target_col in df.columns and ic_daily.mean() < 0.0:
        issues.append(f"OOS RankIC={ic_daily.mean():.4f} < 0 — 模型无预测力")
    if low_features:
        issues.append(f"{len(low_features)} 个特征重要性 < 0.1% — 可删除")
    if 'scale_ratio' in dir() and scale_ratio < 0.1:
        issues.append("预测/真值 σ 比 < 0.1 — 尺度压缩严重")
    if not issues:
        print("  ✅ 模型健康, 无严重问题")
    else:
        for i, issue in enumerate(issues):
            print(f"  {i+1}. {issue}")


if __name__ == '__main__':
    run(
        sys.argv[1] if len(sys.argv) > 1 else None,
        sys.argv[2] if len(sys.argv) > 2 else None,
    )