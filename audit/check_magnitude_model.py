# audit/check_magnitude_model.py
# 幅度模型 (opport_mag / risk_mag) 审计: 预测分布、IC、区分度
# 用法: python audit/check_magnitude_model.py [模型路径] [target_col]
# 默认: external_data/models/chip_opport_magnitude.pkl
import os, sys, joblib, logging, numpy as np, pandas as pd
from scipy.stats import spearmanr

DEFAULT_MODEL = os.path.join('external_data', 'models', 'chip_opport_magnitude.pkl')
DEFAULT_TARGET = 'target_val'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    print(f"模型: {model_path}  ({len(features)} 特征, target={actual_target})")

    # 读 OOS 数据 (复用 newfeat 审计, 特征列对齐)
    data_file = pkg.get('data_file') or 'external_data/audit/chip_accumulation_v6_newfeat_data.csv'
    if not os.path.exists(data_file):
        data_file = 'external_data/audit/chip_accumulation_v6_newfeat_data.csv'
    print(f"审计数据: {data_file}")

    chunks = []
    for ch in pd.read_csv(data_file, usecols=['date'] + features, chunksize=800000):
        ch['date'] = pd.to_datetime(ch['date'])
        ch = ch[(ch['date'] > '2019-12-31') & (ch['date'] <= '2020-12-31')]
        if len(ch) > 0:
            ch['pred'] = model.predict(ch[features])
            chunks.append(ch[['date', 'pred']])

    df = pd.concat(chunks).reset_index(drop=True)
    print(f"OOS 样本数: {len(df):,}")

    # 1. 预测分布检查 (核心: 检测 MSE 压缩)
    p = df['pred'].values
    print(f"\n{'='*60}")
    print(f"1. 预测分布 (检测 MSE 压缩)")
    print(f"{'='*60}")
    print(f"  mean={p.mean():.4f}  std={p.std():.4f}  min={p.min():.4f}  max={p.max():.4f}")
    print(f"  p1={np.percentile(p,1):.4f}  p10={np.percentile(p,10):.4f}  p25={np.percentile(p,25):.4f}")
    print(f"  p50={np.percentile(p,50):.4f}  p75={np.percentile(p,75):.4f}  p90={np.percentile(p,90):.4f}")
    print(f"  p99={np.percentile(p,99):.4f}")
    spread = np.percentile(p, 99) - np.percentile(p, 1)
    # 双条件: spread 太小或 CV 太低都是压缩信号
    if spread < 0.05 or p.std() < 0.01:
        print(f"  ⚠️ 严重 MSE 压缩 (spread={spread:.4f} std={p.std():.4f}) — 绝对值不可靠, 需要日 Z-score")
    elif spread < 0.1 or p.std() < 0.02:
        print(f"  ⚡ MSE 压缩 (spread={spread:.4f} std={p.std():.4f}) — 建议日 Z-score 后使用")
    else:
        print(f"  ✅ 区分度正常 (spread={spread:.4f} std={p.std():.4f})")

    # 2. 日 Z-score 后分布
    grp = df.groupby('date')['pred']
    df['pred_z'] = (df['pred'] - grp.transform('mean')) / (grp.transform('std') + 1e-9)
    pz = df['pred_z'].clip(-5, 5).dropna().values
    print(f"\n  日 Z-score 后:")
    print(f"  std={pz.std():.2f}  >2σ={(pz>2).mean():.1%}  <-2σ={(pz<-2).mean():.1%}")

    # 3. OOS IC (有 target 列时)
    if target_col in df.columns:
        ic = df.groupby('date').apply(
            lambda x: spearmanr(x['pred'], x[target_col])[0]
            if len(x) > 15 and x['pred'].std() > 1e-9 and x[target_col].std() > 1e-9
            else np.nan, include_groups=False
        )
        print(f"\n{'='*60}")
        print(f"3. OOS RankIC")
        print(f"{'='*60}")
        print(f"  RankIC={ic.mean():.4f}  ICIR={ic.mean()/(ic.std()+1e-9):.3f}  天数={ic.notna().sum()}")

    # 4. 特征重要性 top5
    imp = np.maximum(model.feature_importances_, 0)
    idx = np.argsort(imp)[::-1][:5]
    total = imp.sum()
    print(f"\n{'='*60}")
    print(f"4. 特征重要性 Top 5")
    print(f"{'='*60}")
    for i in idx:
        print(f"  {features[i]:<25s} {imp[i]/total:.1%}")


if __name__ == '__main__':
    run(
        sys.argv[1] if len(sys.argv) > 1 else None,
        sys.argv[2] if len(sys.argv) > 2 else None,
    )