import os
import joblib
import pandas as pd
import numpy as np
import logging
import warnings
from scipy.stats import spearmanr, norm

# 静默处理
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def run_ultimate_synergy_audit(
    buy_model_path='chip_accumulation_v6.pkl', 
    risk_model_path='chip_risk_model_v1.pkl',
    buy_data_path='model_data.csv', 
    risk_data_path='model_risk_data.csv'
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # ==========================================================================================
    # 1. 现场生成预测分 (防止 KeyError)
    # ==========================================================================================
    logging.info("加载模型并生成预测分数...")
    
    # A. 买入模型加载与预测
    buy_pkg = joblib.load(buy_model_path)
    b_model, b_features = buy_pkg['model'], buy_pkg['features']
    # 【数据口径修复】优先使用模型绑定的审计数据文件
    if 'data_file' in buy_pkg and os.path.exists(buy_pkg['data_file']):
        buy_data_path = buy_pkg['data_file']
        logging.info(f"买入模型绑定数据: {buy_data_path}")
    buy_df = pd.read_csv(buy_data_path)
    buy_df['date'] = pd.to_datetime(buy_df['date'])
    buy_df['pred'] = b_model.predict(buy_df[b_features])
    
    # B. 风险模型加载与预测
    risk_pkg = joblib.load(risk_model_path)
    r_model, r_features = risk_pkg['model'], risk_pkg['features']
    # 【数据口径修复】优先使用模型绑定的审计数据文件
    if 'data_file' in risk_pkg and os.path.exists(risk_pkg['data_file']):
        risk_data_path = risk_pkg['data_file']
        logging.info(f"风险模型绑定数据: {risk_data_path}")
    risk_df = pd.read_csv(risk_data_path)
    risk_df['date'] = pd.to_datetime(risk_df['date'])
    risk_df['pred_risk'] = r_model.predict(risk_df[r_features])

    # 对齐 OOS 数据
    combined = pd.merge(
        buy_df[['date', 'symbol', 'pred', 'target', 'target_val'] + b_features],
        risk_df[['date', 'symbol', 'pred_risk', 'risk_score'] + r_features],
        on=['date', 'symbol'], how='inner'
    )
    oos_df = combined[combined['date'] >= '2020-01-01'].copy()
    del buy_df, risk_df # 释放内存

    # ==========================================================================================
    # 2. 维度 1: 特征质量与重要性审计
    # ==========================================================================================
    print("\n" + "="*120 + f"\n{'1. 特征重要性审计 (Buy Model vs Risk Model)':^120}\n" + "="*120)
    
    def get_imp_df(model, features):
        imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        imp['Pct'] = imp['Importance'] / imp['Importance'].sum()
        return imp.sort_values('Importance', ascending=False).head(10)

    print("\n[买入模型 Top 10 特征]:")
    print(get_imp_df(b_model, b_features).to_string(index=False))
    print("\n[风险模型 Top 10 特征]:")
    print(get_imp_df(r_model, r_features).to_string(index=False))

    # ==========================================================================================
    # 3. 维度 2: 核心预测表现 (RankIC)
    # ==========================================================================================
    print("\n" + "="*120 + f"\n{'2. 核心预测表现审计 (RankIC)':^120}\n" + "="*120)
    
    def calc_ic(df, p_col, t_col):
        return df.groupby('date').apply(lambda x: spearmanr(x[p_col], x[t_col])[0] if len(x)>20 else np.nan, include_groups=False).mean()

    buy_ic = calc_ic(oos_df, 'pred', 'target')
    risk_ic = calc_ic(oos_df, 'pred_risk', 'risk_score')
    
    print(f"买入模型 (Alpha) OOS RankIC: {buy_ic:.4f}")
    print(f"风险模型 (Risk)  OOS RankIC: {risk_ic:.4f} (预期为负)")

    # ==========================================================================================
    # 4. 维度 3: 分箱单调性审计
    # ==========================================================================================
    print("\n" + "="*120 + f"\n{'3. 分箱单调性审计 (Decile Analysis)':^120}\n" + "="*120)
    
    oos_df['buy_bin'] = oos_df.groupby('date')['pred'].transform(lambda x: pd.qcut(x, 10, labels=False))
    oos_df['risk_bin'] = oos_df.groupby('date')['pred_risk'].transform(lambda x: pd.qcut(x, 10, labels=False))
    
    buy_bins = oos_df.groupby('buy_bin')['target_val'].mean()
    risk_bins = oos_df.groupby('risk_bin')['risk_score'].mean()
    
    print("买入分箱 (收益):")
    print(buy_bins.to_frame().T)
    print("风险分箱 (风险分):")
    print(risk_bins.to_frame().T)

    # ==========================================================================================
    # 5. 维度 4: 协同策略表现 (Synergy)
    # ==========================================================================================
    print("\n" + "="*120 + f"\n{'4. 策略协同表现与避雷效果审计':^120}\n" + "="*120)
    
    # 协同逻辑：剔除风险最高的前 10% (风险分最负的，由于 rank(ascending=True) 会在 Bin 0)
    # 这里的过滤逻辑根据你的训练结果调整：如果最危险的是 Bin 0：
    combined_risk_rank = oos_df.groupby('date')['pred_risk'].rank(pct=True, ascending=True)
    
    # 策略 A
    raw_top20 = oos_df.sort_values(['date', 'pred'], ascending=[True, False]).groupby('date').head(20)
    raw_ret = raw_top20.groupby('date')['target_val'].mean()
    
    # 策略 B
    safe_mask = combined_risk_rank > 0.10
    filtered_top20 = oos_df[safe_mask].sort_values(['date', 'pred'], ascending=[True, False]).groupby('date').head(20)
    filtered_ret = filtered_top20.groupby('date')['target_val'].mean()

    def get_nav_stats(rets):
        ann_ret = rets.mean() * 242
        ann_std = rets.std() * np.sqrt(242)
        sharpe = ann_ret / (ann_std + 1e-9)
        cum_nav = (1 + rets).cumprod()
        mdd = ((cum_nav - cum_nav.cummax()) / cum_nav.cummax()).min()
        return ann_ret, sharpe, mdd

    r_stats = get_nav_stats(raw_ret)
    f_stats = get_nav_stats(filtered_ret)

    print(f"{'指标':<15} | {'原始策略':<20} | {'协同策略':<20} | {'提升'}")
    print("-" * 80)
    print(f"{'年化收益':<15} | {r_stats[0]:>20.2%} | {f_stats[0]:>20.2%} | {f_stats[0]-r_stats[0]:>+8.2%}")
    print(f"{'年化夏普':<15} | {r_stats[1]:>20.4f} | {f_stats[1]:>20.4f} | {f_stats[1]/r_stats[1]-1:>+8.2%}")
    print(f"{'最大回撤':<15} | {r_stats[2]:>20.2%} | {f_stats[2]:>20.2%} | {abs(f_stats[2])-abs(r_stats[2]):>+8.2%}")

    # ==========================================================================================
    # 6. 维度 5: 年度稳定性
    # ==========================================================================================
    print("\n" + "="*120 + f"\n{'5. 年度稳定性审计 (Yearly RankIC)':^120}\n" + "="*120)
    oos_df['year'] = oos_df['date'].dt.year
    yearly = oos_df.groupby('year').apply(
        lambda x: pd.Series({
            'Alpha_IC': x.groupby('date').apply(lambda d: spearmanr(d['pred'], d['target'])[0], include_groups=False).mean(),
            'Risk_IC': x.groupby('date').apply(lambda d: spearmanr(d['pred_risk'], d['risk_score'])[0], include_groups=False).mean()
        }), include_groups=False
    )
    print(yearly)

if __name__ == "__main__":
    run_ultimate_synergy_audit()