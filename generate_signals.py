#!/usr/bin/env python3
"""
generate_signals.py — 生成每日全量候选信号 CSV（无现金/无 quota 限制）

用法:
    python generate_signals.py
    python generate_signals.py --start 2021-01-02 --end 2026-08-17
    python generate_signals.py --out external_data/signals/20260818
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# =========================================================================
# 与 G_pca1_z 生产口径对齐（必须在 import co_compute / backtest 之前）
# =========================================================================
os.environ['BASE_TARGET_SIZE'] = '0.12'
os.environ['POS_MULT_WEIGHT'] = '0.5'
os.environ['POS_MULT_BIAS'] = '0.5'
os.environ['OPPORT_SIZING_COEFF'] = '0.30'
os.environ['OPPORT_SIZING_MIN'] = '0.4'
os.environ['OPPORT_SIZING_MAX'] = '1.8'
os.environ['RISK_MAG_SELL_THRESHOLD'] = '-0.05'
# 信号生成不需要真正下单，quota 设为 0 加速
os.environ['BUY_QUOTA_OVERRIDE'] = '0'

import co_compute
co_compute.FeatureConfig.MKT_FEATURES = ['mkt_macro_regime']
co_compute.FeatureConfig.PC_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'market_pca_g_pca1_z.parquet'
)

import screen
import backtest

ENTRY_PKL = 'chip_accumulation_v6_g_pca1_z.pkl'
RISK_PKL = 'chip_risk_model_v1_g_pca1_z.pkl'
OPPORT_PKL = 'chip_opport_magnitude_excess_for_g.pkl'
RISKMAG_PKL = 'chip_risk_magnitude_for_g.pkl'
WARMUP_BARS = 270


def ensure_models():
    for pkl in [ENTRY_PKL, RISK_PKL, OPPORT_PKL, RISKMAG_PKL]:
        if not os.path.exists(pkl):
            print(f"❌ 模型缺失: {pkl}")
            sys.exit(1)
    backtest.reload_models(ENTRY_PKL, RISK_PKL)
    backtest.load_magnitude_models(OPPORT_PKL, RISKMAG_PKL)


def generate(start_date, end_date, out_dir):
    ensure_models()
    symbols = screen.basic_screen()
    if not symbols:
        print("❌ 股票池为空，请先同步数据")
        sys.exit(1)
    print(f"股票池: {len(symbols)} 只")

    os.makedirs(out_dir, exist_ok=True)
    results_dir = os.path.join(
        out_dir, f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    backtest.run_backtest(
        symbols,
        start_date=start_date,
        end_date=end_date,
        warmup=WARMUP_BARS,
        results_dir=results_dir,
    )

    # 从全局审计列表提取全部买入资格记录
    raw = backtest.BUY_ELIGIBILITY_DETAILS
    if not raw:
        print("⚠️ 没有生成任何信号")
        return

    df = pd.DataFrame(raw)
    # 只保留真正通过买入硬门槛的候选
    signals = df[df['is_eligible'] == True].copy()
    signals = signals.sort_values(['date', 'ml_rank'])

    csv_path = os.path.join(out_dir, 'all_signals.csv')
    signals.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"✅ 信号总数: {len(signals)}")
    print(f"✅ 信号 CSV: {csv_path}")
    print(f"✅ 回测中间产物: {results_dir}")

    # 简单分箱摘要
    signals['ml_rank_bin'] = pd.qcut(signals['ml_rank'], 10, duplicates='drop')
    summary = signals.groupby('ml_rank_bin').agg(
        count=('ml_rank', 'size'),
        avg_opport_mag=('opport_mag_z', 'mean'),
        avg_risk_mag=('risk_mag', 'mean'),
    ).reset_index()
    print("\n=== 模型排名分箱摘要 ===")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description='生成每日候选信号 CSV')
    parser.add_argument('--start', type=str, default='2021-01-02', help='开始日期')
    parser.add_argument('--end', type=str, default='2026-08-17', help='结束日期')
    parser.add_argument('--out', type=str,
                        default='external_data/explore_night/signal_level_backtest_20260818/signals',
                        help='输出目录')
    args = parser.parse_args()
    generate(args.start, args.end, args.out)


if __name__ == '__main__':
    main()
