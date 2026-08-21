#!/usr/bin/env python3
"""
signal_level_backtest.py — 信号级回测：固定本金 per trade，独立评估模型+规则

与 PyBroker 组合回测的关系：
- 共用 signal_engine 的买入/卖出规则；
- 通过 BUY_QUOTA_OVERRIDE 取消每日买入限额；
- 用超大初始现金避免现金约束；
- 以每笔交易的收益率（而非资金曲线）作为信号质量指标。

用法:
    python signal_level_backtest.py
    python signal_level_backtest.py --start 2024-01-02 --end 2024-12-31
    python signal_level_backtest.py --capital-per-trade 100000
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# =========================================================================
# 与 G_pca1_z 生产口径对齐
# =========================================================================
# 信号级回测：把仓位系数压平，确保每笔交易权重一致（只关心收益率）
os.environ['BASE_TARGET_SIZE'] = '0.01'
os.environ['POS_MULT_WEIGHT'] = '0.0'
os.environ['POS_MULT_BIAS'] = '1.0'
os.environ['OPPORT_SIZING_COEFF'] = '0.0'
os.environ['RISK_MAG_SELL_THRESHOLD'] = '-0.05'
# 取消买入限额，所有通过硬门槛的候选都允许成交
os.environ['BUY_QUOTA_OVERRIDE'] = '100000'

import co_compute
co_compute.FeatureConfig.MKT_FEATURES = ['mkt_macro_regime']
co_compute.FeatureConfig.PC_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'market_pca_g_pca1_z.parquet'
)

import screen
import backtest

ENTRY_PKL = os.environ.get('ENTRY_MODEL_PKL', 'chip_accumulation_v6_g_pca1_z.pkl')
RISK_PKL = 'chip_risk_model_v1_g_pca1_z.pkl'
OPPORT_PKL = 'chip_opport_magnitude_excess_for_g.pkl'
RISKMAG_PKL = 'chip_risk_magnitude_for_g.pkl'
WARMUP_BARS = 270
SIGNAL_INITIAL_CASH = 1_000_000_000


def ensure_models():
    for pkl in [ENTRY_PKL, RISK_PKL, OPPORT_PKL, RISKMAG_PKL]:
        if not os.path.exists(pkl):
            print(f"❌ 模型缺失: {pkl}")
            sys.exit(1)
    backtest.reload_models(ENTRY_PKL, RISK_PKL)
    backtest.load_magnitude_models(OPPORT_PKL, RISKMAG_PKL)


def compute_signal_metrics(trades_df, capital_per_trade=100_000):
    """基于 PyBroker 成交记录，按固定本金 per trade 计算信号质量指标。"""
    if trades_df is None or trades_df.empty:
        print("⚠️ 没有成交记录")
        return None

    df = trades_df.copy()
    # PyBroker trades 列名可能是 entry/exit/return 等；兼容常见命名
    entry_col = 'entry' if 'entry' in df.columns else 'entry_price'
    exit_col = 'exit' if 'exit' in df.columns else 'exit_price'
    if entry_col not in df.columns or exit_col not in df.columns:
        print(f"⚠️ 无法识别成交价格列，可用列: {list(df.columns)}")
        return None

    df['entry_price'] = pd.to_numeric(df[entry_col], errors='coerce')
    df['exit_price'] = pd.to_numeric(df[exit_col], errors='coerce')
    # 毛收益率（未扣手续费）
    df['return_pct_gross'] = df['exit_price'] / df['entry_price'] - 1.0
    # 假设双边手续费各 0.12%，合计 0.24%
    fee_pct = 0.0012 * 2
    df['return_pct'] = df['return_pct_gross'] - fee_pct
    df['pnl'] = df['return_pct'] * capital_per_trade

    # 持仓天数（PyBroker 的 bars 列或从日期差计算）
    if 'bars' in df.columns:
        df['hold_bars'] = pd.to_numeric(df['bars'], errors='coerce')
    elif 'entry_date' in df.columns and 'exit_date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df['hold_bars'] = (df['exit_date'] - df['entry_date']).dt.days
    else:
        df['hold_bars'] = np.nan

    wins = df['return_pct'] > 0
    losses = df['return_pct'] <= 0

    metrics = {
        'trade_count': int(len(df)),
        'win_count': int(wins.sum()),
        'loss_count': int(losses.sum()),
        'win_rate': float(wins.mean()),
        'avg_return_pct': float(df['return_pct'].mean()),
        'median_return_pct': float(df['return_pct'].median()),
        'mean_win_pct': float(df.loc[wins, 'return_pct'].mean()) if wins.any() else 0.0,
        'mean_loss_pct': float(df.loc[losses, 'return_pct'].mean()) if losses.any() else 0.0,
        'profit_factor': (
            float(df.loc[wins, 'pnl'].sum() / -df.loc[losses, 'pnl'].sum())
            if losses.any() and df.loc[losses, 'pnl'].sum() != 0 else np.inf
        ),
        'max_single_gain_pct': float(df['return_pct'].max()),
        'max_single_loss_pct': float(df['return_pct'].min()),
        'avg_hold_bars': float(df['hold_bars'].mean()),
        'median_hold_bars': float(df['hold_bars'].median()),
    }
    return df, metrics


def run(start_date, end_date, out_dir, capital_per_trade):
    ensure_models()
    symbols = screen.basic_screen()
    if not symbols:
        print("❌ 股票池为空")
        sys.exit(1)
    print(f"股票池: {len(symbols)} 只，固定本金/笔: {capital_per_trade:,.0f}")

    os.makedirs(out_dir, exist_ok=True)
    results_dir = os.path.join(
        out_dir, f"signal_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    backtest.run_backtest(
        symbols,
        start_date=start_date,
        end_date=end_date,
        warmup=WARMUP_BARS,
        results_dir=results_dir,
        initial_cash=SIGNAL_INITIAL_CASH,
    )

    trades_path = os.path.join(results_dir, 'trades.xlsx')
    trades = pd.read_excel(trades_path) if os.path.exists(trades_path) else None

    result = compute_signal_metrics(trades, capital_per_trade)
    if result is None:
        return
    trades_with_metric, metrics = result

    metric_path = os.path.join(results_dir, 'signal_metrics.txt')
    with open(metric_path, 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"\n=== 信号级回测指标（本金/笔={capital_per_trade:,.0f}）===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 保存带收益率的 trades
    trades_with_metric.to_excel(
        os.path.join(results_dir, 'signal_trades.xlsx'), index=False
    )

    # 保存入场信号审计（含 bias/profit_ratio/ml_rank 等），供后续审计分析
    if backtest.BUY_ELIGIBILITY_DETAILS:
        entry_signals = pd.DataFrame(backtest.BUY_ELIGIBILITY_DETAILS)
        entry_signals = entry_signals[entry_signals['is_eligible'] == True].copy()
        entry_signals.rename(columns={'date': 'entry_date'}, inplace=True)
        entry_signals['entry_date'] = pd.to_datetime(entry_signals['entry_date']).dt.date
        entry_signals.to_csv(
            os.path.join(results_dir, 'entry_signals.csv'), index=False, encoding='utf-8-sig'
        )
        print(f"✅ 入场信号审计: {os.path.join(results_dir, 'entry_signals.csv')}")

    print(f"\n✅ 结果目录: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description='信号级回测')
    parser.add_argument('--start', type=str, default='2024-01-02',
                        help='开始日期（建议先用 1 年测试速度）')
    parser.add_argument('--end', type=str, default='2024-12-31',
                        help='结束日期')
    parser.add_argument('--out', type=str,
                        default='external_data/explore_night/signal_level_backtest_20260818',
                        help='输出目录')
    parser.add_argument('--capital-per-trade', type=float, default=100_000,
                        help='每笔名义本金（仅用于指标计算）')
    parser.add_argument('--entry-pkl', type=str, default=None,
                        help='入口模型 pkl 路径（默认 chip_accumulation_v6_g_pca1_z.pkl）')
    args = parser.parse_args()
    if args.entry_pkl:
        os.environ['ENTRY_MODEL_PKL'] = args.entry_pkl
    run(args.start, args.end, args.out, args.capital_per_trade)


if __name__ == '__main__':
    main()
