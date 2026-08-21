#!/usr/bin/env python3
"""
audit_signal_level.py — 对信号级回测结果做特征审计

用法：
    python audit_signal_level.py \
        --trades external_data/.../signal_trades.xlsx \
        --signals external_data/.../entry_signals.csv \
        --out external_data/.../audit
"""
import os
import argparse
import pandas as pd
import numpy as np


def load_data(trades_path, signals_path):
    trades = pd.read_excel(trades_path)
    signals = pd.read_csv(signals_path)

    trades['symbol'] = trades['symbol'].astype(str)
    signals['symbol'] = signals['symbol'].astype(str)

    trades['entry_date'] = pd.to_datetime(trades['entry_date']).dt.date
    # 兼容 generate_signals 输出的 'date' 列
    date_col = 'entry_date' if 'entry_date' in signals.columns else 'date'
    signals = signals.rename(columns={date_col: 'entry_date'})
    signals['entry_date'] = pd.to_datetime(signals['entry_date']).dt.date

    # PyBroker buy_delay=1：trades.entry_date 是信号决策日的下一个交易日。
    # 因此把 trade entry_date 映射回 signals 里的前一个交易日（决策日）。
    signal_dates = sorted(signals['entry_date'].unique())
    prev_date_map = {signal_dates[i + 1]: signal_dates[i] for i in range(len(signal_dates) - 1)}
    trades['signal_date'] = trades['entry_date'].map(prev_date_map)

    unmatched = trades['signal_date'].isna().sum()
    if unmatched:
        print(f"⚠️ 有 {unmatched} 笔交易无法映射到信号决策日")

    df = trades.merge(
        signals,
        left_on=['symbol', 'signal_date'],
        right_on=['symbol', 'entry_date'],
        how='left',
        suffixes=('', '_signal')
    )
    if len(df) != len(trades):
        print(f"⚠️ join 后行数变化: trades={len(trades)}, joined={len(df)}")
    return df


def bin_summary(df, feature, bins=10, by_sign=False):
    if feature not in df.columns:
        return None
    d = df.dropna(subset=[feature, 'return_pct']).copy()
    if len(d) == 0:
        return None

    if by_sign:
        d['bin'] = pd.cut(d[feature], bins=[-np.inf, -0.05, 0, 0.05, np.inf],
                          labels=['<-5%', '-5~0', '0~5%', '>5%'])
    else:
        try:
            d['bin'] = pd.qcut(d[feature], q=bins, duplicates='drop')
        except ValueError:
            d['bin'] = pd.cut(d[feature], bins=bins)

    summary = d.groupby('bin').agg(
        count=('return_pct', 'size'),
        win_rate=('return_pct', lambda x: (x > 0).mean()),
        avg_return=('return_pct', 'mean'),
        median_return=('return_pct', 'median'),
        mean_win=('return_pct', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        mean_loss=('return_pct', lambda x: x[x <= 0].mean() if (x <= 0).any() else 0),
        profit_factor=('return_pct', lambda x: _profit_factor(x)),
        max_loss=('return_pct', 'min'),
        avg_feature=(feature, 'mean'),
    ).reset_index()
    return summary


def _profit_factor(x):
    wins = x[x > 0].sum()
    losses = -x[x <= 0].sum()
    return wins / losses if losses > 0 else np.inf


def scenario_bias_summary(df):
    """按场景和 bias 符号/阈值看收益。"""
    d = df.dropna(subset=['primary_scenario', 'entry_bias', 'return_pct']).copy()
    if len(d) == 0:
        return None

    def bias_bucket(row):
        b = row['entry_bias']
        scen = row['primary_scenario']
        if 'bottom' in scen:
            return 'bottom_bias<0' if b < 0 else 'bottom_bias>=0'
        elif 'opportunity' in scen:
            return 'opp_bias>-0.05' if b > -0.05 else 'opp_bias<=-0.05'
        elif 'normal' in scen:
            return 'normal_bias>0.05' if b > 0.05 else 'normal_bias<=0.05'
        else:
            return 'other'

    d['bias_bucket'] = d.apply(bias_bucket, axis=1)
    summary = d.groupby('bias_bucket').agg(
        count=('return_pct', 'size'),
        win_rate=('return_pct', lambda x: (x > 0).mean()),
        avg_return=('return_pct', 'mean'),
        median_return=('return_pct', 'median'),
        profit_factor=('return_pct', lambda x: _profit_factor(x)),
        max_loss=('return_pct', 'min'),
        avg_bias=('entry_bias', 'mean'),
    ).reset_index()
    return summary


def conditional_summary(df):
    """测试 profit_ratio_con 如果启用是否有区分度。"""
    if 'profit_ratio_ma3' not in df.columns:
        return None
    d = df.dropna(subset=['profit_ratio_ma3', 'return_pct']).copy()
    if len(d) == 0:
        return None

    if 'profit_ratio_q50' in d.columns:
        d['profit_ratio_con'] = d['profit_ratio_ma3'] > d['profit_ratio_q50']
    else:
        # 如果审计表里没有 q50，按日期截面计算
        d['profit_ratio_q50'] = d.groupby('entry_date')['profit_ratio_ma3'].transform(lambda x: x.quantile(0.5))
        d['profit_ratio_con'] = d['profit_ratio_ma3'] > d['profit_ratio_q50']
    summary = d.groupby('profit_ratio_con').agg(
        count=('return_pct', 'size'),
        win_rate=('return_pct', lambda x: (x > 0).mean()),
        avg_return=('return_pct', 'mean'),
        median_return=('return_pct', 'median'),
        profit_factor=('return_pct', lambda x: _profit_factor(x)),
        max_loss=('return_pct', 'min'),
    ).reset_index()
    return summary


def run(trades_path, signals_path, out_dir):
    df = load_data(trades_path, signals_path)
    os.makedirs(out_dir, exist_ok=True)

    features = ['ml_rank', 'entry_bias', 'profit_ratio_ma3', 'risk_ml_rank', 'risk_mag', 'opport_mag_z']
    with pd.ExcelWriter(os.path.join(out_dir, 'signal_audit.xlsx'), engine='openpyxl') as writer:
        # 总体分布
        overall = pd.DataFrame({
            'trade_count': [len(df)],
            'win_rate': [(df['return_pct'] > 0).mean()],
            'avg_return': [df['return_pct'].mean()],
            'median_return': [df['return_pct'].median()],
            'profit_factor': [_profit_factor(df['return_pct'])],
        })
        overall.to_excel(writer, sheet_name='overall', index=False)

        for feat in features:
            summary = bin_summary(df, feat, bins=10)
            if summary is not None:
                sheet = f'{feat}_bin'[:31]
                summary.to_excel(writer, sheet_name=sheet, index=False)

        # 场景+bias 桶
        scen = scenario_bias_summary(df)
        if scen is not None:
            scen.to_excel(writer, sheet_name='scenario_bias', index=False)

        # profit_ratio_con 条件
        pr = conditional_summary(df)
        if pr is not None:
            pr.to_excel(writer, sheet_name='profit_ratio_con', index=False)

        # 按场景
        if 'primary_scenario' in df.columns:
            scen_summary = df.groupby('primary_scenario').agg(
                count=('return_pct', 'size'),
                win_rate=('return_pct', lambda x: (x > 0).mean()),
                avg_return=('return_pct', 'mean'),
                median_return=('return_pct', 'median'),
                profit_factor=('return_pct', lambda x: _profit_factor(x)),
                max_loss=('return_pct', 'min'),
            ).reset_index()
            scen_summary.to_excel(writer, sheet_name='scenario', index=False)

    print(f"\n✅ 审计报告: {os.path.join(out_dir, 'signal_audit.xlsx')}")

    # 打印关键发现
    print("\n=== 关键发现 ===")
    if 'entry_bias' in df.columns:
        s = bin_summary(df, 'entry_bias', by_sign=True)
        if s is not None:
            print("\n--- entry_bias 分桶 ---")
            print(s[['bin', 'count', 'win_rate', 'avg_return', 'profit_factor', 'max_loss']].to_string(index=False))

    if 'profit_ratio_ma3' in df.columns:
        pr = conditional_summary(df)
        if pr is not None:
            print("\n--- profit_ratio_con (profit_ratio_ma3 > q50) ---")
            print(pr.to_string(index=False))

    scen = scenario_bias_summary(df)
    if scen is not None:
        print("\n--- 场景 + bias 条件 ---")
        print(scen.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='信号级回测审计')
    parser.add_argument('--trades', type=str, required=True, help='signal_trades.xlsx 路径')
    parser.add_argument('--signals', type=str, required=True, help='entry_signals.csv 路径')
    parser.add_argument('--out', type=str, default='external_data/explore_night/signal_level_backtest_20260818/audit', help='输出目录')
    args = parser.parse_args()
    run(args.trades, args.signals, args.out)


if __name__ == '__main__':
    main()
