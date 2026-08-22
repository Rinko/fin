# -*- coding: utf-8 -*-
"""
每日运行脚本：更新数据 → 加载最优模型 → 回测选股 → 输出今日信号

pybroker warmup 机制：数据从 start_date 导入，走完 warmup 个交易日才进入交易期。
因此 start_date 需 = 交易期起点 - warmup 个交易日，才能有足额交易区间。

用法:
  python main.py                               # 默认: 现役模型, 交易期近半年 + warmup, 数据同步
  python main.py --no-data                     # 跳过数据同步
  python main.py --full                        # 完整回测 2021 至今 (与历史比较)
  python main.py --days 180                    # 自定义交易期长度 (交易日)
  python main.py --entry-model chip_accumulation_v6.pkl --risk-model chip_risk_model_v1.pkl  # 指定模型
  python main.py --entry-model chip_*_mktfull.pkl            # 仅覆盖入场模型, 风控用现役
"""
import os
import sys
import argparse
import logging
import subprocess
import pandas as pd
from datetime import datetime

# =============================================================================
# 最佳版本配置：G_pca1_z + opport_mag_excess sizing + risk_mag 硬止损
# 必须在 import backtest 之前设置，因为 backtest.py 在导入时读取环境变量
# =============================================================================
# 仓位基准：0.12 曾导致并发市值达权益的 ~2.9 倍、现金约束在 >50% 交易日拒单；
# 降至 0.04 后峰值占用 ≈0.98×权益，quota 内信号可全部成交（INITIAL_CASH 维持 1M 口径不变）
os.environ['BASE_TARGET_SIZE'] = os.environ.get('BASE_TARGET_SIZE', '0.04')
os.environ['POS_MULT_WEIGHT'] = '0.5'
os.environ['POS_MULT_BIAS'] = '0.5'
os.environ['OPPORT_SIZING_COEFF'] = '0.30'
os.environ['OPPORT_SIZING_MIN'] = '0.4'
os.environ['OPPORT_SIZING_MAX'] = '1.8'
os.environ['RISK_MAG_SELL_THRESHOLD'] = '-0.05'

import co_compute

# 与 G_pca1_z 对齐：单主成分 + 项目根目录预计算 PC 表
co_compute.FeatureConfig.MKT_FEATURES = ['mkt_macro_regime']
co_compute.FeatureConfig.PC_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'market_pca_g_pca1_z.parquet'
)

import screen
import backtest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 现役最优模型 (backtest.py 默认加载旧模型, 必须显式 reload)
ENTRY_PKL = 'chip_accumulation_v6_g_pca1_z.pkl'
RISK_PKL = 'chip_risk_model_v1_g_pca1_z.pkl'
OPPORT_PKL = 'chip_opport_magnitude_excess_for_g.pkl'
RISKMAG_PKL = 'chip_risk_magnitude_for_g.pkl'

WARMUP_BARS = 270        # 与 backtest.py 一致
FULL_START = '2021-01-02'  # 完整回测起点 (历史对比基准)
BASELINE_END = '2026-08-17'  # 当前新基线截止日期


def get_trading_calendar():
    """从 zzqz_df.xlsx 提取 A 股交易日历。"""
    zz = pd.read_excel('zzqz_df.xlsx', usecols=['日期'])
    return pd.to_datetime(zz['日期']).sort_values().reset_index(drop=True)


def trading_days_back(calendar, days):
    """返回距离最新交易日往前 days 个交易日的日期。"""
    return calendar.iloc[len(calendar) - 1 - days]


def sync_data():
    """同步每日数据: financial/macro/zzqz/market/daily 全部任务"""
    print("=" * 60)
    print("Step 1/3: 同步数据 (get_base_data.py --task all)")
    print("=" * 60)
    r = subprocess.run([sys.executable, 'get_base_data.py', '--task', 'all'])
    if r.returncode != 0:
        print("⚠️ 数据同步失败 (可能是网络/验证码问题), 继续用现有数据选股")
    else:
        print("✅ 数据同步完成")


def load_models(entry_pkl=ENTRY_PKL, risk_pkl=RISK_PKL,
                opport_pkl=OPPORT_PKL, riskmag_pkl=RISKMAG_PKL):
    """加载模型 (默认现役最优, 可传自定义)"""
    print("=" * 60)
    print("Step 2/3: 加载模型")
    print("=" * 60)
    for pkl in [entry_pkl, risk_pkl, opport_pkl, riskmag_pkl]:
        if not os.path.exists(pkl):
            print(f"❌ 模型缺失: {pkl}")
            sys.exit(1)
    backtest.reload_models(entry_pkl, risk_pkl)
    backtest.load_magnitude_models(opport_pkl, riskmag_pkl)
    print(f"✅ 模型已加载: {entry_pkl} + {risk_pkl}")
    print(f"✅ 幅度模型已加载: {opport_pkl} + {riskmag_pkl}")


def run_screening(trade_days, full):
    """回测选股, 输出今日信号 (含买入推荐与持仓卖出信号)"""
    print("=" * 60)
    print("Step 3/3: 回测选股")
    print("=" * 60)
    symbols = screen.basic_screen()
    if not symbols:
        print("❌ 股票池为空, 请先确认数据同步")
        sys.exit(1)
    print(f"股票池: {len(symbols)} 只")

    cal = get_trading_calendar()
    if full:
        start_date = FULL_START
        mode_desc = "完整区间 (2021 至今, 可历史比较)"
    else:
        total_bars = WARMUP_BARS + trade_days
        start_date = trading_days_back(cal, total_bars).strftime('%Y-%m-%d')
        mode_desc = f"近 {trade_days} 交易日交易期 + {WARMUP_BARS} warmup"
    # 新基线统一使用 2026-08-17 作为结束日，确保与近期实验对齐
    end_date = BASELINE_END

    print(f"回测模式: {mode_desc}")
    print(f"回测区间: {start_date} ~ {end_date} (warmup={WARMUP_BARS} 交易日)")

    suffix = os.environ.get('RESULTS_DIR_SUFFIX', '')
    results_dir = f"results/daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    backtest.run_backtest(
        symbols,
        start_date=start_date,
        end_date=end_date,
        warmup=WARMUP_BARS,
        results_dir=results_dir,
        initial_cash=float(os.environ.get('INITIAL_CASH', '1000000')),
    )
    print(f"结果目录: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description='每日选股')
    parser.add_argument('--no-data', action='store_true', help='跳过数据同步')
    parser.add_argument('--full', action='store_true', help='完整回测 2021 至今 (历史比较)')
    parser.add_argument('--days', type=int, default=120, help='交易期长度, 交易日 (默认 120 ≈ 近半年)')
    parser.add_argument('--entry-model', default=ENTRY_PKL, help=f'入场模型 pkl (默认 {ENTRY_PKL})')
    parser.add_argument('--risk-model', default=RISK_PKL, help=f'风控模型 pkl (默认 {RISK_PKL})')
    args = parser.parse_args()

    if not args.no_data:
        sync_data()
    load_models(args.entry_model, args.risk_model)
    run_screening(args.days, args.full)


if __name__ == '__main__':
    main()
