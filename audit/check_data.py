# audit/check_data.py
# 数据健壮性全量自动化审计
# 输入: debug_inference_results.csv (由 backtest 在 DEBUG_INFERENCE=1 时生成, 全市场日频推理快照)
import os
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from collections import OrderedDict

# 当前生产管线的大盘特征列 (backtest.py _fetch_data mkt_cols)
MKT_COLS = ['mkt_trend', 'mkt_vol', 'mkt_liq', 'mkt_position']

# 大盘拥挤度/广度共享常数 (daily 共享, 进入两个模型的 MKT 通道)
MKT_RATIO_COLS = ['congestion', 'high20_ratio', 'low20_ratio']


def _parse_date_col(df, col='date'):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='mixed')


def run_data_audit(file_path='debug_inference_results.csv'):
    # 终端彩色输出配置
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def print_title(text):
        print("\n" + "=" * 80)
        print(f" {BOLD}{text}{RESET}")
        print("=" * 80)

    def print_res(status, text):
        if status == "PASS":
            print(f"[{GREEN}PASS{RESET}] {text}")
        elif status == "WARN":
            print(f"[{YELLOW}WARN{RESET}] {text}")
        elif status == "FATAL":
            print(f"[{RED}FATAL{RESET}] {text}")

    print_title("数据特征库 (debug_inference_results) 健壮性自动化审计 (V7)")

    if not os.path.exists(file_path):
        print_res("FATAL", f"未找到诊断文件: '{file_path}'。")
        print("  >> 请先运行回测: DEBUG_INFERENCE=1 PYTHONPATH=. python -c \"import backtest; backtest.run_backtest(...)\"")
        print("=" * 80 + "\n")
        return

    try:
        df = pd.read_csv(file_path, dtype={'symbol': str})
        _parse_date_col(df)
    except Exception as e:
        print_res("FATAL", f"读取 CSV 文件并转换日期失败: {e}")
        return

    total_rows = len(df)
    print(f" 数据集总行数: {total_rows:,.0f}")
    print(f" 覆盖股票数量: {df['symbol'].nunique() if 'symbol' in df.columns else '未找到 symbol 列'}")
    if 'date' in df.columns:
        print(f" 时序时间跨度: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")

    # =========================================================================
    # 0. 样本主键唯一性审计
    # =========================================================================
    print_title("0. 样本唯一性主键完整性审计 (Unique Key Integrity)")
    if 'symbol' in df.columns and 'date' in df.columns:
        dup = df.duplicated(subset=['date', 'symbol']).sum()
        if dup > 0:
            print_res("FATAL", f"发现 {dup} 行 (symbol, date) 重复主键，会干扰回测开仓精度，请排查数据源拼接。")
        else:
            print_res("PASS", "(symbol, date) 主键全局唯一，无冗余样本对齐 Bug。")
        # 每日样本数分布 (覆盖率)
        daily_cnt = df.groupby('date')['symbol'].count()
        print(f" 每日横截面样本: 最小 {daily_cnt.min():.0f} / 平均 {daily_cnt.mean():.0f} / 最大 {daily_cnt.max():.0f} 只")
    else:
        print_res("WARN", "缺少 'symbol' 或 'date' 列，跳过唯一性检查。")

    # =========================================================================
    # 1. NaN / Inf 泄露审计
    # =========================================================================
    print_title("1. 特征缺失值 (NaN) 与无穷值 (Inf) 检查")
    num_cols = df.select_dtypes(include=[np.number]).columns
    invalid_mask = pd.DataFrame(False, index=df.index, columns=num_cols)
    invalid_mask = df[num_cols].isna() | np.isinf(df[num_cols].astype(float))
    invalid_cols = invalid_mask.any()

    if not invalid_cols.any():
        print_res("PASS", f"全部 {len(num_cols)} 个数值列无 NaN / ±Inf，数据清洗与防御性填充正常。")
    else:
        for col in invalid_cols.index[invalid_cols.values]:
            bad = int(invalid_mask[col].sum())
            print_res("FATAL", f"列 {col:<28} 存在 NaN/Inf: {bad} 行 ({bad/total_rows:.3f}%)")

    # =========================================================================
    # 2. 大盘环境解耦因子 + 共享常数审计
    # =========================================================================
    print_title("2. 大盘环境因子审计")
    mkt_missing = [c for c in MKT_COLS if c not in df.columns]
    if mkt_missing:
        print_res("FATAL", f"未找到大盘解耦因子: {mkt_missing}")
    else:
        for col in MKT_COLS:
            col_data = df[col].dropna().values
            std_val = np.std(col_data)
            uniq = len(np.unique(col_data))
            if std_val < 1e-4 and np.allclose(col_data, 0.5):
                print_res("FATAL", f"大盘因子 {col:<12} 静默失败！全部被 fillna(0.5) 覆盖，请检查日期类型对齐。")
            elif std_val < 1e-4:
                print_res("WARN", f"大盘因子 {col:<12} 波动偏低 (Std={std_val:.6f})，全时期近常数 {col_data[0]:.3f}")
            else:
                print_res("PASS", f"大盘因子 {col:<12} 正常。均值 {np.mean(col_data):.4f} | Std {std_val:.4f} | 唯一状态 {uniq}")

    print_title("3. 大盘拥挤度/广度共享常数审计")
    for col in MKT_RATIO_COLS:
        if col not in df.columns:
            print_res("WARN", f"缺共享常数列 {col}")
            continue
        col_data = df[col].dropna().values
        if col == 'congestion' and (np.max(col_data) < 0.3 or np.min(col_data) < 0.0):
            print_res("WARN", f"{col}: 范围 [{np.min(col_data):.4f}, {np.max(col_data):.4f}] 异常 0.35 中性占比 "
                              f"{np.mean(np.isclose(col_data, 0.35)):.1%}")
        if np.isclose(col_data, col_data[0]).mean() > 0.95 and np.std(col_data) < 1e-4:
            print_res("WARN", f"{col} 全期常数 {col_data[0]:.4f}，大盘共享常数可能未装配。")
        else:
            print_res("PASS", f"{col:<16} 正常。均值 {np.mean(col_data):.4f} | Std {np.std(col_data):.4f}")

    # =========================================================================
    # 4. 财务过滤审计
    # =========================================================================
    print_title("4. 基本面财务过滤 (is_profit_ok) 对齐审计")
    if 'is_profit_ok' not in df.columns:
        print_res("WARN", "未找到财务过滤因子 'is_profit_ok'。")
    else:
        pok = df['is_profit_ok'].dropna()
        tt = (pok.sum() / len(pok)) * 100
        if tt == 0.0:
            print_res("FATAL", "'is_profit_ok' 100% 均为 False，基本面数据对齐失败 (symbol 的 sh/sz 格式?) 全市场会被一刀切过滤。")
        elif tt == 100.0:
            print_res("WARN", "'is_profit_ok' 100% 均为 True，不符合基本面分布，请核实数据源。")
        else:
            print_res("PASS", f"'is_profit_ok' True 占比 {tt:.2f}%，分布正常。")

    # =========================================================================
    # 5. 截面 Z-Score 特征质量 + clip 边界审计
    # =========================================================================
    print_title("5. 截面 Z-Score 特征质量与 clip(-3,3) 边界审计")
    z_cols = [c for c in df.columns if c.endswith('_z')]
    z_clip_missing = ['profit_bias_div_z', 'mkt_trend']  # 复合特征 + 大盘 Z 编码示例
    if not z_cols:
        print_res("WARN", "未找到 '_z' 后缀截面标准化特征。")
    else:
        bad_z = []
        for col in z_cols:
            col_data = df[col].dropna().values
            if col_data.size == 0:
                bad_z.append((col, "全部为空"))
                continue
            mean_val, std_val = np.mean(col_data), np.std(col_data)
            if std_val < 1e-4:
                bad_z.append((col, f"Std=0 (全为常数)，单日截面样本过少或 groupby 失效"))
            elif not np.allclose(mean_val, 0.0, atol=1e-2) or not (0.3 < std_val < 1.1):
                bad_z.append((col, f"均值 {mean_val:.4f} (期望0) | Std {std_val:.4f} (期望1.0)"))
        if bad_z:
            print_res("FATAL", f"{len(bad_z)} 个截面特征异常:")
            for col, err in bad_z[:20]:
                print(f"  - {col:<26} {err}")
        else:
            print_res("PASS", f"共 {len(z_cols)} 个截面特征全部通过 (每日横截面标准化有效)。")
        # clip(-3,3) 边界审计 (co_compute.apply_standardization)
        over = {c: int((df[c].abs() > 3.01).sum()) for c in z_cols if c in df.columns}
        over = {k: v for k, v in over.items() if v > 0}
        if over:
            bad_rows = sum(over.values())
            print_res("WARN", f"{len(over)} 个特征越出 |z|>3 区间 (疑似 clip 未生效或新增特征漏 clip): 共 {bad_rows} 行")
            for k, v in list(over.items())[:10]:
                print(f"  - {k:<26} 越界 {v} 行")
        else:
            print_res("PASS", f"全部 {len(z_cols)} 个 Z 特征均在 [-3, 3] 裁剪边界内，clip 生效。")

    # =========================================================================
    # 6. 筹码特征物理边界审计
    # =========================================================================
    print_title("6. 筹码财务特征物理数值审计")
    chip_cols = ['profit_ratio', 'concentration_70']
    missing_chip = [c for c in chip_cols if c not in df.columns]
    if missing_chip:
        print_res("WARN", f"缺失基础筹码指标: {missing_chip}")
    else:
        pr = df['profit_ratio'].dropna().values
        cc = df['concentration_70'].dropna().values
        if pr.size and (pr.min() < 0.0 or pr.max() > 1.0):
            print_res("FATAL", f"获利盘比例越界 [0,1]: 当前 [{pr.min():.4f}, {pr.max():.4f}]")
        elif pr.size and np.std(pr) < 1e-4:
            print_res("FATAL", "获利盘比例全为常数 (Std=0)，Numba 筹码算法可能失效，请检查 turnover 等输入。")
        else:
            print_res("PASS", f"获利盘比例区间 [{0 if not pr.size else pr.min():.4f}, {1 if not pr.size else pr.max():.4f}]，物理合法。")
        if cc.size and (cc.min() < 0.0 or cc.max() > 1.0):
            print_res("FATAL", f"筹码集中度越界 [0,1]: 当前 [{cc.min():.4f}, {cc.max():.4f}]")
        else:
            print_res("PASS", f"筹码集中度区间 [{0 if not cc.size else cc.min():.4f}, {1 if not cc.size else cc.max():.4f}]，物理合法。")

    # 停牌天数 (若存在)
    if 'suspension_duration' in df.columns:
        susp = df['suspension_duration'].dropna().values
        if susp.size and susp.min() < 0.0:
            print_res("FATAL", f"停牌天数存在负数: min={susp.min():.2f}")
        else:
            print_res("PASS", f"停牌天数正常 (均值 {np.mean(susp):.2f} 日, 最长 {np.max(susp):.0f} 日)")
    if 'is_suspended' in df.columns and df['is_suspended'].sum() > 0:
        print_res("FATAL", f"仍有 {df['is_suspended'].sum()} 行停牌日残留在回测集。")

    # 面值过滤
    if 'close' in df.columns:
        low = df[df['close'] < 1.0]
        if not low.empty:
            print_res("FATAL", f"仍残存 {len(low)} 行 <1.0 元退市风险价格。")
        else:
            print_res("PASS", "已全部拦截 1.0 元以下退市风险仙股。")

    # =========================================================================
    # 7. 双通道模型得分审计 (raw_ml_score 入场 / risk_ml_score 风控)
    # =========================================================================
    print_title("7. 双通道模型得分审计 (raw_ml_score 入场 + risk_ml_score 风控)")
    for col, label in [('raw_ml_score', '入场'), ('risk_ml_score', '风控')]:
        if col not in df.columns:
            print_res("FATAL", f"未找到模型得分列 {col}，模型推理未完成！")
            continue
        s = df[col].dropna()
        std_all = s.std()
        # 每日截面 std 的平均值: 衡量当日排序区分度 (若某日常数 → 无排序能力)
        daily_std = df.groupby('date')[col].std().dropna()
        if std_all < 1e-4:
            print_res("FATAL", f"{label}模型分 {col} 全局 Std=0 (全为常数)，特征对齐失败。")
        elif daily_std.mean() < 1e-4:
            print_res("WARN", f"{label}模型分 {col} 单日截面 Std≈0，模型丧失日频排序能力。")
        else:
            print_res("PASS", f"{label}模型分 {col}: 全局 Std={std_all:.4f} | 日截面 Std 均值={daily_std.mean():.4f} | 范围 [{s.min():.4f}, {s.max():.4f}]")
            # 2/98 分位看尾部污染的占比
            q02, q98 = s.quantile(0.02), s.quantile(0.98)
            if q02 == q98:
                print_res("WARN", f"{col} 2% 与 98% 分位数重合，分分布极端收缩。")

    # =========================================================================
    # 8. 排序指标审计 (ml_rank / risk_ml_rank) 与平滑相关性 (raw_ml_score vs risk_ml_score)
    # =========================================================================
    print_title("8. 排名指标与双模型独立性审计")
    for col in ['ml_rank', 'risk_ml_rank']:
        if col in df.columns:
            r = df[col].dropna()
            if (r < 0).any() or (r > 1).any():
                print_res("FATAL", f"{col} 越出 [0,1]，rank(pct=True) 编码错误。")
            else:
                print_res("PASS", f"{col}: 范围 [{r.min():.4f}, {r.max():.4f}]，每日截面 pct-rank 合法。")
        else:
            print_res("WARN", f"缺少排名列 {col}。")

    if {'raw_ml_score', 'risk_ml_score'}.issubset(df.columns):
        corr = df.groupby('date').apply(
            lambda g: spearmanr(g['raw_ml_score'], g['risk_ml_score'])[0] if len(g) > 20 else np.nan,
            include_groups=False
        ).mean()
        if abs(corr) > 0.7:
            print_res("WARN", f"入场/风控模型日截面 Rank 相关 {corr:.3f} (>0.7)，风控可能只是 Alpha 反向表达，缺独立防守价值。")
        else:
            print_res("PASS", f"入场/风控模型日截面 RankIC 相关 {corr:.3f}，双塔保持独立性。")

    print("\n" + "=" * 80)
    print(" 诊断完成！若存在 [FATAL] 标记，请优先修复策略或特征工程。")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_data_audit()