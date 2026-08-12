import pandas as pd
import numpy as np
import os

# 配置路径
GLOBAL_FILE = 'global_strategy_audit.csv'
INDIVIDUAL_FILE = 'individual_stocks_audit_filtered.csv'

def run_audit():
    print("="*60)
    print(" 策略海选逻辑深度审计报告 ")
    print("="*60)

    if not os.path.exists(GLOBAL_FILE) or not os.path.exists(INDIVIDUAL_FILE):
        print("错误：未找到审计CSV文件。")
        return

    # 1. 加载数据 (强制 symbol 为补零字符串)
    df_g = pd.read_csv(GLOBAL_FILE, index_col=0, parse_dates=True)
    df_i = pd.read_csv(INDIVIDUAL_FILE, parse_dates=['date'], dtype={'symbol': str})
    df_i['symbol'] = df_i['symbol'].str.zfill(6)

    # ==========================================
    # A. 漏网之鱼：首层物理过滤审计
    # ==========================================
    # 注意：代码开头有一个 if len < 90 or price <= 2: return False
    # 这部分股票不会出现在 CSV 里。
    print(f"\n[1. 数据覆盖率]")
    print(f"每日平均海选样本数: {df_i.groupby('date')['symbol'].count().mean():.0f} 只")
    print("注：由于代码首行过滤了上市不满90天或股价<=2元的股票，上述样本已排除此类个股。")

    # ==========================================
    # B. 过滤器效果审计 (由易到难)
    # ==========================================
    print(f"\n[2. 过滤器拦截归因 (漏斗分析)]")
    total = len(df_i)
    
    # 定义各层拦截原因（注意这里要和你的代码逻辑对应）
    df_i['fail_active'] = ~df_i['is_active']
    df_i['fail_profit'] = ~df_i['is_profit_ok']
    df_i['fail_ml_score'] = df_i['ml_score'] <= df_i['ml_threshold']
    df_i['fail_position'] = df_i['close_too_high'] | df_i['is_crashing']
    df_i['fail_market'] = ~df_i['market_ok']
    
    def print_stat(name, series):
        count = series.sum()
        pct = (count / total) * 100
        print(f" - {name:20}: 拦截 {count:8} 次 | 占比 {pct:6.2f}%")

    print_stat("流动性/波动率不足", df_i['fail_active'])
    print_stat("基本面(ROE/净利)不合格", df_i['fail_profit'])
    print_stat("模型评分未达标", df_i['fail_ml_score'])
    print_stat("位置过高或崩盘坠落", df_i['fail_position'])
    print_stat("大盘择时拦截", df_i['fail_market'])

    # ==========================================
    # C. 逻辑一致性校验
    # ==========================================
    # 你的最终逻辑是：is_eligible = is_chip_ready and can_buy_in_this_market and is_active
    # 注意：is_chip_ready 内部还包含了 congestion_too_high，但 CSV 没存这个布尔值
    print(f"\n[3. 决策一致性校验]")
    # 检查是否有 eligible 为 True 但 active 为 False 的
    bad_eligible = df_i[(df_i['is_eligible'] == True) & (df_i['is_active'] == False)]
    if not bad_eligible.empty:
        print(f"❌ 严重错误：发现 {len(bad_eligible)} 条记录流动性不足却入选了！")
    else:
        print("✅ 流动性一致性检查通过。")

    # ==========================================
    # D. 数值范围审计 (数值溢出/量级检查)
    # ==========================================
    print(f"\n[4. 核心数值异常监控]")
    
    # 1. 检查成交额单位 (看 liq_limit 是否在合理量级)
    med_liq = df_g['thresh_amount_ma20'].median()
    print(f"成交额门槛中位数: {med_liq:,.0f} (若此值仅为几千，请检查数据单位是否为万元)")
    
    # 2. 检查 ML Score 离群值
    ml_max = df_i['ml_score'].max()
    ml_min = df_i['ml_score'].min()
    print(f"ML 分数范围: {ml_min:.2f} ~ {ml_max:.2f}")
    if abs(ml_max) > 10:
        print("⚠️ 警告：ML 分数出现异常极值，请检查推理逻辑。")

    # ==========================================
    # E. 选股对齐 (Cross-Check)
    # ==========================================
    print(f"\n[5. 最终买入席位验证]")
    # 验证 before_exec_fn 是否真的选了 eligible 里面分最高的
    sample_dates = df_g[df_g['strat_top_x_buys'].notna()].index.unique()
    if not sample_dates.empty:
        test_dt = sample_dates[-1]
        top_str = df_g.loc[test_dt, 'strat_top_x_buys']
        top_list = sorted([s.strip().zfill(6) for s in str(top_str).split('|')])
        
        candidates = df_i[(df_i['date'] == test_dt) & (df_i['is_eligible'] == True)]
        # 排除掉 top_list 长度为 0 的情况
        if len(top_list) > 0:
            expected = sorted(candidates.sort_values('ml_score', ascending=False).head(len(top_list))['symbol'].tolist())
            if top_list == expected:
                print(f"✅ {test_dt.date()} 选股对齐：Top-{len(top_list)} 符合 ML 排序。")
            else:
                print(f"⚠️ {test_dt.date()} 选股对齐异常：实际买入 {top_list}，高分合格者 {expected}")

    print("\n" + "="*60)

if __name__ == "__main__":
    run_audit()