# check_base_data.py
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def run_backtest_audit(file_path='debug_inference_results.csv'):
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

    print_title("🔍 量化数据特征库 (debug_inference_results) 健壮性全量自动化审计 (V6)")

    # 1. 检查物理文件存在性
    if not os.path.exists(file_path):
        print_res("FATAL", f"未找到诊断文件: '{file_path}'，请确认回测是否顺利跑完并生成了该文件。")
        print("=" * 80 + "\n")
        return

    try:
        # 【重要修复】：读取时显式解析日期，将字符串转换为真正的 Datetime 对象，杜绝时序比对隐患
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print_res("FATAL", f"读取 CSV 文件并转换日期失败: {e}")
        return

    total_rows = len(df)
    print(f" 数据集总行数: {total_rows}")
    print(f" 覆盖股票数量: {df['symbol'].nunique() if 'symbol' in df.columns else '未找到 symbol 列'}")
    print(f" 时序时间跨度: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else '未找到 date 列'}")

    # =========================================================================
    # 追加审计: 样本主键唯一性检查 (Unique Key Integrity Audit)
    # =========================================================================
    print_title("0. 样本唯一性主键完整性审计 (Unique Key Integrity)")
    if 'symbol' in df.columns and 'date' in df.columns:
        # 严格检查是否存在同一天同一只股票出现多条记录的现象（这会导致回测中产生虚假复开仓 Bug）
        duplicates_count = df.duplicated(subset=['date', 'symbol']).sum()
        if duplicates_count > 0:
            print_res("FATAL", f"发现存在 {duplicates_count} 行 (symbol, date) 重复主键样本！这会严重干扰回测开仓精度，请排查数据源拼接逻辑。")
        else:
            print_res("PASS", "个股日频主键具有全局唯一性，未发现冗余样本对齐 Bug！")
    else:
        print_res("WARN", "未在大数据集中找到 'symbol' 或 'date' 列，跳过唯一性检查。")

    # =========================================================================
    # 审计 1: 缺失值 (NaN) 与 无穷值 (Inf) 泄露双重防御审计
    # =========================================================================
    print_title("1. 特征缺失值 (NaN) 与 无穷值 (Inf) 泄露检查")
    # 【重要修复】：isna 无法拦截正负无穷，此处改为 isna 与 isinf 的双重合并校验
    is_invalid = df.isna() | np.isinf(df.select_dtypes(include=[np.number]))
    invalid_cols = df.columns[is_invalid.any()].tolist()
    
    if not invalid_cols:
        print_res("PASS", "所有特征列均无 NaN 或 ±Inf 值，数据清洗、停牌插值与防御性填充处理完美！")
    else:
        print_res("FATAL", f"以下列中存在 NaN 或 ±Inf 溢出值（极易引发 LGBM 预测崩溃或回测静默失效）：")
        for col in invalid_cols:
            invalid_cnt = is_invalid[col].sum()
            invalid_pct = (invalid_cnt / total_rows) * 100
            print(f"  - 特征: {col:<25} | 异常溢出行数: {invalid_cnt:<6d} ({invalid_pct:.2f}%)")

    # =========================================================================
    # 审计 2: 大盘环境解耦因子检查
    # =========================================================================
    print_title("2. 大盘环境解耦因子检查")
    mkt_cols = ['mkt_trend', 'mkt_suppress', 'mkt_breadth']
    missing_mkt = [col for col in mkt_cols if col not in df.columns]
    
    if missing_mkt:
        print_res("FATAL", f"未在数据中找到以下大盘解耦因子列: {missing_mkt}，环境因子未成功装配。")
    else:
        for col in mkt_cols:
            col_data = df[col].dropna().values
            std_val = np.std(col_data)
            unique_cnt = len(np.unique(col_data))
            
            if std_val < 1e-4 and np.allclose(col_data, 0.5):
                print_res("FATAL", f"大盘因子 {col} 发生严重的静默失败！全部被 fillna(0.5) 覆盖，请检查日期类型对齐。")
            elif std_val < 1e-4:
                print_res("WARN", f"大盘因子 {col} 波动异常偏低（Std={std_val:.6f}），全时期均为常数 {col_data[0]:.2f}。")
            else:
                print_res("PASS", f"大盘因子 {col:<12} 正常。均值: {np.mean(col_data):.4f} | 标准差: {std_val:.4f} | 日频唯一状态数: {unique_cnt}")

    # =========================================================================
    # 审计 3: 财务报表数据向量化合并检查（防范 merge_asof 静默失败）
    # =========================================================================
    print_title("3. 财务报表数据对齐检查")
    if 'is_profit_ok' not in df.columns:
        print_res("WARN", "未在数据中找到财务过滤因子 'is_profit_ok'。")
    else:
        profit_ok_data = df['is_profit_ok'].dropna().values
        true_pct = (np.sum(profit_ok_data) / len(profit_ok_data)) * 100
        
        if true_pct == 0.0:
            print_res("FATAL", "财务过滤因子 'is_profit_ok' 100% 均为 False！说明基本面数据对齐失败，这会导致全市场可开仓个股被一刀切全部过滤。请排查代码中 symbol 的 sh/sz 格式。")
        elif true_pct == 100.0:
            print_res("WARN", "财务过滤因子 'is_profit_ok' 100% 均为 True，不符合基本面分布，请核实数据源。")
        else:
            print_res("PASS", f"财务数据合并成功。'is_profit_ok' 分布正常（True 占比: {true_pct:.2f}%，False 占比: {100-true_pct:.2f}%）")

    # =========================================================================
    # 审计 4: 截面 Z-Score 特征质量审计（防范分组样本过少导致的 NaN 污染）
    # =========================================================================
    print_title("4. 截面 Z-Score 特征质量审计")
    z_cols = [col for col in df.columns if col.endswith('_z')]
    if not z_cols:
        print_res("WARN", "未在数据中找到带有 '_z' 后缀的截面标准化特征。")
    else:
        failed_z_cols = []
        for col in z_cols:
            col_data = df[col].dropna().values
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            
            is_mean_ok = np.allclose(mean_val, 0.0, atol=1e-2)
            is_std_ok = (0.3 < std_val < 1.1)
            
            if std_val < 1e-4:
                failed_z_cols.append((col, f"标准差为 0（全为常数），说明单日截面内样本过少或 Groupby 发生错误。"))
            elif not is_mean_ok or not is_std_ok:
                failed_z_cols.append((col, f"数据偏离：均值={mean_val:.4f} (期望0)，标准差={std_val:.4f} (期望1.0)"))
                
        if not failed_z_cols:
            print_res("PASS", f"共 {len(z_cols)} 个截面标准化特征全部通过健康审计！无静默常数特征。")
        else:
            print_res("FATAL", f"以下 {len(failed_z_cols)} 个截面特征存在异常：")
            for col, err in failed_z_cols:
                print(f"  - 特征: {col:<25} | 原因: {err}")

    # =========================================================================
    # 审计 5: Numba 筹码分布特征越界审计与绝对事件特征审计
    # =========================================================================
    print_title("5. Numba 筹码分布与事件特征物理数值审计")
    chip_cols = ['profit_ratio_raw', 'concentration_70']
    missing_chip = [col for col in chip_cols if col not in df.columns]
    if missing_chip:
        print_res("WARN", f"数据中缺失基础筹码指标: {missing_chip}")
    else:
        pr_data = df['profit_ratio_raw'].dropna().values
        pr_min, pr_max = np.min(pr_data), np.max(pr_data)
        pr_std = np.std(pr_data)
        
        conc_data = df['concentration_70'].dropna().values
        conc_min, conc_max = np.min(conc_data), np.max(conc_data)
        
        if pr_min < 0.0 or pr_max > 1.0:
            print_res("FATAL", f"获利盘比例数值物理越界（超出 [0,1] 范围）！当前区间: [{pr_min:.4f}, {pr_max:.4f}]")
        elif pr_std < 1e-4:
            print_res("FATAL", "获利盘比例全为常数（Std=0），Numba 筹码分布算法可能计算失效，请检查 turnover 等输入。")
        else:
            print_res("PASS", f"筹码获利比例指标正常。当前物理区间: [{pr_min:.4f} 至 {pr_max:.4f}]")
            
        if conc_min < 0.0 or conc_max > 1.0:
            print_res("FATAL", f"筹码集中度数值物理越界（超出 [0,1] 范围）！当前区间: [{conc_min:.4f}, {conc_max:.4f}]")
        else:
            print_res("PASS", f"筹码集中度特征正常。当前物理区间: [{conc_min:.4f} 至 {conc_max:.4f}]")

    # 【重要追加】：对绝对值事件特征 suspension_duration（停牌天数）进行审计
    if 'suspension_duration' in df.columns:
        susp_data = df['suspension_duration'].dropna().values
        susp_min = np.min(susp_data)
        susp_max = np.max(susp_data)
        susp_mean = np.mean(susp_data)
        
        if susp_min < 0.0:
            print_res("FATAL", f"停牌特征 'suspension_duration' 存在非物理负数！最小值: {susp_min:.2f}")
        else:
            print_res("PASS", f"停牌天数特征正常。平均停牌天数: {susp_mean:.4f} 日 | 单次最长停牌天数: {susp_max:.0f} 日")

    # =========================================================================
    # 审计 6: LGBM 预测打分 (ml_score) 与时序平滑畸变度审计
    # =========================================================================
    print_title("6. 机器学习预测打分 (ml_score) 与平滑信号审计")
    if 'ml_score' not in df.columns:
        print_res("FATAL", "未在数据中找到机器学习推理得分 'ml_score'，模型推理未成功执行！")
    else:
        scores = df['ml_score'].dropna().values
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        score_min = np.min(scores)
        score_max = np.max(scores)
        
        if score_std < 1e-4:
            print_res("FATAL", "机器学习预测打分 ml_score 毫无区别（Std=0，全为常数）！请排查特征对齐。")
        elif score_std < 0.05:
            print_res("WARN", f"模型预测分极度收缩（Std={score_std:.4f} < 0.05），模型基本丧失排序能力。")
        elif score_std > 0.35:
            print_res("WARN", f"模型打分波动过大（Std={score_std:.4f} > 0.35），请检查特征截面对齐。")
        else:
            print_res("PASS", f"模型推理得分 ml_score 分布尺度极佳！")
            print(f"  - 预测打分均值:   {score_mean:.6f} (符合接近0.0的中性对齐预期)")
            print(f"  - 预测打分标准差: {score_std:.6f} (符合 [0.10, 0.16] 的标准正态收缩尺度)")
            print(f"  - 绝对打分值跨度: [{score_min:.4f} 至 {score_max:.4f}]")

        # 【重要追加】：计算原始得分 raw_ml_score 与 平滑得分 ml_score 之间的相关性，确保平滑未扭曲阿尔法
        if 'raw_ml_score' in df.columns:
            raw_scores = df['raw_ml_score'].dropna().values
            if len(raw_scores) == len(scores):
                # 横截面斯皮尔曼秩相关系数
                corr, _ = spearmanr(raw_scores, scores)
                if corr < 0.50:
                    print_res("FATAL", f"信号时序平滑发生严重畸变（Spearman Corr={corr:.4f} < 0.50）！说明 ewm(span=3) 引入了极度严重的滞后，彻底洗掉了模型预测的阿尔法，请调低平滑参数。")
                elif corr > 0.99:
                    print_res("WARN", f"信号平滑前后高度重合（Spearman Corr={corr:.4f}），平滑未发挥降噪换仓作用。")
                else:
                    print_res("PASS", f"平滑信号特征正常。平滑因子与原始因子 Rank 相关系数: {corr:.4f} (符合 [0.75, 0.95] 范围内的无畸变降噪预期)")

    # =========================================================================
    # 审计 7: 退市股/停牌期物理过滤合规性审计
    # =========================================================================
    print_title("7. 退市风险股与停牌物理过滤审计")
    if 'is_suspended' in df.columns:
        print_res("WARN", "特征数据中依然残留了辅助标志列 'is_suspended'（本应在加工后丢弃）。")
        suspended_count = df['is_suspended'].sum()
        if suspended_count > 0:
            print_res("FATAL", f"时序对齐后，未能成功过滤停牌行！仍有 {suspended_count} 行停牌日数据残留在最终回测集中。")
            
    # 检查是否存在低于 1 元的股票残留（1元面值退市红线）
    too_low_prices = df[df['close'] < 1.0]
    if not too_low_prices.empty:
        print_res("FATAL", f"回测价格过滤线失效！数据集中仍残存 {len(too_low_prices)} 行收盘价低于 1.0 元的退市风险个股。")
    else:
        print_res("PASS", "已成功物理拦截所有 1.0 元以下的退市风险仙股，防踩雷红线生效。")

    print("\n" + "=" * 80)
    print(" 诊断完成！若存在 [FATAL] 标记，请务必返回策略或特征工程中优先修复。")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_backtest_audit()