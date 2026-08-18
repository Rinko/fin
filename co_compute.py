from numba import njit
import numpy as np
import logging
import os
import pandas as pd
import talib
import sqlite3
import logging


class FeatureConfig:
    # 1. 原始业务特征 (需要进行横截面 Z-Score 的列)
    BIZ_FEATURES = [
        'ema_profit', 'res_profit', 
        'res_conc_90', 'ema_conc_70',      
        'ema_conc_90_v',                   
        'ema_peak_density',                
        'dist_to_avg', 'dist_to_high90',   
        'ema_penetrate_up', 'ema_decay_dn',
        'ema_vol_stab', 'res_vol_stab',
        'ema_vp_corr', 'res_vp_corr',
        'ema_cost_v',
        'ema_turnover_vol', 'ema_turnover_max_res',
        'ema_bias_norm', 'res_bias_norm',
        'acc_confirm', 'vp_diverg'
    ]
    BIZ_RISK_FEATURES = [
        'ema_profit', 'res_profit',
        'res_conc_90', 'ema_conc_70',
        'ema_conc_90_v',
        'ema_peak_density',
        'dist_to_avg', 'dist_to_high90',
        'ema_penetrate_up', 'ema_decay_dn',
        'ema_vol_stab', 'res_vol_stab',
        'ema_vp_corr',  # 风控通道不使用 res_vp_corr：raw 通道下 res_vp_corr = vp_corr - ema_vp_corr = 0
        'ema_cost_v',
        'ema_turnover_vol', 'ema_turnover_max_res',
        'ema_bias_norm', 'res_bias_norm',
        'stock_congestion', 'high_vol_interaction', 'vp_corr_decay',
        'acc_confirm', 'vp_diverg'
    ]

    # 2. 大盘环境特征 (直接使用的列，不参与个股 Z-Score)
    # 原始大盘列，用于计算复合特征和供给 is_market_ok / 个股动态窗口使用
    MKT_RAW_FEATURES = [
        'mkt_trend', 'mkt_vol', 'mkt_liq', 'mkt_position',
        'congestion', 'high20_ratio', 'low20_ratio'
    ]
    # 喂入模型的大盘特征：使用滚动 PCA 生成的 3 个正交、可解释的市场因子。
    # 命名对应其主导业务含义，替代高度相关原始大盘列。
    MKT_FEATURES = [
        'mkt_macro_regime',      # 宏观/资金/趋势综合状态
        'mkt_breadth_spread',    # 新高 vs 新低股的广度结构
        'mkt_congestion_pressure' # 市场拥挤度/内部结构压力
    ]

    # 2.6 外部预计算 PCA 表路径。
    # 实验/生产可通过该字段强制训练与回测使用同一张 PC 表，避免手动 merge parquet 或 monkeypatch。
    # 为 None 时按默认逻辑从 MKT_RAW_FEATURES 实时计算。
    PC_TABLE_PATH = None

    # 2.5 行业内分位特征 (横截面 rank, 天然 0~1, 无需 Z-Score, 不加权)
    # ind_inner_rank: 个股 rs_20 在所属申万一级行业内的分位排名 (个股级)
    # ind_rank_20/60: 31 个申万一级行业按行业RS排名的分位 (全市场共享常数, 行业级)
    # ⚠️ 2026-08 gate 均 FAIL: ind_inner_rank 与 rs_20 相关 0.72 (重复信息);
    #    ind_rank 单因子 IC 显著为负 (-0.043) 且模型内增量仅 +0.0012。
    #    现置空防污染, 计算代码 (compute_industry_inner_rank/map_industry_rank) 保留待重启用。
    RANK_FEATURES = []

    # 3. 模型最终喂入的列名 (自动生成)
    @classmethod
    def get_model_input_features(cls):
        # 个股特征全部带上 _z 后缀
        z_features = [f"{col}_z" for col in cls.BIZ_FEATURES]
        # 加上复合特征
        composite = ['profit_bias_div_z']
        # 加上大盘特征
        return z_features + composite + cls.MKT_FEATURES + cls.RANK_FEATURES

    @classmethod
    def get_risk_model_input_features(cls):
        # 个股特征全部带上 _z 后缀
        z_features = [f"{col}_z" for col in cls.BIZ_RISK_FEATURES]
        # 加上复合特征
        composite = ['profit_bias_div_z']
        # 加上大盘特征
        return z_features + composite + cls.MKT_FEATURES + cls.RANK_FEATURES
    
# ==========================================
# 大盘复合特征构造
# ==========================================
def add_market_composite_features(df):
    """
    从原始大盘特征构造复合特征，减少高相关市场因子对模型的分裂点占用。
    要求 df 中已包含 MKT_RAW_FEATURES 列。
    """
    # 流动性/波动/仓位体制：三者高度相关，共同描述市场整体资金环境
    df['mkt_liquidity_regime'] = df[['mkt_vol', 'mkt_liq', 'mkt_position']].mean(axis=1)
    # 拥挤度 + 新高/新低广度：描述市场内部结构压力
    df['mkt_breadth_stress'] = df[['congestion', 'high20_ratio', 'low20_ratio']].mean(axis=1)
    return df


def add_market_pca_features(df, n_components=3, min_periods=60):
    """
    基于原始大盘特征做滚动 PCA，生成 3 个具有业务可解释性的正交市场特征。

    参数:
    - df: 必须包含 'date' 列以及 FeatureConfig.MKT_RAW_FEATURES 列。
    - n_components: 保留的主成分数（默认 3）。
    - min_periods: 最少需要多少历史样本才开始估计 PCA。

    说明:
    - 每一天的 PC 仅使用当日之前的历史数据拟合，避免未来信息泄露。
    - 符号按“载荷绝对值最大的原始特征为正”锚定，保证跨日期命名含义稳定。
    - 返回 df 新增 mkt_macro_regime, mkt_breadth_spread, mkt_congestion_pressure 列。
    """
    cols = FeatureConfig.MKT_RAW_FEATURES
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"add_market_pca_features 缺少大盘列: {missing}")

    df = df.sort_values('date').copy().reset_index(drop=True)
    X = df[cols].ffill().fillna(0.0).values.astype(np.float64)
    n = len(df)
    pc_names = ['mkt_macro_regime', 'mkt_breadth_spread', 'mkt_congestion_pressure']
    if n_components != len(pc_names):
        pc_names = [f"mkt_pc{i+1}" for i in range(n_components)]
    # 每个主成分按预设的经济含义锚定符号，使命名跨日期稳定
    sign_anchors = ['mkt_trend', 'high20_ratio', 'congestion']
    pcs = np.full((n, n_components), np.nan, dtype=np.float64)

    for i in range(min_periods, n):
        X_hist = X[:i]
        if np.isnan(X_hist).any():
            continue
        mu = np.mean(X_hist, axis=0)
        sigma = np.std(X_hist, axis=0)
        sigma[sigma == 0] = 1.0
        X_std = (X_hist - mu) / sigma
        # 历史样本不足或特征维度不够时跳过
        if X_std.shape[0] < n_components or X_std.shape[1] < n_components:
            continue
        try:
            # 使用 SVD 手工实现 PCA，避免引入 sklearn 依赖
            U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
            comps = Vt[:n_components, :].copy()  # (n_components, p)
            # 符号对齐：按预设业务锚定列强制为正，确保
            # mkt_macro_regime / mkt_breadth_spread / mkt_congestion_pressure
            # 的经济含义在不同交易日保持一致。
            for k in range(n_components):
                if k < len(sign_anchors) and sign_anchors[k] in cols:
                    anchor_idx = cols.index(sign_anchors[k])
                else:
                    anchor_idx = int(np.argmax(np.abs(comps[k])))
                if comps[k, anchor_idx] < 0:
                    comps[k] *= -1.0
            x_cur = (X[i] - mu) / sigma
            pcs[i] = x_cur @ comps.T
        except np.linalg.LinAlgError:
            continue

    for k, name in enumerate(pc_names):
        df[name] = pcs[:, k]
    return df


def build_market_pca_table(mkt_raw_df, min_periods=60, pc_table_path=None):
    """
    统一入口：生成 PCA 市场特征表。

    参数:
    - mkt_raw_df: 必须包含 'date' 列以及 FeatureConfig.MKT_RAW_FEATURES 列。
    - min_periods: PCA 最小历史样本数（实时计算时使用）。
    - pc_table_path: 外部预计算 PC 表路径。为 None 时优先读取
      FeatureConfig.PC_TABLE_PATH；均未设置时按默认从 MKT_RAW_FEATURES 实时计算。
      预计算表必须包含 'date' 列与 FeatureConfig.MKT_FEATURES 列。

    返回:
    - DataFrame，只含 ['date'] + FeatureConfig.MKT_FEATURES。
    """
    # 优先使用外部预计算 PC 表，确保训练/回测大盘特征口径一致
    if pc_table_path is None:
        pc_table_path = FeatureConfig.PC_TABLE_PATH

    if pc_table_path is not None:
        if not os.path.exists(pc_table_path):
            raise FileNotFoundError(f"预计算 PC 表不存在: {pc_table_path}")
        pc_df = pd.read_parquet(pc_table_path)
        if 'date' not in pc_df.columns:
            raise ValueError(f"预计算 PC 表缺少 'date' 列: {pc_table_path}")
        pc_df['date'] = pd.to_datetime(pc_df['date'])
        pc_cols = list(FeatureConfig.MKT_FEATURES)
        missing = [c for c in pc_cols if c not in pc_df.columns]
        if missing:
            raise ValueError(f"预计算 PC 表缺少 MKT_FEATURES 列 {missing}: {pc_table_path}")
        # 只保留输入日期所需的最小集合，避免依赖多余未来数据
        df = mkt_raw_df[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.merge(pc_df[['date'] + pc_cols], on='date', how='left')
        if df[pc_cols].isna().all().any():
            logging.warning("预计算 PC 表与输入日期存在交集缺失，对应 MKT_FEATURES 将填充 0")
            df[pc_cols] = df[pc_cols].fillna(0.0)
        return df

    cols = FeatureConfig.MKT_RAW_FEATURES
    missing = [c for c in cols if c not in mkt_raw_df.columns]
    if missing:
        raise ValueError(f"build_market_pca_table 缺少必要大盘列: {missing}")

    df = mkt_raw_df.sort_values('date').copy().reset_index(drop=True)

    # 业务中性填充：只有开头因无历史数据而缺失的少量行会被填充
    fill_values = {
        'mkt_trend': 0.5, 'mkt_vol': 0.5, 'mkt_liq': 0.5, 'mkt_position': 0.5,
        'congestion': 0.35, 'high20_ratio': 0.1, 'low20_ratio': 0.1
    }
    for col in cols:
        df[col] = df[col].ffill().fillna(fill_values.get(col, 0.5))

    df = add_market_pca_features(
        df, n_components=len(FeatureConfig.MKT_FEATURES), min_periods=min_periods
    )

    # 对 PCA 大盘特征做时间序列 Z-Score（只用历史数据，无未来信息）。
    # 这样市场特征和个股 Z-Score 特征同尺度，等权 feature_contri 才公平。
    for col in FeatureConfig.MKT_FEATURES:
        pc = df[col]
        mean = pc.expanding(min_periods=min_periods).mean().shift(1)
        std = pc.expanding(min_periods=min_periods).std().shift(1)
        std = std.replace(0, np.nan)
        df[col] = ((pc - mean) / std).fillna(0.0).clip(-3, 3)

    return df[['date'] + list(FeatureConfig.MKT_FEATURES)]


def build_market_raw_z_table(mkt_raw_df, min_periods=60):
    """
    统一入口：对原始大盘特征各自做时间序列 Z-Score。

    参数:
    - mkt_raw_df: 必须包含 'date' 列以及 FeatureConfig.MKT_RAW_FEATURES 列。
    - min_periods: 滚动/扩展统计最小样本数。

    返回:
    - DataFrame，只含 ['date'] + FeatureConfig.MKT_FEATURES（即 *_z 列）。
    """
    cols = FeatureConfig.MKT_RAW_FEATURES
    missing = [c for c in cols if c not in mkt_raw_df.columns]
    if missing:
        raise ValueError(f"build_market_raw_z_table 缺少必要大盘列: {missing}")

    df = mkt_raw_df.sort_values('date').copy().reset_index(drop=True)

    fill_values = {
        'mkt_trend': 0.5, 'mkt_vol': 0.5, 'mkt_liq': 0.5, 'mkt_position': 0.5,
        'congestion': 0.35, 'high20_ratio': 0.1, 'low20_ratio': 0.1
    }
    for col in cols:
        df[col] = df[col].ffill().fillna(fill_values.get(col, 0.5))

    out_cols = []
    for col in cols:
        z_name = f"{col}_z"
        s = df[col]
        mean = s.expanding(min_periods=min_periods).mean().shift(1)
        std = s.expanding(min_periods=min_periods).std().shift(1)
        std = std.replace(0, np.nan)
        df[z_name] = ((s - mean) / std).fillna(0.0).clip(-3, 3)
        out_cols.append(z_name)

    return df[['date'] + out_cols]


# ==========================================
# Numba 核心算子：动态可变窗口 EMA
# ==========================================
@njit
def dynamic_ema(x, windows):
    n = len(x)
    out = np.full(n, np.nan)  # 使用 full 初始化更简洁
    
    start_idx = -1
    for i in range(n):
        if not np.isnan(x[i]):
            start_idx = i
            break
            
    if start_idx == -1:
        return out
        
    out[start_idx] = x[start_idx]
    
    for i in range(start_idx + 1, n):
        if np.isnan(x[i]):
            # 场景 1：停牌。保持昨天的 EMA 值
            out[i] = out[i-1]
        else:
            # 获取并校验窗口大小
            w = windows[i]
            
            # --- 关键修改：处理 windows[i] 为 NaN 的情况 ---
            if np.isnan(w):
                # 如果窗口无效，此时面临选择：
                # A. 继承昨天 (out[i] = out[i-1]) —— 推荐
                # B. 断掉 (out[i] = np.nan)
                out[i] = out[i-1]
                continue

            if w < 2.0: w = 2.0
            alpha = 2.0 / (w + 1.0)
            
            # --- 关键修改：检查前值是否有效 ---
            if np.isnan(out[i-1]):
                # 场景 2：如果之前因为某种原因断了，重新开始
                out[i] = x[i]
            else:
                # 正常计算
                out[i] = alpha * x[i] + (1.0 - alpha) * out[i-1]
                
    return out

# ==========================================
# Numba 核心算子：计算筹码相关信息
# ==========================================
@njit
def calculate_chip_metrics_numba(close_arr, vwap_arr, high_arr, low_arr, turnover_arr):
    n = len(close_arr)
    if n == 0:
        empty = np.zeros(0)
        # 【修正 Bug 2】补齐至 9 个空数组返回，确保 Numba 静态类型推导编译通过
        return empty, empty, empty, empty, empty, empty, empty, empty, empty
    
    # 1. 静态超宽网格配置
    base_price = 0.0001
    max_price = 10000000.0
    ratio = 1.001
    log_ratio = np.log(ratio)
    # 增加微小偏移 1e-9 防止浮点数 int 转换下取整误差
    bins_count = int(np.log(max_price / base_price) / log_ratio + 1e-9) + 1

    # 预分配
    bin_prices = np.zeros(bins_count)
    for b in range(bins_count):
        bin_prices[b] = base_price * (ratio ** b)
        
    chips = np.zeros(bins_count)
    cum_chips = np.zeros(bins_count)

    # 输出指标
    out_profit_ratio = np.zeros(n)
    out_avg_cost = np.zeros(n)
    out_cost_90_low = np.zeros(n)
    out_cost_90_high = np.zeros(n)
    out_conc_90 = np.zeros(n)
    out_cost_70_low = np.zeros(n)
    out_cost_70_high = np.zeros(n)
    out_conc_70 = np.zeros(n)
    # 【修正 Bug 1】移出循环，在外部完成预分配，保障高性能与正确的时序数据写入
    out_peak_density = np.zeros(n) 
    
    total_weight_sum = 0.0
    total_cost_product = 0.0

    for i in range(n):
        # 换手率处理
        to = float(turnover_arr[i]) / 100.0
        if to > 1.0: to = 1.0
        if to < 0.0: to = 0.0
        
        # 初始帧检查：如果第一天换手率为0，强制赋予极小值以初始化筹码，否则后续全是NaN
        if i == 0 and to <= 0.0:
            to = 1.0 
        
        # 衰减旧筹码
        remain_ratio = 1.0 - to
        for b in range(bins_count):
            chips[b] *= remain_ratio
        total_weight_sum *= remain_ratio
        total_cost_product *= remain_ratio
        
        # 价格边界映射与索引计算 (加入 1e-9 防止精度偏离)
        p_low = max(base_price, min(max_price, float(low_arr[i])))
        p_high = max(p_low, min(max_price, float(high_arr[i])))
        # 业务改进：使用 vwap 作为分布中心比 close 更合理，若无 vwap 则退化回 close
        p_center = float(vwap_arr[i])
        if np.isnan(p_center) or p_center < p_low or p_center > p_high:
            p_center = float(close_arr[i])
        p_center = max(p_low, min(p_high, p_center))
        
        idx_low = int(np.log(p_low / base_price) / log_ratio + 1e-9)
        idx_high = int(np.log(p_high / base_price) / log_ratio + 1e-9)
        idx_center = int(np.log(p_center / base_price) / log_ratio + 1e-9)
        
        idx_low = max(0, min(bins_count - 1, idx_low))
        idx_high = max(0, min(bins_count - 1, idx_high))
        idx_center = max(idx_low, min(idx_high, idx_center))
        
        # 填充今日新筹码（三角形分布模拟）
        total_weight_sum += to
        if idx_high == idx_low:
            chips[idx_low] += to
            total_cost_product += bin_prices[idx_low] * to
        else:
            # 第一次遍历计算总权重
            w_sum = 0.0
            for b in range(idx_low, idx_high + 1):
                if b <= idx_center:
                    # 左侧斜率：当 b=idx_low, w=0; 当 b=idx_center, w=1
                    # 修正：即使 idx_center == idx_low，分母也至少为 1 避免除零
                    denom = float(idx_center - idx_low)
                    w = (b - idx_low) / denom if denom > 0 else 1.0
                else:
                    # 右侧斜率：当 b=idx_high, w=0; 当 b=idx_center, w=1
                    denom = float(idx_high - idx_center)
                    w = (idx_high - b) / denom if denom > 0 else 1.0
                w_sum += w
            
            # 第二次遍历分配筹码
            if w_sum > 0:
                inv_w_sum = to / w_sum
                for b in range(idx_low, idx_high + 1):
                    if b <= idx_center:
                        w = (b - idx_low) / float(idx_center - idx_low) if idx_center > idx_low else 1.0
                    else:
                        w = (idx_high - b) / float(idx_high - idx_center) if idx_high > idx_center else 1.0
                    bin_share = w * inv_w_sum
                    chips[b] += bin_share
                    total_cost_product += bin_prices[b] * bin_share

        # 计算平均成本
        if total_weight_sum > 1e-12:
            out_avg_cost[i] = total_cost_product / total_weight_sum
        else:
            out_avg_cost[i] = p_center
        
        # 计算累积分布
        current_sum = 0.0
        for b in range(bins_count):
            current_sum += chips[b]
            cum_chips[b] = current_sum
        total_chips_now = cum_chips[-1]
        
        if total_chips_now > 1e-12:
            # 获利盘比例（线性插值提高精度）
            idx_c = idx_center # 使用中心价计算获利盘
            p_target = p_center
            p_l_bin = bin_prices[idx_c]
            cum_l = cum_chips[idx_c]
            
            if idx_c < bins_count - 1:
                p_h_bin = bin_prices[idx_c + 1]
                cum_h = cum_chips[idx_c + 1]
                # 线性插值
                fraction = (p_target - p_l_bin) / (p_h_bin - p_l_bin)
                interpolated_cum = cum_l + fraction * (cum_h - cum_l)
            else:
                interpolated_cum = cum_l
            
            out_profit_ratio[i] = interpolated_cum / total_chips_now
            # 筹码单峰峰值高度：计算当前单格（0.1%步长）内筹码占比
            # np.max(chips) 由 Numba 在编译期自动优化为高性能的 C 级极值并行查找
            out_peak_density[i] = np.max(chips) / (total_chips_now + 1e-9)
            
            # 计算分位数 (90% 和 70% 成本区间)
            t_5 = 0.05 * total_chips_now
            t_15 = 0.15 * total_chips_now
            t_85 = 0.85 * total_chips_now
            t_95 = 0.95 * total_chips_now
            
            # searchsorted 在 Numba 中性能极佳
            idx_5 = np.searchsorted(cum_chips, t_5)
            idx_15 = np.searchsorted(cum_chips, t_15)
            idx_85 = np.searchsorted(cum_chips, t_85)
            idx_95 = np.searchsorted(cum_chips, t_95)
            
            # 边界保护
            idx_5 = min(bins_count - 1, idx_5)
            idx_15 = min(bins_count - 1, idx_15)
            idx_85 = min(bins_count - 1, idx_85)
            idx_95 = min(bins_count - 1, idx_95)
            
            p_5 = bin_prices[idx_5]
            p_15 = bin_prices[idx_15]
            p_85 = bin_prices[idx_85]
            p_95 = bin_prices[idx_95]

            out_cost_90_low[i], out_cost_90_high[i] = p_5, p_95
            out_cost_70_low[i], out_cost_70_high[i] = p_15, p_85

            # 集中度计算
            if (p_85 + p_15) > 0:
                out_conc_70[i] = (p_85 - p_15) / (p_85 + p_15)
            if (p_95 + p_5) > 0:
                out_conc_90[i] = (p_95 - p_5) / (p_95 + p_5)
        
        # 数值稳定性：防止长期累积导致的浮点数下溢
        # 每 250 天检查一次总权重，过低则归一化到 1.0
        if i % 250 == 0 and total_chips_now < 0.01 and total_chips_now > 0:
            norm_factor = 1.0 / total_chips_now
            for b in range(bins_count):
                chips[b] *= norm_factor
            total_weight_sum *= norm_factor
            total_cost_product *= norm_factor

    return (
        out_profit_ratio, out_avg_cost, 
        out_cost_90_low, out_cost_90_high, out_conc_90, 
        out_cost_70_low, out_cost_70_high, out_conc_70,
        out_peak_density
    )


# ==========================================
# 计算chip_penetration
# ==========================================
@njit
def calculate_chip_penetration_numba(close_arr, profit_ratio_arr, turnover_arr):
    n = len(close_arr)
    out_pos = np.full(n, np.nan) # 初始设为 nan，方便区分缺失值
    out_neg = np.full(n, np.nan)
    
    # 从 1 开始，第一个值设为 0
    out_pos[0] = 0.0
    out_neg[0] = 0.0
    
    for i in range(1, n):
        # 1. 安全性检查：处理 NaN
        if np.isnan(close_arr[i]) or np.isnan(close_arr[i-1]) or \
           np.isnan(profit_ratio_arr[i]) or np.isnan(turnover_arr[i]):
            out_pos[i] = 0.0
            out_neg[i] = 0.0
            continue

        # 2. 换手率处理 (假设输入 1.0 代表 1%)
        to = turnover_arr[i] / 100.0  
        if to < 0.0001: 
            to = 0.0001
            
        # 使用价格变化绝对值，避免除以0
        p_change = (close_arr[i] / close_arr[i-1]) - 1.0
        d_profit = profit_ratio_arr[i] - profit_ratio_arr[i-1]
        
        # 情况 A: 进攻穿透 (价涨且获利盘增)
        if p_change > 0.0001 and d_profit > 0:
            val = d_profit / to
            # 建议 Clip，防止数值爆炸，100.0 是一个合理的经验阈值
            out_pos[i] = min(val, 100.0)
            out_neg[i] = 0.0
        
        # 情况 B: 杀跌穿透 (价跌且获利盘减)
        elif p_change < -0.0001 and d_profit < 0:
            val = abs(d_profit) / to
            out_neg[i] = min(val, 100.0)
            out_pos[i] = 0.0
            
        else:
            # 包含背离情况和价格微动
            out_pos[i] = 0.0
            out_neg[i] = 0.0
            
    return out_pos, out_neg

# ==========================================
# 计算大盘环境感知因子
# ==========================================
def calculate_global_mkt_factors(file_path='zzqz_df.xlsx'):
    """
    计算全维度市场环境因子
    """
    try:
        df = pd.read_excel(file_path)
        # 统一列名处理
        df = df.rename(columns={'日期': 'date', '收盘': 'close', '成交额': 'amount', '成交量': 'volume'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        # 预计算收益率 (对数收益率在统计上更稳健)
        df['ret'] = np.log(df['close'] / df['close'].shift(1))

        # --- 1. 趋势因子 (Trend): 20日风险调整收益分位数 ---
        roll_mean_20 = df['ret'].rolling(20).mean()
        roll_std_20 = df['ret'].rolling(20).std()
        # 避免除以0，Sharpe反映趋势的稳定性
        sharpe = roll_mean_20 / (roll_std_20 + 1e-9)
        mkt_trend_rank = sharpe.rolling(252, min_periods=60).rank(pct=True)

        # --- 2. 波动因子 (Vol): 20日波动率在一年中的排位 ---
        # 修正：通常波动率越高，分位数越大，代表市场越不稳定
        mkt_vol_rank = roll_std_20.rolling(252, min_periods=60).rank(pct=True)

        # --- 3. 流动性因子 (Liq): 成交额相对于长期的扩张度 ---
        # 使用成交额 amount 替代 volume
        amt_ma20 = df['amount'].rolling(20).mean()
        # 反映当前成交热度处于历史什么位置
        mkt_liq_rank = amt_ma20.rolling(252, min_periods=60).rank(pct=True)

        # --- 4. 乖离因子 (Bias): 短期超买超卖 ---
        bias = df['close'] / df['close'].rolling(20).mean()
        mkt_bias_rank = bias.rolling(252, min_periods=60).rank(pct=True)

        # --- 5. 价格位置因子 (Position): 反映中长期支撑阻力位 ---
        mkt_position_rank = df['close'].rolling(250, min_periods=120).rank(pct=True)

        # --- 6. 大盘对数收益基准 (供个股相对强度特征作参照) ---
        # 个股相对强度 = 个股对数收益 - 大盘对数收益 (同期窗口)
        mkt_ret_20 = df['ret'].rolling(20).sum()
        mkt_ret_60 = df['ret'].rolling(60).sum()

        # 封装结果
        res = {
            'mkt_trend': mkt_trend_rank,      # 趋势得分 (0-1)
            'mkt_vol': mkt_vol_rank,          # 风险得分 (0-1, 越大越震荡)
            'mkt_liq': mkt_liq_rank,          # 流动性得分 (0-1, 越大越活跃)
            'mkt_bias': mkt_bias_rank,        # 超买得分 (0-1, 越大越超买)
            'mkt_position': mkt_position_rank, # 长期位置得分 (0-1, 越大离高点越近)
            'mkt_ret_20': mkt_ret_20,         # 大盘 20 日对数收益 (相对强度基准)
            'mkt_ret_60': mkt_ret_60          # 大盘 60 日对数收益 (相对强度基准)
        }
        
        final_df = pd.DataFrame(res)
        final_df = final_df.fillna(0.5)
        final_df = final_df.reset_index() 

        return final_df

    except Exception as e:
        logging.error(f"大盘因子计算失败: {e}")
        raise

# ==========================================
# 计算大盘广度，拥挤度
# ==========================================
def calculate_high_low_stats(stock_data, lookback_periods=[5, 10, 20, 60]):
    # 1. 预处理与时序对齐修复
    stock_data = stock_data.sort_values(['symbol', 'date'])
    
    # 使用 transform 确保个股时序索引完美对齐，消除未来函数
    stock_data['prev_close'] = stock_data.groupby('symbol')['close'].shift(1)
    stock_data['is_up'] = (stock_data['close'] > stock_data['prev_close']).astype(int)
    stock_data['is_down'] = (stock_data['close'] < stock_data['prev_close']).astype(int)
    stock_data['is_valid'] = stock_data['close'].notna().astype(int) # 用于分母

    for period in lookback_periods:
        rolling_group = stock_data.groupby('symbol')['close']
        h_val = rolling_group.transform(lambda x: x.rolling(period, min_periods=period).max())
        l_val = rolling_group.transform(lambda x: x.rolling(period, min_periods=period).min())
        
        stock_data[f'high{period}'] = (stock_data['close'] >= h_val).astype(int)
        stock_data[f'low{period}'] = (stock_data['close'] <= l_val).astype(int)

    # 2. 聚合日期数据
    agg_dict = {
        'is_up': 'sum',
        'is_down': 'sum',
        'is_valid': 'sum',
        **{f'high{period}': 'sum' for period in lookback_periods},
        **{f'low{period}': 'sum' for period in lookback_periods}
    }
    breadth_stats = stock_data.groupby('date').agg(agg_dict).reset_index()
    breadth_stats.rename(columns={'is_up': 'count_up', 'is_down': 'count_down'}, inplace=True)

    # 3. 特征工程一：占比比例化 (ML 模型特征对齐)
    total = breadth_stats['is_valid'] + 1e-9
    breadth_stats['up_ratio'] = breadth_stats['count_up'] / total
    for period in lookback_periods:
        breadth_stats[f'high{period}_ratio'] = breadth_stats[f'high{period}'] / total
        breadth_stats[f'low{period}_ratio'] = breadth_stats[f'low{period}'] / total

    # 4. 【补全：规则风控引擎特有的时序均线比例与加速度因子，杜绝场景判断失灵】
    # 时序 5日均线强度比例（high_ratio / low_ratio）
    ma5_high = breadth_stats['high20'].rolling(5).mean().replace(0, 0.1)
    ma5_low = breadth_stats['low20'].rolling(5).mean().replace(0, 0.1)
    breadth_stats['high_ratio'] = breadth_stats['high20'] / ma5_high
    breadth_stats['low_ratio'] = breadth_stats['low20'] / ma5_low

    # 绝对值时序速度与加速度 (high_v, low_v, high_a, low_a)
    breadth_stats['high_v'] = breadth_stats['high20'].diff()
    breadth_stats['low_v'] = breadth_stats['low20'].diff()
    breadth_stats['high_a'] = breadth_stats['high_v'].diff()
    breadth_stats['low_a'] = breadth_stats['low_v'].diff()

    # 时序加速度平滑因子 (low_a_smooth, high_a_smooth) -> is_market_ok 核心输入
    breadth_stats['low_a_smooth'] = breadth_stats['low_a'].rolling(window=3).mean()
    breadth_stats['high_a_smooth'] = breadth_stats['high_a'].rolling(window=3).mean()

    # ML 占比变动加速度
    breadth_stats['high20_v'] = breadth_stats['high20_ratio'].rolling(3).mean().diff()
    breadth_stats['high20_a'] = breadth_stats['high20_v'].diff()
    breadth_stats['high20_a_smooth'] = breadth_stats['high20_a'].rolling(3).mean()
    for col in ['low20', 'low60', 'high20', 'high60']:
        if col in breadth_stats.columns:
            q_vals = {0.2: 'q20', 0.3: 'q30', 0.4: 'q40', 0.5: 'q50', 0.6: 'q60',
                      0.7: 'q70', 0.75: 'q75', 0.9: 'q90', 0.95: 'q95'}
            qcols = {f'{col}_{tag}': breadth_stats[col].rolling(120, min_periods=30).quantile(p)
                     for p, tag in q_vals.items()}
            breadth_stats = pd.concat([breadth_stats, pd.DataFrame(qcols, index=breadth_stats.index)], axis=1)
    
    # 2. 广度均线 (用于判断动能斜率)
    breadth_stats['high10_ma5'] = breadth_stats['high10'].rolling(5, min_periods=2).mean()
    breadth_stats['high20_ma5'] = breadth_stats['high20'].rolling(5, min_periods=1).mean()
    breadth_stats['low20_ma5'] = breadth_stats['low20'].rolling(5, min_periods=1).mean()

    # 3. 补全你代码中用到的其他特定分位数 (按需添加)
    extra_qcols = {
        'low_ratio_q80': breadth_stats['low_ratio'].rolling(120, min_periods=30).quantile(0.8),
        'low_v_q80': breadth_stats['low_v'].rolling(120, min_periods=30).quantile(0.8),
        'high_ratio_q85': breadth_stats['high_ratio'].rolling(120, min_periods=30).quantile(0.85),
        'high_ratio_q90': breadth_stats['high_ratio'].rolling(120, min_periods=30).quantile(0.9),
    }
    breadth_stats = pd.concat([breadth_stats, pd.DataFrame(extra_qcols, index=breadth_stats.index)], axis=1)

    # 4. ratio 滚动分位数 (替代 is_market_ok 中的固定百分比阈值, 随大盘整体情况动态)
    for col in ['low20_ratio', 'low10_ratio', 'high20_ratio', 'high10_ratio']:
        if col in breadth_stats.columns:
            q_vals = {0.1: 'q10', 0.2: 'q20', 0.3: 'q30', 0.4: 'q40', 0.5: 'q50',
                      0.6: 'q60', 0.7: 'q70', 0.75: 'q75', 0.8: 'q80', 0.85: 'q85',
                      0.9: 'q90', 0.95: 'q95'}
            qcols = {f'{col}_{tag}': breadth_stats[col].rolling(120, min_periods=30).quantile(p)
                     for p, tag in q_vals.items()}
            breadth_stats = pd.concat([breadth_stats, pd.DataFrame(qcols, index=breadth_stats.index)], axis=1)

    # 5. 极速向量化计算大盘拥挤度 (V6 物理性能版)
    def get_top_pct_ratio(group, pct=0.05):
        vals = group.values
        vals = vals[~np.isnan(vals) & (vals > 0)]
        if len(vals) == 0: return 0.0
        vals.sort()
        vals = vals[::-1]
        n_top = max(1, int(len(vals) * pct))
        return np.sum(vals[:n_top]) / np.sum(vals)

    congestion = stock_data.groupby('date')['amount'].apply(get_top_pct_ratio)
    breadth_stats['congestion'] = breadth_stats['date'].map(congestion)

    # 先利用时序连续性填充空洞，再用经验中值兜底最开始的行
    breadth_stats['congestion'] = breadth_stats['congestion'].ffill().fillna(0.35)
    
    # --- 6. 拥挤度衍生指标 ---
    # 使用 min_periods=1 确保冷启动期也有平滑均线，不产生新的 NaN
    breadth_stats['congestion_ma20'] = (
        breadth_stats['congestion']
        .rolling(window=20, min_periods=1)
        .mean()
    )
    # 计算 Bias：当前状态相对于动态中轴的偏离
    breadth_stats['congestion_bias'] = (
        breadth_stats['congestion'] / (breadth_stats['congestion_ma20'] + 1e-9)
    )
    # 最终防御：Bias 在最极端情况下（如全场零成交）应保持中性
    breadth_stats['congestion_bias'] = breadth_stats['congestion_bias'].ffill().fillna(1.0)

    # 1. 趋势与中性类：填 0.5
    neutral_05_cols = ['up_ratio']
    breadth_stats[neutral_05_cols] = breadth_stats[neutral_05_cols].fillna(0.5)
    
    # 2. 状态对齐类：填 1.0
    neutral_10_cols = ['high_ratio', 'low_ratio', 'congestion_bias']
    breadth_stats[neutral_10_cols] = breadth_stats[neutral_10_cols].fillna(1.0)
    
    # 4. 变化量与极端占比类：填 0
    # 填充空值：冷启动期使用 ffill + 常规中性值
    breadth_stats = breadth_stats.ffill().fillna(0)

    return breadth_stats

# ==========================================
# 计算大盘广度、环境，合并，保存文件
# ==========================================
def sync_market_context_file(cache_dir, output_path='market_context_cache.parquet',
                             pc_table_path=None):
    """
    扫描所有数据库，生成全市场环境因子（广度、拥挤度等）并持久化。
    使用 Parquet 格式，读取速度比 CSV 快 10 倍以上。

    统一经 LocalDataCache 读取（qfq 前复权，与 backtest 推理时的
    GLOBAL_MARKET_STATS 完全对齐），并对齐中证全指 (000985) 股票池口径。

    参数:
    - pc_table_path: 外部预计算 PCA 表路径。提供时，直接合并该表中的
      FeatureConfig.MKT_FEATURES 列到输出，避免实验脚本手动覆盖项目根目录 parquet。
    """
    from local_data_cache import LocalDataCache
    from screen import basic_screen
    logging.info("开始同步大盘环境快照...")

    all_symbols = basic_screen(cache_dir=cache_dir)
    logging.info(f"中证全指股票池: {len(all_symbols)} 只")
    ldc = LocalDataCache(cache_dir=cache_dir)
    basic_data_list = []

    # 1. 极简读取：只取计算广度必需的列 (qfq 对齐 backtest)
    for i, s in enumerate(all_symbols):
        try:
            df = ldc.get_stock_data(s, '1990-01-01', '2100-01-01', adjust='qfq', mode=2)
            if df is None or df.empty:
                continue
            df['symbol'] = s
            basic_data_list.append(df[['date', 'symbol', 'close', 'high', 'low', 'amount']])
        except Exception:
            continue
        if i % 1000 == 0:
            logging.info(f"已读取 {i} / {len(all_symbols)}")

    full_market = pd.concat(basic_data_list)
    full_market['date'] = pd.to_datetime(full_market['date'])
    
    # 2. 调用原有的广度计算函数
    mkt_breadth = calculate_high_low_stats(full_market)
    
    # 3. 补充：合并大盘 5 维环境因子 (Trend, Vol, Liq...)
    # 这样这张表就包含了“关于大盘的一切”
    mkt_factors = calculate_global_mkt_factors('zzqz_df.xlsx')
    final_context = mkt_breadth.merge(mkt_factors, on='date', how='left')

    # 4. 生成/合并 PCA 正交大盘特征并持久化（统一走公共组件）
    # 若提供外部 PC 表，先删除旧 MKT_FEATURES 列再合并，防止口径污染
    if pc_table_path is not None:
        for col in list(FeatureConfig.MKT_FEATURES):
            if col in final_context.columns:
                final_context = final_context.drop(columns=[col])
    final_context = final_context.merge(
        build_market_pca_table(final_context, min_periods=60, pc_table_path=pc_table_path),
        on='date', how='left'
    )

    # 5. 持久化
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    final_context.to_parquet(output_path)
    logging.info(f"大盘环境快照已同步至: {output_path} (共 {len(final_context)} 个交易日)")
    return final_context


# ==========================================
# 公共计算个股特征
# ==========================================
def compute_individual_indicators(df, mkt_factors, use_smooth=True):
    """
    计算单只股票的时序原始特征（未进行截面标准化）
    
    参数:
    - df: 包含基础行情 [open, high, low, close, volume, turnover, vwap] 的 DataFrame
    - mkt_factors: 大盘环境因子 DataFrame (需包含 mkt_vol 列)
    - use_smooth: bool, True 为训练对齐版(全平滑), False 为时效实验版(脉冲不平滑)
    """
    if df.empty:
        return df

    # --- 1. 基础预处理 ---
    # 数值防御：防止极低价格导致的除零和溢出
    price_cols = ['open', 'high', 'low', 'close', 'vwap']
    df[price_cols] = df[price_cols].clip(lower=0.01)
    df['turnover'] = df['turnover'].fillna(0.0).clip(lower=0.0001)
    
    # 获取底层数组 (用于 Numba 加速算子)
    c = df['close'].values.astype(np.float64)
    v = df['vwap'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    t = df['turnover'].values.astype(np.float64)

    # --- 2. 自适应窗口计算 ---
    # 根据大盘波动率动态调整 EMA 的平滑周期
    # mkt_factors 应提前通过日期 map 到 df 中
    mkt_vol = df['date'].map(mkt_factors['mkt_vol']).ffill().fillna(0.5).values
    multiplier = (1.5 - mkt_vol).clip(0.6, 1.4)
    w_slow = 30.0 * multiplier
    w_mid = 20.0 * multiplier
    w_fast = 15.0 * multiplier

    # --- 3. 调用筹码核心算子 ---
    # chips 包含: [获利盘, 平均成本, 90低, 90高, 90集中度, 70低, 70高, 70集中度, 峰值密度]
    chips = calculate_chip_metrics_numba(c, v, h, l, t)
    p_pos, p_neg = calculate_chip_penetration_numba(c, chips[0], t)

    # --- 4. 筹码结构特征 (核心骨架：必须平滑) ---
    # 筹码获利盘与单峰形态高度
    df['ema_profit'] = dynamic_ema(chips[0], w_slow)
    df['ema_profit'] = df['ema_profit'].fillna(0.5)
    df['res_profit'] = chips[0] - df['ema_profit']
    df['ema_peak_density'] = dynamic_ema(chips[8], w_slow)
    
    # 筹码集中度残差与 70% 集中度
    df['ema_conc_90'] = dynamic_ema(chips[4], w_mid)
    df['res_conc_90'] = chips[4] - df['ema_conc_90']
    df['ema_conc_70'] = dynamic_ema(chips[7], w_mid)

    # --- 5. 脉冲敏感型特征 (开关：实验对比核心) ---
    
    # (A) 筹码 90% 集中度收缩速度 (Velocity)
    conc_90_v = pd.Series(chips[4]).pct_change(5).fillna(0).values.astype(np.float64)
    conc_90_v = np.clip(conc_90_v, -0.20, 0.20)
    if use_smooth:
        df['ema_conc_90_v'] = dynamic_ema(conc_90_v, w_mid)
    else:
        df['ema_conc_90_v'] = conc_90_v

    # (B) 间歇式放量最大残差 (Volume Surge)
    t_mean_60 = pd.Series(t).rolling(60, min_periods=1).mean().values + 1e-9
    t_ratio_60 = (t / t_mean_60).astype(np.float64)
    t_max_res = pd.Series(t_ratio_60).rolling(20).max().fillna(1.0).values
    if use_smooth:
        df['ema_turnover_max_res'] = dynamic_ema(t_max_res, w_slow)
    else:
        df['ema_turnover_max_res'] = t_max_res

    # (C) 20日时序量价相关性 (VP Divergence)
    c_pct = pd.Series(c).pct_change().fillna(0)
    v_pct = pd.Series(df['volume'].values).pct_change().fillna(0)
    vp_corr = c_pct.rolling(20).corr(v_pct).fillna(0.0).values.astype(np.float64)
    if use_smooth:
        df['ema_vp_corr'] = dynamic_ema(vp_corr, w_slow)
    else:
        df['ema_vp_corr'] = vp_corr
    # 建议3b: 量价相关性残差 (当日 vp_corr 相对平滑基线的偏离)
    # 残差为正 = 近期量价协同增强(量价齐升/齐跌), 残差为负 = 量价背离(价动量缩)
    df['res_vp_corr'] = vp_corr - df['ema_vp_corr']

    # --- 6. 价格位置与支撑特征 (Raw Ratios) ---
    df['dist_to_avg'] = (c / (chips[1] + 1e-9)) - 1.0
    df['dist_to_high90'] = (c / (chips[3] + 1e-9)) - 1.0

    # --- 7. 穿透力特征 (EMA 平滑) ---
    df['ema_penetrate_up'] = dynamic_ema(p_pos, w_slow)
    df['ema_decay_dn'] = dynamic_ema(p_neg, w_slow)

    # --- 8. 乖离率标准化特征 (V5 核心逻辑) ---
    df['bias_20'] = (c / (talib.SMA(c, 20) + 1e-9)) - 1.0
    # 用 60 日滚动标准差作为分母进行归一化，解决不同波动个股的数值对齐
    bias_std = df['bias_20'].rolling(60).std().fillna(0.02)
    bias_norm = (df['bias_20'] / (bias_std + 1e-9)).astype(np.float64)
    df['ema_bias_norm'] = dynamic_ema(bias_norm.values, w_fast)
    df['ema_bias_norm'] = df['ema_bias_norm'].fillna(0.0)
    df['res_bias_norm'] = bias_norm - df['ema_bias_norm']

    # --- 9. 筹码重心位移速度 (EMA 平滑) ---
    avg_cost_velocity = pd.Series(chips[1]).pct_change(5).fillna(0).values.astype(np.float64)
    avg_cost_velocity = np.clip(avg_cost_velocity, -0.15, 0.15)
    df['ema_cost_v'] = dynamic_ema(avg_cost_velocity, w_mid)

    # --- 10. 量能平稳性特征 ---
    t_mean_20 = pd.Series(t).rolling(20, min_periods=1).mean().values + 1e-9
    turnover_rel = (t / t_mean_20).astype(np.float64)
    df['ema_turnover_vol'] = dynamic_ema(turnover_rel, w_slow)

    # --- 10.5 量能多维拆解组合特征 (历史探索 turn_vol_*) ---
    # 1) turn_vol_mom: 换手率动量 (5日/20日均量 - 1)
    t_mean_5 = pd.Series(t).rolling(5, min_periods=1).mean().values + 1e-9
    df['turn_vol_mom'] = (t_mean_5 / t_mean_20 - 1.0).astype(np.float64)
    # 2) turn_vol_stab: 换手率稳定性 (高=量能稳定)
    t_std_20 = pd.Series(t).rolling(20, min_periods=1).std().values
    t_std_20 = np.nan_to_num(t_std_20, nan=0.0)
    df['turn_vol_stab'] = -(t_std_20 / t_mean_20).astype(np.float64)
    # 3) turn_price_sync: 量价同步 (同向=+1, 反向=-1, 任一不动=0)
    t_ret = pd.Series(t).pct_change().fillna(0).values
    c_ret = pd.Series(c).pct_change().fillna(0).values
    df['turn_price_sync'] = (np.sign(t_ret) * np.sign(c_ret)).astype(np.float64)

    # 建议2: 吸筹确认交互项 (放量 × 筹码变集中)
    # 放量(turnover_vol>1) 且筹码变集中(conc_90 当日低于平滑值) → 确认主力吸筹 → 正值
    # 孤立放量但筹码不集中 → 负值或接近0 → 不奖励
    # 用相对变化率让量级与换手率可乘 (集中度绝对变化仅 ~0.002 级, 直接乘趋近于0)
    # 连续形式: (turnover_vol - 1) × (-conc_90 相对变化), 不放量时自然为负/0
    activation = (df['ema_turnover_vol'] - 1.0)
    conc_tightening = (-df['res_conc_90'] / (df['ema_conc_90'] + 1e-9))
    df['acc_confirm'] = activation * conc_tightening

    # 建议3: 高位缩量背离 (价格乖离高位 × 量能萎缩 → 上涨动能枯竭警示)
    # ema_bias_norm 高位(+z) 同时 ema_turnover_vol 缩量(<1) → 背离 → 负值
    # 价格高位 + 放量 → 维持强势 → 正值
    vol_deficit = (df['ema_turnover_vol'] - 1.0).clip(upper=0.0)
    df['vp_diverg'] = -vol_deficit * df['ema_bias_norm'].clip(lower=0.0)

    vol_stab = pd.Series(t).rolling(20).std() / (pd.Series(t).rolling(20).mean() + 1e-9)
    df['ema_vol_stab'] = dynamic_ema(vol_stab.fillna(0).values, w_mid)
    df['res_vol_stab'] = vol_stab.values - df['ema_vol_stab']

    # --- 补全 A：个股相对拥挤度 (Individual Crowding) ---
    # 个股成交额 / 个股过去20日平均成交额
    df['stock_congestion'] = df['amount'] / (df['amount'].rolling(20).mean() + 1e-9)
    # 配合 Z-Score 标准化，能识别出谁在“异常放量”

    # --- 补全 B：高位震荡熵 (Volatility Surge at Highs) ---
    # 计算价格在 20 日高点附近的波动剧烈程度
    high_20 = df['close'].rolling(20).max()
    df['high_vol_interaction'] = ((df['close'] / (high_20 + 1e-9)) - 1.0) * df['res_vol_stab'] # 靠近高点且波动放大

    # --- 补全 C：主力离场迹象 (V6 逻辑) ---
    # 价升量减 (Divergence) 的极端情况
    # 使用你已有的 ema_vp_corr (量价相关性)
    # 如果 ema_vp_corr 从 0.8 跌回 0.2，说明拉升动能枯竭
    df['vp_corr_decay'] = df['ema_vp_corr'].diff(3)

    # --- 附加：回测业务需要的原始指标（不参与模型 Z-Score，仅用于逻辑判断）
    df['profit_ratio'] = chips[0]
    df['avg_cost'] = chips[1]
    df['concentration_70'] = chips[7]
    df['chip_penetration'] = p_pos

    # --- 补全 D：个股相对市场强度 (Relative Strength vs Market) ---
    # 业务含义：模型只做绝对贴成本/缩量判断，缺"该股相对大盘/板块的强弱"维度。
    # 相对强度 = 个股对数收益 - 大盘对数收益 (同期窗口)，反映个股超额动量。
    # 正 = 个股强于大盘 (领涨/抗跌)，负 = 弱于大盘 (滞涨/领跌)。
    # 注: 原始 close 非复权, 除权日会有单日假摔, 由截面 Z-score clip(-3,3) 兜底。
    c_log = pd.Series(np.log(c + 1e-9))
    stock_ret_20 = c_log.diff(20).fillna(0.0)
    stock_ret_60 = c_log.diff(60).fillna(0.0)
    if 'mkt_ret_20' in mkt_factors.columns:
        mkt_ret_20 = df['date'].map(mkt_factors['mkt_ret_20']).ffill().fillna(0.0).values
        mkt_ret_60 = df['date'].map(mkt_factors['mkt_ret_60']).ffill().fillna(0.0).values
    else:
        mkt_ret_20 = 0.0
        mkt_ret_60 = 0.0
    df['rs_20'] = (stock_ret_20 - mkt_ret_20).astype(np.float64)
    df['rs_60'] = (stock_ret_60 - mkt_ret_60).astype(np.float64)
    # 保存个股原始对数收益 (供行业中性化残差 rs_ind_20/60 使用, 不直接进模型)
    df['stock_ret_20'] = stock_ret_20.astype(np.float64)
    df['stock_ret_60'] = stock_ret_60.astype(np.float64)

    # --- 11. 全局数值防御 (返回前最后一步) ---
    # 将所有的 inf 替换为 0，防止标准化时崩溃
    df = df.replace([np.inf, -np.inf], np.nan)

    # 其余特征大多以 0 为基准
    df = df.fillna(0.0)

    return df


# ==========================================
# 训练目标计算（公共入口，避免各训练脚本口径漂移）
# ==========================================
def compute_entry_target(df, window=20, eps=0.0001):
    """
    计算入场模型训练目标：Gain-to-Pain Ratio (GPR)。

    返回在原始 df 上新增两列：
    - target_val: 未来 window 日真实收益率（用于审计 PnL）
    - gpr_target: 未来 window 日 GPR（用于训练目标）
    """
    if df.empty or 'close' not in df.columns:
        return df
    daily_ret = df['close'].pct_change(1)
    df['target_val'] = df['close'].pct_change(window).shift(-window)
    pos_rets = daily_ret.clip(lower=0)
    neg_rets = daily_ret.clip(upper=0).abs()
    f_pos_sum = pos_rets.rolling(window).sum().shift(-window)
    f_neg_sum = neg_rets.rolling(window).sum().shift(-window)
    df['gpr_target'] = f_pos_sum / (f_neg_sum + eps)
    return df


def compute_risk_target(df, hold_window=5):
    """
    计算风控模型训练目标：未来 hold_window 日最大日内跌幅（仅负值，已标准化）。

    返回在原始 df 上新增一列 risk_score。
    """
    if df.empty or 'close' not in df.columns:
        return df
    daily_ret = df['close'].pct_change(1)
    sigma = daily_ret.rolling(20).std()
    f_max_loss = daily_ret.shift(-1).rolling(hold_window).min()
    raw_risk = f_max_loss / (sigma + 0.005)
    df['risk_score'] = raw_risk.clip(-5.0, 0.0)
    return df


# ==========================================
# 标准化并整合输出
# ==========================================
def compute_industry_inner_rank(final_df, industry_map):
    """
    计算行业内个股排名特征 ind_inner_rank (0~1 分位)。

    对每日每行业分组, 对 rs_20 做 rank(pct=True) —— 个股在其所属申万一级行业内的相对强弱。
    - 缺失行业映射的股票填 0.5 (中性, 不参与行业内排名)
    - 组内仅 1 只时 rank 为 NaN, 也填 0.5
    修改 final_df 原址, 增加 ind_inner_rank 列。
    """
    if final_df is None or final_df.empty:
        return final_df
    if not industry_map:
        final_df['ind_inner_rank'] = 0.5
        return final_df
    ind_code = final_df['symbol'].astype(str).str.zfill(6).map(industry_map).fillna('')
    final_df['_ind_code'] = ind_code
    final_df['ind_inner_rank'] = (
        final_df.groupby(['date', '_ind_code'])['rs_20'].rank(pct=True)
    )
    # 组内样本过少 (<5) 时排名无统计意义, 填 0.5 中性, 避免孤立行业股恒得满分
    group_sizes = final_df.groupby(['date', '_ind_code'])['rs_20'].transform('size')
    final_df.loc[group_sizes < 5, 'ind_inner_rank'] = 0.5
    final_df['ind_inner_rank'] = final_df['ind_inner_rank'].fillna(0.5).astype(np.float64)
    final_df.drop(columns=['_ind_code'], inplace=True)
    return final_df


def calculate_industry_rank_table(industry_daily, mkt_factors):
    """
    计算每日每个申万一级行业的 ind_rank (0~1 分位)。

    行业RS = 行业指数对数收益 - 大盘对数收益 (与个股 rs_20/60 同一基准 mkt_ret_20/60)。
    每日对 31 个行业按行业RS截面 rank(pct=True), 得 ind_rank_20/60。
    返回 long DataFrame: 日期, 行业代码, ind_rank_20, ind_rank_60
    """
    if industry_daily is None or industry_daily.empty:
        return pd.DataFrame(columns=['日期', '行业代码', 'ind_rank_20', 'ind_rank_60'])

    df = industry_daily[['代码', '日期', '收盘']].copy()
    df['代码'] = df['代码'].astype(str)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['代码', '日期'])

    df['ret'] = np.log(df['收盘'] / df['收盘'].shift(1))
    df['ind_ret_20'] = df.groupby('代码')['ret'].transform(lambda x: x.rolling(20).sum())
    df['ind_ret_60'] = df.groupby('代码')['ret'].transform(lambda x: x.rolling(60).sum())

    # 大盘对数收益基准 (mkt_factors 可含 date 列, 也可为 date 索引)
    if 'date' in mkt_factors.columns:
        mkt = mkt_factors[['date', 'mkt_ret_20', 'mkt_ret_60']].copy()
    else:
        mkt = mkt_factors.reset_index()
        if 'index' in mkt.columns:
            mkt = mkt.rename(columns={'index': 'date'})
    date_col = 'date' if 'date' in mkt.columns else mkt.columns[0]
    mkt_20 = dict(zip(pd.to_datetime(mkt[date_col]), mkt['mkt_ret_20']))
    mkt_60 = dict(zip(pd.to_datetime(mkt[date_col]), mkt['mkt_ret_60']))

    df['mkt_ret_20'] = df['日期'].map(mkt_20).ffill()
    df['mkt_ret_60'] = df['日期'].map(mkt_60).ffill()

    df['ind_rs_20'] = df['ind_ret_20'] - df['mkt_ret_20']
    df['ind_rs_60'] = df['ind_ret_60'] - df['mkt_ret_60']

    # 每日 31 行业截面排名 (分位 0~1)
    df['ind_rank_20'] = df.groupby('日期')['ind_rs_20'].rank(pct=True)
    df['ind_rank_60'] = df.groupby('日期')['ind_rs_60'].rank(pct=True)

    out = df[['日期', '代码', 'ind_rank_20', 'ind_rank_60']].copy()
    out['ind_rank_20'] = out['ind_rank_20'].fillna(0.5)
    out['ind_rank_60'] = out['ind_rank_60'].fillna(0.5)
    return out


def map_industry_rank(final_df, industry_map, ind_rank_table):
    """
    将 ind_rank_20/60 (行业级共享常数) 映射到个股行。
    final_df 需含 'symbol' 和 'date' 列; 无行业归属或行业表缺失时填 0.5 中性。
    修改 final_df 原址。
    """
    if ind_rank_table is None or ind_rank_table.empty:
        final_df['ind_rank_20'] = 0.5
        final_df['ind_rank_60'] = 0.5
        return final_df
    if not industry_map:
        final_df['ind_rank_20'] = 0.5
        final_df['ind_rank_60'] = 0.5
        return final_df

    rank_index = ind_rank_table.set_index(['日期', '代码'])
    final_df['_ind_code'] = final_df['symbol'].astype(str).str.zfill(6).map(industry_map).fillna('')
    keys = list(zip(final_df['date'], final_df['_ind_code']))
    for col in ['ind_rank_20', 'ind_rank_60']:
        final_df[col] = rank_index.reindex(keys)[col].to_numpy()
        final_df[col] = final_df[col].fillna(0.5).astype(np.float64)
    final_df.drop(columns=['_ind_code'], inplace=True)
    return final_df


def calculate_industry_ret_table(industry_daily):
    """
    计算每个行业每日的行业对数收益 (滚动 20/60 日求和)。
    返回 long DataFrame: 日期, 行业代码, ind_ret_20, ind_ret_60
    用于行业中性化残差 rs_ind_20/60 = 个股对数收益 - 行业对数收益。
    """
    if industry_daily is None or industry_daily.empty:
        return pd.DataFrame(columns=['日期', '行业代码', 'ind_ret_20', 'ind_ret_60'])
    df = industry_daily[['代码', '日期', '收盘']].copy()
    df['代码'] = df['代码'].astype(str)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['代码', '日期'])
    df['ret'] = np.log(df['收盘'] / df['收盘'].shift(1))
    df['ind_ret_20'] = df.groupby('代码')['ret'].transform(lambda x: x.rolling(20).sum())
    df['ind_ret_60'] = df.groupby('代码')['ret'].transform(lambda x: x.rolling(60).sum())
    out = df[['日期', '代码', 'ind_ret_20', 'ind_ret_60']].copy()
    out = out.rename(columns={'代码': '行业代码'})
    return out


def map_industry_rs(final_df, industry_map, ind_ret_table):
    """
    映射行业中性化相对强度 rs_ind_20/60 = 个股对数收益 - 所属行业对数收益。
    剔除行业共同因子, 修正 rs_20/60 混入行业动量的问题 (rs 实验失败根因)。
    final_df 需含 stock_ret_20/60 (compute_individual_indicators 已保存) 与 symbol/date。
    无行业归属或行业表缺失时填 0 (中性, 等价于不加行业调整)。
    修改 final_df 原址。
    """
    if ind_ret_table is None or ind_ret_table.empty or not industry_map:
        final_df['rs_ind_20'] = 0.0
        final_df['rs_ind_60'] = 0.0
        return final_df

    ret_index = ind_ret_table.set_index(['日期', '行业代码'])
    final_df['_ind_code'] = final_df['symbol'].astype(str).str.zfill(6).map(industry_map).fillna('')
    keys = list(zip(final_df['date'], final_df['_ind_code']))
    for col, src in [('rs_ind_20', 'ind_ret_20'), ('rs_ind_60', 'ind_ret_60')]:
        ind_ret = ret_index.reindex(keys)[src].to_numpy()
        ind_ret = np.nan_to_num(ind_ret, nan=0.0)
        final_df[col] = (final_df['stock_ret_' + col[-2:]] - ind_ret).astype(np.float64)
    final_df.drop(columns=['_ind_code'], inplace=True)
    return final_df


def apply_standardization(final_df, industry_map=None, ind_rank_table=None, ind_ret_table=None, features=None):
    """
    执行横截面 Z-Score 标准化及复合特征计算

    参数:
    - final_df: 包含所有个股原始特征的合并 DataFrame，必须包含 'date' 列
    - industry_map: {symbol6位: 行业代码} 映射; 提供时计算行业内排名特征 ind_inner_rank
    - ind_rank_table: 行业排名表 (calculate_industry_rank_table 输出); 提供时映射 ind_rank_20/60
    - ind_ret_table: 行业收益表 (calculate_industry_ret_table 输出); 提供时映射 rs_ind_20/60
    - features: 需要标准化的原始特征名列表。None 时使用 BIZ_FEATURES 与 BIZ_RISK_FEATURES 的并集。
    """
    if final_df.empty:
        return final_df

    # 0. 行业内排名 (独立分位特征, 不走 Z-Score)
    compute_industry_inner_rank(final_df, industry_map)

    # 0.5 行业排名 (行业级共享常数, 不走 Z-Score)
    map_industry_rank(final_df, industry_map, ind_rank_table)

    # 0.6 行业中性化相对强度 (个股对数收益 - 行业对数收益)
    map_industry_rs(final_df, industry_map, ind_ret_table)

    # 1. 获取配置好的特征列表
    if features is None:
        biz_features = list(dict.fromkeys(FeatureConfig.BIZ_FEATURES + FeatureConfig.BIZ_RISK_FEATURES))
    else:
        biz_features = features

    # 2. 预处理：数值防御
    # 替换无穷值，并对原始特征进行基础填充，防止 transform 失败
    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    
    # 3. 核心计算：横截面 Z-Score
    # 使用 groupby('date') 确保每一天是一个独立的参考系
    # transform('mean') 和 transform('std') 是最高效的向量化写法
    logging.info(f"开始对 {len(biz_features)} 个特征进行横截面标准化...")
    
    # 为了内存效率，我们采用分列处理而非整体 transform
    for col in biz_features:
        if col not in final_df.columns:
            logging.warning(f"特征列 {col} 不在 DataFrame 中，跳过归一化。")
            continue
            
        # 计算每日均值和标准差
        group = final_df.groupby('date')[col]
        m = group.transform('mean')
        s = group.transform('std')
        
        # 执行 Z-Score: (x - mean) / std
        # 1e-9 防止除以 0；clip(-3, 3) 消除极值影响（去极值化）
        z_col_name = f"{col}_z"
        final_df[z_col_name] = ((final_df[col] - m) / (s + 1e-9)).clip(-3, 3)
        
        # 填充缺失值：如果全天该列都一样（std=0），则填 0（中性）
        final_df[z_col_name] = final_df[z_col_name].fillna(0.0)

    # 4. 复合特征计算 (Divergence 逻辑)
    # 【业务含义】：筹码获利盘与价格乖离的背离程度
    # 如果获利盘极高（z极大）但乖离率并不高（z较小），则 div 很大，代表筹码高度锁定且未透支价格
    if 'ema_profit_z' in final_df.columns and 'ema_bias_norm_z' in final_df.columns:
        final_df['profit_bias_div_z'] = (final_df['ema_profit_z'] - final_df['ema_bias_norm_z']).clip(-3, 3)
    else:
        final_df['profit_bias_div_z'] = 0.0

    return final_df


