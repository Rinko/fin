import os
os.makedirs(os.path.join('external_data','logs'), exist_ok=True)
import pybroker
from pybroker import Strategy, StrategyConfig
from pybroker.data import DataSource
from pybroker.common import FeeMode, PositionMode
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, linregress
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, mean_squared_error
from numba import njit,jit
from math import sqrt, pi, exp
import talib
import logging
import joblib
import warnings
# 导入公共算子
try:
    import co_compute 
except ImportError:
    logging.error("无法加载 co_compute.py，请检查路径")

# 过滤 LightGBM 噪声参数警告日志
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
from stock_fetcher_bao import BaostockCodeFetcher
from local_data_cache import LocalDataCache
import is_market_ok
import signal_engine

# =========================================================================
# 运行时热修复：修复 PyBroker 官方源码中 info_loaded_bar_data 拼写逗号导致的错误
# =========================================================================
import pybroker.log

def patched_info_loaded_bar_data(self, symbols, start_date, end_date, timeframe):
    self._info(
        "Loaded:\n"
        f"namespace={self._scope.data_source_cache_ns}\n"
        f"{start_date} to {end_date}\n"
        f"timeframe: {timeframe}\n"
        f"{sorted(symbols)}"
    )

pybroker.log.Logger.info_loaded_bar_data = patched_info_loaded_bar_data

# =========================================================================
# 基础配置与多级缓存开启
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join('external_data','logs','log.log'), encoding='utf-8', mode='w'),
    ]
)
# 开启数据源物理缓存
# pybroker.enable_data_source_cache(
#     namespace='chip_strategy_10y'
# )
pd.set_option('display.max_columns', None)

# =========================================================================
# 预先在全局加载预训练模型包
# =========================================================================
# 哨兵：模型一律经 reload_models()/load_magnitude_models() 显式初始化（由统一入口保证时序）。
# 禁止依赖 import 期隐式加载；未初始化即使用会在此处立即暴露。
GLOBAL_MODEL_PKG = GLOBAL_RISK_MODEL_PKG = None
trained_lgbm = model_features = trained_risk_lgbm = model_risk_features = None

def reload_models(entry_pkl='chip_accumulation_v6.pkl', risk_pkl='chip_risk_model_v1.pkl'):
    """热重载模型包，供滚动重训脚本在每次折叠后切换使用新训练权重。"""
    global GLOBAL_MODEL_PKG, GLOBAL_RISK_MODEL_PKG
    global trained_lgbm, model_features, trained_risk_lgbm, model_risk_features
    GLOBAL_MODEL_PKG = joblib.load(entry_pkl)
    GLOBAL_RISK_MODEL_PKG = joblib.load(risk_pkl)
    trained_lgbm = GLOBAL_MODEL_PKG['model']
    model_features = GLOBAL_MODEL_PKG['features']
    trained_risk_lgbm = GLOBAL_RISK_MODEL_PKG['model']
    model_risk_features = GLOBAL_RISK_MODEL_PKG['features']
    print(f"模型已重载: {entry_pkl} + {risk_pkl}")

# =========================================================================
# 可选：幅度模型（探索用，不加载时不影响生产）
# =========================================================================
GLOBAL_OPPORT_MAG_PKG = None
GLOBAL_RISK_MAG_PKG = None
trained_opport_mag_lgbm = None
opport_mag_features = None
trained_risk_mag_lgbm = None
risk_mag_features = None

# 幅度模型 / 仓位系数实验参数（可通过环境变量调整，不影响未加载幅度模型的生产回测）
BASE_TARGET_SIZE = float(os.environ.get('BASE_TARGET_SIZE', '0.05'))
POS_MULT_WEIGHT = float(os.environ.get('POS_MULT_WEIGHT', '1.0'))
POS_MULT_BIAS = float(os.environ.get('POS_MULT_BIAS', '0.0'))
OPPORT_SIZING_COEFF = float(os.environ.get('OPPORT_SIZING_COEFF', '0.15'))
OPPORT_SIZING_MIN = float(os.environ.get('OPPORT_SIZING_MIN', '0.5'))
OPPORT_SIZING_MAX = float(os.environ.get('OPPORT_SIZING_MAX', '1.5'))

def load_magnitude_models(opport_pkl=None, risk_pkl=None):
    """热加载机会/风险幅度模型（可选）。"""
    global GLOBAL_OPPORT_MAG_PKG, GLOBAL_RISK_MAG_PKG
    global trained_opport_mag_lgbm, opport_mag_features
    global trained_risk_mag_lgbm, risk_mag_features
    if opport_pkl is not None:
        GLOBAL_OPPORT_MAG_PKG = joblib.load(opport_pkl)
        trained_opport_mag_lgbm = GLOBAL_OPPORT_MAG_PKG['model']
        opport_mag_features = GLOBAL_OPPORT_MAG_PKG['features']
        print(f"机会幅度模型已加载: {opport_pkl}")
    if risk_pkl is not None:
        GLOBAL_RISK_MAG_PKG = joblib.load(risk_pkl)
        trained_risk_mag_lgbm = GLOBAL_RISK_MAG_PKG['model']
        risk_mag_features = GLOBAL_RISK_MAG_PKG['features']
        print(f"风险幅度模型已加载: {risk_pkl}")

# pm = ProxyManager(config_file="./proxies.json")
fetcher = BaostockCodeFetcher()
stock_data_cache = LocalDataCache(code_fetcher=fetcher, cache_dir="./stock_data_cache")

GLOBAL_MARKET_STATS = pd.DataFrame()
GLOBAL_SCREEN_THRESHOLDS = {}

def get_price_limit_rate(symbol):
    if symbol.startswith(('30', '68')):
        return 0.2
    if symbol.startswith(('92', '87','83', '43')):
        return 0.3
    return 0.1

# ===== 财务数据加载的物理整合与向量化匹配 =====
_financial_data = None
_financial_dict = {}

def load_financial_data():
    global _financial_data, _financial_dict
    if _financial_data is None:
        _financial_data = pd.read_csv(
            'financial_reports_all.csv',
            dtype={
                '股票代码': 'str',
                '每股收益': 'float64',
                '净利润-净利润': 'float64',
            },
            parse_dates=['报告日期', '最新公告日期']
        )
        
        _financial_data = _financial_data.sort_values(by=['股票代码', '报告日期'], ascending=[True, False])
        
        for symbol, group in _financial_data.groupby('股票代码'):
            dates = group['报告日期'].values 
            _financial_dict[symbol] = (dates, group)
            
        _financial_data.set_index(['股票代码', '报告日期'], inplace=True)
        print(f"财务数据加载并缓存完成，共 {len(_financial_data)} 条记录")
    return _financial_data

def get_precomputed_financials_for_symbol(symbol):
    load_financial_data()
    if symbol not in _financial_dict:
        return None
    _, group = _financial_dict[symbol]
    
    # 升序排序，便于差分运算
    fin_df = group.sort_values('报告日期', ascending=True).copy()
    
    required_cols = ['净利润-净利润', '净利润-同比增长', '每股收益', '每股经营现金流量', '每股净资产', '净资产收益率']
    for col in required_cols:
        if col not in fin_df.columns:
            fin_df[col] = 0.0
            
    fin_df['is_profit_ok'] = (
        (fin_df['净利润-净利润'] > 0) &
        (fin_df['净利润-同比增长'] > 0) &
        (fin_df['每股收益'] > 0)
        # (fin_df['每股经营现金流量'] > 0)
    )
    fin_df['roe_up'] = fin_df['净资产收益率'].diff().fillna(0.0)
    
    return fin_df[['报告日期', 'is_profit_ok', '每股净资产', 'roe_up']]

class AKShareChipDataSource(DataSource):
    def __init__(self):
        super().__init__()
        self.model_schedule = None
        pybroker.register_columns(
            'symbol',
            # 'open', 'high', 'low', 'amount','amplitude','change','change_pct',
            'close', 'turnover','amount_ma20',
            'profit_ratio', 
            # 'concentration_90','avg_cost', 
            'concentration_70','chip_penetration',
            # 'vol_ma5', 'vol_ma20',
            'close_ma20',# 'close_ma5', 'close_ma10', 'close_ma60',
            # 'turnover_ma10',
            # 'limit_up',
            'atr','atr_ratio',
            #  'atr_ratio_zscore',
            'obv', 
            # 'obv_ema',
            'adx', 
            # 'plus_di', 'minus_di',
            'bias_20',#'bias_20_avg','bias_20_std',
            # 'price_to_cost', 'turnover_ratio', 'conc_70_slope',
            # 'ma_squeeze',
            # 优化新增的向量化列，用于高效率回测循环访问
            'is_profit_ok', 
            # 'bp_ratio', 'roe_up',
            # --- v4 新增：截面列与模型分 ---
            # 'ma_squeeze_zscore', 'profit_ratio_zscore', 'price_to_cost_zscore',
            'ml_rank','risk_ml_rank'
        )
        # 动态注册幅度模型列（若已加载）
        extra_cols = []
        if trained_opport_mag_lgbm is not None:
            extra_cols.append('opport_mag')
        if trained_risk_mag_lgbm is not None:
            extra_cols.append('risk_mag')
        if extra_cols:
            pybroker.register_columns(*extra_cols)

    def _fetch_data(self, symbols, start_date, end_date, timeframe, adjust="qfq", online=False,
                    model_schedule=None):
        """
        重构后的数据准备函数：实现双通道特征对齐与内存优化
        - 入场模型：特征全平滑 (use_smooth=True)
        - 风险模型：特征不平滑 (use_smooth=False)
        - model_schedule: [(seg_start, seg_end, entry_pkl, risk_pkl), ...] 滚动分段模型调度。
          为空时用全局现役模型对全区间打分；提供时按日期段切换对应 fold 模型。
        """
        if model_schedule is None:
            model_schedule = getattr(self, 'model_schedule', None)
        # 用于分别存储平滑流和原始流的数据
        all_data_smooth = []
        all_data_raw = []
        
        # 0. 加载个股->申万一级行业映射 (行业内排名特征用; 缺失时 ind_inner_rank 填中性)
        try:
            from industry_data import load_industry_map
            industry_map = load_industry_map()
        except Exception as e:
            logging.warning(f"行业映射加载失败, ind_inner_rank 将填中性值: {e}")
            industry_map = {}

        # 1. 准备大盘环境快照 (Market Context)
        logging.info("Step 1: 准备大盘环境快照...")
        try:
            mkt_factors = co_compute.calculate_global_mkt_factors('zzqz_df.xlsx')
            # FIX: 必须按日期索引，否则 df['date'].map(mkt_factors[col]) 会按 RangeIndex 映射全部失败
            if mkt_factors is not None and 'date' in mkt_factors.columns:
                mkt_factors['date'] = pd.to_datetime(mkt_factors['date'])
                mkt_factors = mkt_factors.set_index('date')
        except Exception as e:
            logging.error(f"大盘数据加载失败: {e}")
            mkt_factors = None

        # 1.5 行业排名表 (行业级共享常数; 缺失时 ind_rank_20/60 填中性)
        ind_rank_table = None
        if mkt_factors is not None:
            try:
                from industry_data import load_industry_daily
                ind_daily = load_industry_daily()
                if not ind_daily.empty:
                    ind_rank_table = co_compute.calculate_industry_rank_table(ind_daily, mkt_factors)
            except Exception as e:
                logging.warning(f"行业排名表计算失败, ind_rank_20/60 将填中性值: {e}")

        for symbol in symbols:
            try:
                # 获取原始行情数据
                df = stock_data_cache.get_stock_data(symbol, start_date, end_date, online=online)
                if df.empty: continue

                # 基础清理与数值防御
                df['symbol'] = df['symbol'].astype(str).str.replace('.0', '', regex=False)
                df['date'] = pd.to_datetime(df['date'], format='mixed')
                df['is_suspended'] = df['open'].isna()
                
                df['close'] = df['close'].ffill()
                df[['open', 'high', 'low']] = df[['open', 'high', 'low']].fillna(df['close'])
                df['vwap'] = ((df['high'] + df['low'] + 2 * df['close']) / 4.0).clip(0.01)

                # 计算基础业务指标 (这部分在分叉前计算，避免重复运算)
                df['close_ma20'] = talib.SMA(df['close'].values, 20)
                df['amount_ma20'] = talib.SMA(df['amount'].values, 20)
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
                df['atr_ratio'] = df['atr'] / df['close']
                df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14).fillna(20)
                df['obv'] = talib.OBV(df['close'], df['volume']).fillna(0)
                
                fin_df = get_precomputed_financials_for_symbol(symbol)
                if fin_df is not None and not fin_df.empty:
                    fin_df = fin_df.rename(columns={'报告日期': 'date'})
                    fin_df['date'] = pd.to_datetime(fin_df['date'])
                    df = pd.merge_asof(df.sort_values('date'), fin_df.sort_values('date'), on='date', direction='backward')
                    # FIX: 财务数据缺失时前向填充；仍缺失的用中性净资产 1.0，避免 NaN 污染 bp_ratio
                    df['每股净资产'] = df['每股净资产'].ffill().bfill().fillna(1.0)
                    df['is_profit_ok'] = df['is_profit_ok'].fillna(False).astype(bool)
                    df['roe_up'] = df['roe_up'].fillna(0.0)
                    df['bp_ratio'] = (df['close'] / df['每股净资产'].replace(0, np.nan)).fillna(1.0)
                else:
                    df['is_profit_ok'], df['roe_up'], df['bp_ratio'] = False, 0.0, 1.0

                # 过滤停牌与面值退市风险股
                df = df[(df['is_suspended'] == False) & (df['close'] >= 1.0)].copy()
                df = df[df['date'] >= pd.to_datetime(start_date)].copy()
                df = df.drop(columns=['is_suspended'])
                
                if df.empty: continue

                # ==========================================================
                # 【核心分叉点】：复制基础数据，分别计算平滑与非平滑特征
                # ==========================================================
                # 1) 平滑流 (用于入场模型)
                df_smooth = co_compute.compute_individual_indicators(df.copy(), mkt_factors, use_smooth=True)
                all_data_smooth.append(df_smooth)

                # 2) 原始流 (用于风控离场模型)
                df_raw = co_compute.compute_individual_indicators(df.copy(), mkt_factors, use_smooth=False)
                all_data_raw.append(df_raw)

            except Exception as e:
                print(f"个股处理失败: {symbol}, {e}")

        if not all_data_smooth: return pd.DataFrame()

        # 合并全市场数据
        final_df_smooth = pd.concat(all_data_smooth).reset_index(drop=True)
        final_df_raw = pd.concat(all_data_raw).reset_index(drop=True)
        
        # 释放临时列表内存
        del all_data_smooth, all_data_raw
        import gc
        gc.collect()

        # ==========================================================
        # 2. 全局指标计算 (基于平滑流样本池计算大盘广度等)
        # ==========================================================
        global GLOBAL_MARKET_STATS, GLOBAL_SCREEN_THRESHOLDS
        try:
            GLOBAL_MARKET_STATS = co_compute.calculate_high_low_stats(final_df_smooth)
            GLOBAL_MARKET_STATS.set_index('date', inplace=True)
            GLOBAL_MARKET_STATS = GLOBAL_MARKET_STATS.sort_index()
        except Exception as e:
            logging.error(f"全局指标计算失败: {e}")
            raise

        # 映射大盘/环境列到两个数据流中 (确保两边基础环境一致)
        # FIX: 使用 MKT_RAW_FEATURES 映射原始列，再生成 PCA 正交大盘特征喂入模型

        # 先构建滚动 PCA 市场特征表（统一调用 co_compute 公共组件）
        raw_cols = [c for c in co_compute.FeatureConfig.MKT_RAW_FEATURES]
        mkt_raw_for_pca = GLOBAL_MARKET_STATS.reset_index()[['date'] + [c for c in raw_cols if c in GLOBAL_MARKET_STATS.columns]]
        for col in raw_cols:
            if col not in mkt_raw_for_pca.columns and mkt_factors is not None and col in mkt_factors.columns:
                mkt_raw_for_pca[col] = mkt_raw_for_pca['date'].map(mkt_factors[col])
        mkt_pc_df = co_compute.build_market_pca_table(mkt_raw_for_pca, min_periods=60)

        def _attach_market_features(df_temp):
            for col in raw_cols:
                if col in mkt_factors.columns:
                    mapped = df_temp['date'].map(mkt_factors[col])
                else:
                    mapped = df_temp['date'].map(GLOBAL_MARKET_STATS[col])
                df_temp[col] = mapped.ffill()
                if df_temp[col].isna().any():
                    raise ValueError(f"大盘特征 {col} 存在无法映射的日期，请检查 mkt_factors 覆盖范围")
            for col in ['congestion', 'high20_ratio', 'low20_ratio']:
                if df_temp[col].isna().any():
                    logging.warning(f"全局统计特征 {col} 存在 {df_temp[col].isna().sum()} 个 NaN，使用业务中性值填充")
                    df_temp[col] = df_temp[col].fillna({'congestion': 0.35, 'high20_ratio': 0.1, 'low20_ratio': 0.1}.get(col))
            # 合并 PCA 正交大盘特征
            df_temp = df_temp.merge(mkt_pc_df, on='date', how='left')
            return df_temp

        final_df_smooth = _attach_market_features(final_df_smooth)
        final_df_raw = _attach_market_features(final_df_raw)

        # ==========================================================
        # 3. 横截面标准化 (Z-Score) - 两个通道独立进行，防止特征值污染
        # ==========================================================
        final_df_smooth = co_compute.apply_standardization(
            final_df_smooth, industry_map=industry_map, ind_rank_table=ind_rank_table)
        final_df_raw = co_compute.apply_standardization(
            final_df_raw, industry_map=industry_map, ind_rank_table=ind_rank_table)

        # ==========================================================
        # 4. 执行机器学习推理
        # ==========================================================
        if model_schedule:
            # 滚动分段推理：按日期段加载对应 fold 的入场+风控模型打分
            raw_score = np.full(len(final_df_smooth), np.nan)
            risk_score = np.full(len(final_df_raw), np.nan)
            for seg_start, seg_end, entry_pkl, risk_pkl in model_schedule:
                seg_start, seg_end = pd.Timestamp(seg_start), pd.Timestamp(seg_end)
                mask_s = (final_df_smooth['date'] >= seg_start) & (final_df_smooth['date'] <= seg_end)
                if mask_s.sum() == 0:
                    continue
                epkg = joblib.load(entry_pkl)
                epkg_feats = epkg['features']
                raw_score[mask_s] = epkg['model'].predict(final_df_smooth.loc[mask_s, epkg_feats])

                mask_r = (final_df_raw['date'] >= seg_start) & (final_df_raw['date'] <= seg_end)
                rpkg = joblib.load(risk_pkl)
                rpkg_feats = rpkg['features']
                risk_score[mask_r] = rpkg['model'].predict(final_df_raw.loc[mask_r, rpkg_feats])
            # 3) 合并预测得分 (防御性校验确保行数和索引完全一致)
            assert len(final_df_smooth) == len(final_df_raw), "平滑通道与原始通道行数不一致，对齐失败"
            final_df_smooth['raw_ml_score'] = raw_score
            final_df_smooth['risk_ml_score'] = risk_score
            # 确认每个 fold 段都被打分覆盖，避免 NaN 泄漏进回测
            uncovered = final_df_smooth['raw_ml_score'].isna().sum()
            if uncovered > 0:
                raise ValueError(f"model_schedule 未覆盖所有日期段，{uncovered} 行缺分")
        else:
            # 1) 入场模型推理 (使用平滑后的特征输入)
            # 【修复】改用模型自身保存的 features, 而非动态配置, 避免特征改造后新旧模型列数不匹配
            model_input_features = [f for f in model_features]
            final_df_smooth['raw_ml_score'] = trained_lgbm.predict(final_df_smooth[model_input_features])

            # 2.5) 可选：两阶段头部精排模型
            # 先用一阶段模型粗筛每日 top 30%，再对 top 30% 用二阶段模型重新打分
            stage2_model_pkl = os.environ.get('STAGE2_MODEL_PKL')
            if stage2_model_pkl and os.path.exists(stage2_model_pkl):
                stage2_pkg = joblib.load(stage2_model_pkl)
                stage2_model = stage2_pkg['model']
                stage2_features = [f for f in stage2_pkg['features']]
                final_df_smooth['stage1_rank'] = final_df_smooth.groupby('date')['raw_ml_score'].rank(pct=True, ascending=False)
                # 默认用一阶段分数；top 30% 替换为二阶段分数
                final_df_smooth['final_ml_score'] = final_df_smooth['raw_ml_score']
                mask_top30 = final_df_smooth['stage1_rank'] <= 0.30
                if mask_top30.any():
                    final_df_smooth.loc[mask_top30, 'final_ml_score'] = stage2_model.predict(
                        final_df_smooth.loc[mask_top30, stage2_features]
                    )
                # 让后 70% 分数远低于 top 30%，确保 ml_rank 反映二阶段排序
                bottom_min = final_df_smooth.loc[mask_top30, 'final_ml_score'].min() if mask_top30.any() else final_df_smooth['final_ml_score'].min()
                final_df_smooth.loc[~mask_top30, 'final_ml_score'] = bottom_min - 10.0
                final_df_smooth['raw_ml_score'] = final_df_smooth['final_ml_score']
                final_df_smooth = final_df_smooth.drop(columns=['stage1_rank', 'final_ml_score'])

            # 2) 风险模型推理 (使用非平滑的脉冲敏感型特征输入)
            risk_model_input_features = [f for f in model_risk_features]
            final_df_raw['risk_ml_score'] = trained_risk_lgbm.predict(final_df_raw[risk_model_input_features])

            # 3) 合并预测得分 (防御性校验确保行数和索引完全一致)
            assert len(final_df_smooth) == len(final_df_raw), "平滑通道与原始通道行数不一致，对齐失败"
            final_df_smooth['risk_ml_score'] = final_df_raw['risk_ml_score'].values

        # 3.5) 可选：幅度模型推理
        if trained_opport_mag_lgbm is not None:
            final_df_smooth['opport_mag'] = trained_opport_mag_lgbm.predict(
                final_df_smooth[opport_mag_features]
            )
        if trained_risk_mag_lgbm is not None:
            final_df_smooth['risk_mag'] = trained_risk_mag_lgbm.predict(
                final_df_raw[risk_mag_features]
            )

        # 4) 立即销毁原始通道数据，释放宝贵的内存空间
        del final_df_raw
        gc.collect()

        # 后续操作统一基于 final_df_smooth 展开 (重命名为 final_df 以契合后续逻辑)
        final_df = final_df_smooth

        # 计算排序值
        final_df['ml_rank'] = final_df.groupby('date')['raw_ml_score'].rank(pct=True, ascending=False)
        final_df['risk_ml_rank'] = final_df.groupby('date')['risk_ml_score'].rank(pct=True, ascending=True)

        # 分位数风控轨道计算
        for q in [30, 50, 75, 90]:
            GLOBAL_MARKET_STATS[f'low20_q{q}'] = GLOBAL_MARKET_STATS['low20'].rolling(120, min_periods=30).quantile(q/100)

        # 每日截面阈值字典
        daily_thresholds = final_df.groupby('date').agg({
            'amount_ma20': lambda x: x.quantile(0.3),
            'atr_ratio': lambda x: x.quantile(0.2)
        })
        GLOBAL_SCREEN_THRESHOLDS = daily_thresholds.to_dict(orient='index')
        
        # 调试输出 (默认关闭避免 9GB 磁盘占用, 检查诊断时用 DEBUG_INFERENCE=1 开启)
        if os.environ.get('DEBUG_INFERENCE') == '1':
            final_df.to_csv('debug_inference_results.csv', index=False)
            logging.info(f"!!! 诊断文件已生成: debug_inference_results.csv")

        # ==========================================================
        # 5. 内存瘦身：仅保留回测必需的列
        # ==========================================================
        registered_cols = [
            'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover',
            'amount_ma20', 'close_ma20', 'profit_ratio', 'concentration_70',
            'chip_penetration', 'atr', 'atr_ratio', 'obv', 'vol_dryness', 'adx',
            'bias_20', 'ma_squeeze', 'ml_rank', 'risk_ml_rank', 'is_profit_ok'
        ]
        if trained_opport_mag_lgbm is not None:
            registered_cols.append('opport_mag')
            # 机会幅度的日度 Z-Score，用于个股仓位 sizing
            final_df['opport_mag_z'] = final_df.groupby('date')['opport_mag'].transform(
                lambda x: ((x - x.mean()) / (x.std() + 1e-9)).clip(-3, 3).fillna(0.0)
            )
            registered_cols.append('opport_mag_z')
        if trained_risk_mag_lgbm is not None:
            registered_cols.append('risk_mag')
        
        # 提取列并强制释放内存
        final_df = final_df[[c for c in registered_cols if c in final_df.columns]].copy()
        gc.collect()
        
        logging.info(f"数据准备完毕。样本总数: {len(final_df)}")
        return final_df
    

# ===== 真正适合 PyBroker 的自定义指标区 (1-D 数组) =====
@njit(cache=True)
def numba_rolling_quantile(arr, window, percentile):
    n = len(arr)
    res = np.full(n, np.nan)
    if n < window:
        return res
    min_periods = window // 3
    for i in range(n):
        start = max(0, i - window + 1)
        window_slice = arr[start:i+1]
        valid_slice = window_slice[~np.isnan(window_slice)]
        if len(valid_slice) >= min_periods:
            res[i] = np.percentile(valid_slice, percentile * 100)
    return res

def x_quantiles(bar_data, indicator, lookback, percentile=0.8):
    values = getattr(bar_data, indicator)
    if values is None or len(values) == 0:
        # 如果没有数据，返回一个与时间轴等长的 NaN 数组，避免进入 Numba
        return np.full(len(bar_data.date), np.nan)
    
    # 3. 检查窗口大小是否超过数据长度（可选，但建议增加以增强健壮性）
    if len(values) < lookback:
        return np.full(len(values), np.nan)

    # 4. 只有确保 values 是有效的 ndarray 且非空时，才调用 Numba 函数
    return numba_rolling_quantile(values, lookback, percentile)


turnover_q90 = pybroker.indicator('turnover_q90', x_quantiles,indicator='turnover',lookback=90,percentile=0.9)
adx_q_over = pybroker.indicator('adx_q_over', x_quantiles,indicator='adx',lookback=90,percentile=0.95)
# adx_q_high = pybroker.indicator('adx_q_high', x_quantiles,indicator='adx',lookback=90,percentile=0.9)
# adx_q_low = pybroker.indicator('adx_q_low', x_quantiles,indicator='adx',lookback=90,percentile=0.8)
# adx_q_pass = pybroker.indicator('adx_q_pass', x_quantiles,indicator='adx',lookback=90,percentile=0.5)
# bias_20_q_over = pybroker.indicator('bias_20_q_over', x_quantiles,indicator='bias_20',lookback=90,percentile=0.95)
bias_20_q_high = pybroker.indicator('bias_20_q_high', x_quantiles,indicator='bias_20',lookback=90,percentile=0.8)
# bias_20_q_low = pybroker.indicator('bias_20_q_low', x_quantiles,indicator='bias_20',lookback=90,percentile=0.2)
# conc70_q_high = pybroker.indicator('conc70_q_high', x_quantiles,indicator='concentration_70',lookback=90,percentile=0.8)
conc70_q_low = pybroker.indicator('conc70_q_low', x_quantiles,indicator='concentration_70',lookback=90,percentile=0.2)
# profit_ratio_q5 = pybroker.indicator( 'profit_ratio_q5', x_quantiles, indicator='profit_ratio', lookback=90, percentile=0.05)
profit_ratio_q50 = pybroker.indicator( 'profit_ratio_q50', x_quantiles, indicator='profit_ratio', lookback=90, percentile=0.5)
profit_ratio_q20 = pybroker.indicator( 'profit_ratio_q20', x_quantiles, indicator='profit_ratio', lookback=90, percentile=0.2)
# chip_penetration_q90 = pybroker.indicator( 'chip_penetration_q90', x_quantiles, indicator='chip_penetration', lookback=90, percentile=0.9)

@njit(cache=True)
def numba_calc_conc_down(slopes, lookback):
    n = len(slopes)
    res = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if np.isnan(slopes[i]) or slopes[i] >= 0:
            res[i] = False
            continue
            
        start = max(0, i - lookback + 1)
        window = slopes[start:i+1]
        neg_slopes = window[window < 0]
        neg_slopes = neg_slopes[~np.isnan(neg_slopes)]
        
        if len(neg_slopes) >= 2:
            ref = np.percentile(neg_slopes, 90)
            res[i] = slopes[i] < ref
        elif len(neg_slopes) == 1:
            res[i] = True 
        else:
            res[i] = False
    return res

@njit(cache=True)
def numba_calc_conc_up(slopes, lookback):
    n = len(slopes)
    res = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if np.isnan(slopes[i]) or slopes[i] <= 0:
            res[i] = False
            continue
            
        start = max(0, i - lookback + 1)
        window = slopes[start:i+1]
        pos_slopes = window[window > 0]
        pos_slopes = pos_slopes[~np.isnan(pos_slopes)]
        
        if len(pos_slopes) >= 2:
            ref = np.percentile(pos_slopes, 10)
            res[i] = slopes[i] > ref
        elif len(pos_slopes) == 1:
            res[i] = True
        else:
            res[i] = False
    return res

def conc_ud(bar_data, lookback=90, direction='down'):
    s70_arr = bar_data.concentration_70
    slope70 = talib.LINEARREG_SLOPE(s70_arr, timeperiod=5) * 1000
    
    if direction == 'down':
        return numba_calc_conc_down(slope70, lookback)
    if direction == 'up':
        return numba_calc_conc_up(slope70, lookback)
    
    return np.zeros_like(s70_arr, dtype=bool)

conc_down = pybroker.indicator('conc_down', conc_ud, lookback=90,direction='down')
conc_up = pybroker.indicator('conc_up', conc_ud, lookback=90,direction='up')

# 开启这些单列计算的缓存
# pybroker.enable_indicator_cache('turnover_q90')
# pybroker.enable_indicator_cache('conc70_q_high')
# pybroker.enable_indicator_cache('conc70_q_low')
# pybroker.enable_indicator_cache('profit_ratio_q5')
# pybroker.enable_indicator_cache('profit_ratio_q10')
# pybroker.enable_indicator_cache('profit_ratio_q20')
# pybroker.enable_indicator_cache('bias_20_q_high')
# pybroker.enable_indicator_cache('bias_20_q_low')
# pybroker.enable_indicator_cache('conc_down')
# pybroker.enable_indicator_cache('conc_up')
# pybroker.enable_indicator_cache('chip_penetration_q90')
# pybroker.enable_indicator_cache('is_profit_ok_ind')
# pybroker.enable_indicator_cache('bp_ratio_ind')
# pybroker.enable_indicator_cache('roe_up_ind')

# ===== 其他环境逻辑保持不变 =====
def calculate_money_supply_signal(money_df, date):
    target_date = pd.Timestamp(date)
    if target_date.day < 15:
        lookup_date = target_date.replace(day=1) - pd.DateOffset(months=1)
    else:
        lookup_date = target_date.replace(day=1)
        
    available_df = money_df[money_df.index < lookup_date]
    if len(available_df) < 6:
        return 0.5

    recent = available_df.tail(4)
    y_scissors = recent['剪刀差'].values
    y_m1 = recent['货币(狭义货币M1)同比增长'].values
    x = np.arange(len(y_scissors))
    
    slope_scissors = np.polyfit(x, y_scissors, 1)[0]
    slope_m1 = np.polyfit(x, y_m1, 1)[0]
    
    curr_val = y_scissors[-1]  
    prev_val = y_scissors[-2]  
    
    if slope_scissors < -1.2 and curr_val < -4.0:
        return 0.1  
    
    if curr_val > prev_val or slope_scissors > 0.1:
        return 0.8 if slope_m1 > 0 else 0.6
    
    if slope_scissors < -1 or curr_val < -1.0:
        return 0.3  
        
    return 0.5

money_df = pd.read_excel(
    'macro_china_supply_of_money_df.xlsx',
    dtype={
        '统计时间': 'str',
        '货币和准货币（广义货币M2）': 'float32',
        '货币和准货币（广义货币M2）同比增长': 'float32',
        '货币(狭义货币M1)': 'float32',
        '货币(狭义货币M1)同比增长': 'float32',
        '流通中现金(M0)': 'float32',
        '流通中现金(M0)同比增长': 'float32'
    }
)
money_df['统计时间'] = pd.to_datetime(money_df['统计时间'], format='%Y.%m')
money_df['剪刀差'] = money_df['货币(狭义货币M1)同比增长'] - money_df['货币和准货币（广义货币M2）同比增长']
money_df.set_index('统计时间', inplace=True)
money_df.sort_index(inplace=True)

zzqz_df = pd.read_excel('zzqz_df.xlsx')
zzqz_df = zzqz_df.rename(columns={
    '日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low',
    '收盘': 'close', '成交量': 'volume', '成交额': 'amount',
    '振幅': 'amplitude', '涨跌幅': 'change_pct', '涨跌额': 'change', '换手率': 'turnover'
})      
zzqz_df['date'] = pd.to_datetime(zzqz_df['date'], format='%Y-%m-%d')
zzqz_df.set_index('date', inplace=True)
zzqz_df['vol_ma5'] = zzqz_df['volume'].rolling(5).mean()
zzqz_df['vol_ma20'] = zzqz_df['volume'].rolling(20).mean()
zzqz_df['vol_ma60'] = zzqz_df['volume'].rolling(60).mean()
zzqz_df['close_ma5'] = zzqz_df['close'].rolling(5).mean()
zzqz_df['close_ma10'] = zzqz_df['close'].rolling(10).mean()
zzqz_df['close_ma20'] = zzqz_df['close'].rolling(20).mean()
zzqz_df['close_ma60'] = zzqz_df['close'].rolling(60).mean()
zzqz_df['close_q30_w20'] = zzqz_df['close'].rolling(20, min_periods=5).quantile(0.3)
zzqz_df['close_q30_w60'] = zzqz_df['close'].rolling(60, min_periods=15).quantile(0.3)
zzqz_df['volume_q15_w120'] = zzqz_df['volume'].rolling(120, min_periods=30).quantile(0.15)
zzqz_df['close_max_w20'] = zzqz_df['close'].rolling(20, min_periods=5).max()  # 用于谨慎高位判定

_daily_market_cache = {}
# =========================================================================
# 函数 1：【核心提取】统一计算量价背离与趋势信号（不含买卖执行，纯算法）
# =========================================================================
def get_trend_signals(ctx, scenario):
    close = ctx.close[-1]
    turnover = ctx.turnover
    turnover_q90 = ctx.indicator('turnover_q90')[-1]
    
    bias_20 = ctx.bias_20[-1]
    bias_20_q_high = ctx.indicator('bias_20_q_high')[-1]
    # bias_20_q_over = ctx.indicator('bias_20_q_over')[-1]
    
    adx = ctx.adx[-1]
    # adx_q_high = ctx.indicator('adx_q_high')[-1]
    adx_q_over = ctx.indicator('adx_q_over')[-1]
    # adx_q_low = ctx.indicator('adx_q_low')[-1]
    # adx_q_pass = ctx.indicator('adx_q_pass')[-1]
    
    lookback = 30  # 适当拉长回溯期以捕捉明显的波峰波谷
    bullish_divergence = False
    bearish_divergence = False

    if len(ctx.obv) >= lookback + 5:
        # 1. 提取当前数据（使用均值平滑噪点）
        curr_close = ctx.close[-1]
        curr_obv = np.mean(ctx.obv[-3:])  # 最近3天OBV均值
        
        # 2. 确定参考区间（回溯期内，排除最近5天的干扰）
        ref_close = ctx.close[-lookback:-5]
        ref_obv = ctx.obv[-lookback:-5]
        
        # --- 牛背离逻辑：价格创新低，OBV不创新低 ---
        # 找到参考期内的价格最低点及其对应的 OBV
        idx_price_min = np.argmin(ref_close)
        price_low_ref = ref_close[idx_price_min]
        obv_at_price_low = ref_obv[idx_price_min]
        
        # 判断：当前价格低于前低，但当前OBV高于前低时的OBV
        bullish_divergence = (curr_close < price_low_ref) and (curr_obv > obv_at_price_low)
        
        # --- 熊背离逻辑：价格创新高，OBV不创新高 ---
        # 找到参考期内的价格最高点及其对应的 OBV
        idx_price_max = np.argmax(ref_close)
        price_high_ref = ref_close[idx_price_max]
        obv_at_price_high = ref_obv[idx_price_max]
        
        # 判断：当前价格高于前高，但当前OBV低于前高时的OBV
        bearish_divergence = (curr_close > price_high_ref) and (curr_obv < obv_at_price_high)

    # 信号默认状态初始化
    # sell_signal_confirmed = True
    # # buy_signal_confirmed = adx < adx_q_over and bias_20 < bias_20_q_over and not bearish_divergence
    # buy_signal_confirmed = True
    
    # # 结合市场四大场景（底/机/警/他）微调买卖信号
    if 'bottom' in scenario:
        # buy_signal_confirmed = True
        tend_broke = (turnover[-1] > turnover_q90) and bearish_divergence and (bias_20 > bias_20_q_high)
    elif 'opportunity' in scenario:
        # buy_signal_confirmed = buy_signal_confirmed and adx > adx_q_pass
        tend_broke = bearish_divergence
    elif 'caution' in scenario:
        # buy_signal_confirmed = buy_signal_confirmed and adx > adx_q_high and bullish_divergence
        tend_broke = tend_broke = (turnover[-1] > turnover_q90) and bearish_divergence and (bias_20 > bias_20_q_high)
    else:
        # buy_signal_confirmed = buy_signal_confirmed and adx > adx_q_high
        tend_broke = bearish_divergence

    return bullish_divergence, tend_broke

BUY_ELIGIBILITY_DETAILS = []
# 买入资格与打分逻辑已迁移至 signal_engine.check_buy_eligibility_and_score


def before_exec_fn(ctx_map):
    if not ctx_map:
        return

    first_ctx = next(iter(ctx_map.values()))
    current_dt = first_ctx.dt
    available_market = GLOBAL_MARKET_STATS[GLOBAL_MARKET_STATS.index < current_dt]
    current_market = available_market.iloc[-1]

    # 1. 每天仅计算一次大盘环境并统一存入缓存
    if current_dt not in _daily_market_cache:
        money_sig = calculate_money_supply_signal(money_df, current_dt)
        
        # 提取过去 500 个交易日的拥挤度数据
        recent_congestion = available_market['congestion'].tail(500)
        current_congestion = current_market['congestion']
        # 动态阈值：过去两年的 96 分位数
        dynamic_threshold = recent_congestion.quantile(0.99)
        # 熔断开关
        congestion_too_high = (current_congestion > dynamic_threshold)
            
        mkt_status = is_market_ok.scenario_based_market_judgment(current_dt, zzqz_df, GLOBAL_MARKET_STATS,len(ctx_map))
        
        current_dt_ts = pd.Timestamp(current_dt).normalize() 
        # 使用 Timestamp 对象直1从字典取值
        day_limit = GLOBAL_SCREEN_THRESHOLDS.get(current_dt_ts, {'amount_ma20': 0, 'atr_ratio': 0})
        primary_scenario = mkt_status['primary_scenario']

        scenario_map = {
            'opportunity': float(os.environ.get('ML_THRESH_OPPORTUNITY', '0.02')),
            'bottom': float(os.environ.get('ML_THRESH_BOTTOM', '0.03')),
            'normal': float(os.environ.get('ML_THRESH_NORMAL', '0.01')),
            'caution': float(os.environ.get('ML_THRESH_CAUTION', '0.01')),
            'risk': float(os.environ.get('ML_THRESH_RISK', '0.01')),
        }
        daily_ml_threshold = scenario_map[primary_scenario]

        _daily_market_cache[current_dt] = {
            'money_supply_signal': money_sig,
            'congestion_too_high': congestion_too_high,
            'is_market_ok': mkt_status['is_market_ok'],
            'primary_scenario': primary_scenario,
            'position_multiplier': mkt_status['position_multiplier'],
            'decision_reason': mkt_status['decision_reason'],
            'day_limit': day_limit,
            'daily_ml_threshold': daily_ml_threshold
        }
        
    daily_env = _daily_market_cache[current_dt]
    daily_candidates = []
    scenario = daily_env['primary_scenario']
    quota_override = os.environ.get('BUY_QUOTA_OVERRIDE')
    if quota_override is not None:
        buy_quota = int(quota_override)
    elif 'bottom' in scenario:
        buy_quota = int(os.environ.get('BUY_QUOTA_BOTTOM', '5'))
    elif 'opportunity' in scenario:
        buy_quota = int(os.environ.get('BUY_QUOTA_OPPORTUNITY', '5'))
    elif 'normal' in scenario:
        buy_quota = int(os.environ.get('BUY_QUOTA_NORMAL', '5'))
    elif 'caution' in scenario:
        buy_quota = int(os.environ.get('BUY_QUOTA_CAUTION', '5'))
    elif 'risk' in scenario:
        buy_quota = int(os.environ.get('BUY_QUOTA_RISK', '5'))
    else:
        buy_quota = 5

    for symbol, ctx in ctx_map.items():
        if ctx.long_pos(): 
            continue
            
        is_eligible, ml_rank, audit = signal_engine.check_buy_eligibility_and_score(ctx, daily_env)
        BUY_ELIGIBILITY_DETAILS.append(audit)
        if is_eligible:
            daily_candidates.append((symbol, ml_rank))        
    # ml_rank 是 pct rank(ascending=False)，越小代表模型打分越高，应升序取最头部
    daily_candidates.sort(key=lambda x: x[1])
    daily_env['top_x_buys'] = set([x[0] for x in daily_candidates[:buy_quota]])

    if len(daily_candidates) > 0:
        # top_scores = [round(x[1], 2) for x in daily_candidates[:3]]
        logging.info(
            f"[{current_dt.date()}] 场景: {scenario} | "
            f"大盘过线: {daily_env['is_market_ok']} | "
            f"全市场: {len(ctx_map)}只 | "
            f"过硬门槛: {len(daily_candidates)}只 | "
            f"今日限额: {buy_quota}只 | "
            # f"Top3得分: {top_scores}"
            f"最总过关: {len(daily_env.get('top_x_buys', set()))}只"
        )


# =========================================================================
# 函数 4：【订单执行】最终的策略下单网关（买入实行流速卡死，卖出实行独立看盘）
# =========================================================================
EXIT_SNAPSHOTS = []
def chip_strategy(ctx):
    position = ctx.long_pos()
    
    if len(ctx.close) < 1 or ctx.close[-1] <= 0:
        return 
        
    close = ctx.close[-1]
    symbol = ctx.symbol
    rate = get_price_limit_rate(symbol)
    ml_rank = ctx.ml_rank[-1]
    risk_ml_rank = ctx.risk_ml_rank[-1]
    atr = ctx.atr[-1]
    
    daily_env = _daily_market_cache.get(ctx.dt)
    primary_scenario  = daily_env['primary_scenario']

    scenario_map = {
        'opportunity': 0.3,   
        'bottom': 0.3,       
        'normal': 0.3,      
        'caution': 0.3,
        'risk' : 0.3
    }
    dynamic_threshold = scenario_map[primary_scenario]
    
    if not daily_env:
        return

    # -------------------------------------------------------------------------
    # 【买入分支】：未持仓股票（由 before_exec_fn 严格限流）
    # -------------------------------------------------------------------------
    if not position:
        ctx.cancel_all_pending_orders(ctx.symbol)
        top_x_buys = daily_env.get('top_x_buys', set())
        if symbol not in top_x_buys:
            return

        target_size = signal_engine.compute_target_size(
            ctx, daily_env,
            base_target_size=BASE_TARGET_SIZE,
            pos_mult_weight=POS_MULT_WEIGHT,
            pos_mult_bias=POS_MULT_BIAS,
            opport_sizing_coeff=OPPORT_SIZING_COEFF,
            opport_sizing_min=OPPORT_SIZING_MIN,
            opport_sizing_max=OPPORT_SIZING_MAX,
            trained_opport_mag_lgbm=trained_opport_mag_lgbm,
        )

        # 限制 1：逆向计算 T+1 日的最大价格承载力（交集最小值）
        max_price_bias = ctx.close_ma20[-1] * 1.05                     # 防止 T+1 冲高变成加速追涨
        max_price_gap = close * 1.05                     # 防止 T+1 大幅高开
        limit_up = round(close * (1 + rate), 2)
        max_price_limit = limit_up - 0.01               # 防止 T+1 一字涨停强行排队

        ctx.buy_limit_price = min(max_price_bias, max_price_gap, max_price_limit)

        ctx.buy_shares = ctx.calc_target_shares(target_size=target_size, price=close)
        ctx.stop_loss = 3 * atr

    # -------------------------------------------------------------------------
    elif position:
        should_sell, sell_reason = signal_engine.evaluate_sell_signal(
            ctx, daily_env, position,
            trained_risk_mag_lgbm=trained_risk_mag_lgbm,
        )

        if should_sell:
            limit_down = round(close * (1 - rate), 2)
            ctx.sell_limit_price = limit_down + 0.01
            ctx.sell_shares = position.shares

            EXIT_SNAPSHOTS.append({
                'symbol': ctx.symbol,
                'exit_date': ctx.dt,
                'sell_reason': sell_reason,
                'exit_risk_ml_rank': risk_ml_rank,
                'buy_limit_price': getattr(ctx, 'buy_limit_price', 0),
                'sell_limit_price': round(close * (1 - rate), 2)
            })

def run_backtest(symbols, start_date=None, end_date=None, warmup=270, results_dir=None,
                 model_schedule=None, initial_cash=1_000_000):
    # FIX: 防止同进程多次调用时全局状态污染
    global GLOBAL_MARKET_STATS, GLOBAL_SCREEN_THRESHOLDS, _daily_market_cache, BUY_ELIGIBILITY_DETAILS, EXIT_SNAPSHOTS
    GLOBAL_MARKET_STATS = pd.DataFrame()
    GLOBAL_SCREEN_THRESHOLDS = {}
    _daily_market_cache.clear()
    BUY_ELIGIBILITY_DETAILS.clear()
    EXIT_SNAPSHOTS.clear()

    data_source = AKShareChipDataSource()
    
    if start_date is None:
        start_date = datetime(2021, 1, 2)
    if end_date is None:
        end_date = datetime(2026, 8, 7)
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    config = StrategyConfig(
        initial_cash=initial_cash,
        fee_mode=FeeMode.ORDER_PERCENT,
        fee_amount=0.12,    
        enable_fractional_shares=False, 
        position_mode= PositionMode.LONG_ONLY,  
        buy_delay=1,  
        sell_delay=1,  
        exit_on_last_bar=True,  
        bars_per_year=252,  
        bootstrap_sample_size=500
    )
    
    strategy = Strategy(
        data_source,
        start_date=start_date,
        end_date=end_date,
        config=config
    )

    # 滚动分段模型调度：注入给数据源，在 _fetch_data 推理阶段按日期段切换模型
    if model_schedule:
        data_source.model_schedule = model_schedule

    # 注册原生极速 indicators 机制，并完全卸载 pybroker.model，避免回测框架内的高频 I/O 损耗
    strategy.add_execution(chip_strategy, symbols=symbols, indicators=[
        turnover_q90,
        conc_down,
        conc_up,
        # conc70_q_high,
        conc70_q_low,
        # profit_ratio_q5,
        profit_ratio_q50,
        profit_ratio_q20,
        # bias_20_q_over, 
        bias_20_q_high,
        # bias_20_q_low, 
        # adx_q_high,
        # adx_q_low,
        adx_q_over
        # adx_q_pass,
        # chip_penetration_q90
        # 优化后加入的模型及财务加速 Indicator 内存引用
        # ml_score_indicator,
        # is_profit_ok_ind,
        # bp_ratio_ind,
        # roe_up_ind
    ])
    
    strategy.set_before_exec(before_exec_fn)
    
    result = strategy.backtest(
        warmup=warmup,
        disable_parallel=True,
        calc_bootstrap=True
    )  
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if results_dir is None:
        results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/result.txt", "w", encoding='utf-8') as f:
        f.write("=== Metrics ===\n")
        f.write(str(result.metrics_df))
        f.write("\n\n=== Bootstrap Confidence Intervals ===\n")
        f.write(str(result.bootstrap.conf_intervals))
        f.write("\n\n=== Bootstrap Drawdown Confidence ===\n")
        f.write(str(result.bootstrap.drawdown_conf))

    result.trades.to_excel(f'{results_dir}/trades.xlsx', sheet_name='Sheet1', index=False)
    result.orders.to_excel(f'{results_dir}/orders.xlsx', sheet_name='Sheet1', index=False)

    print(f"结果已保存到: {results_dir}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(result.portfolio.index, result.portfolio['market_value'], 'b-', linewidth=2)
    ax1.set_title('Portfolio Market Value', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Market Value', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(result.portfolio.index, result.portfolio['cash'], 'g-', linewidth=2)
    ax2.set_title('Portfolio Cash Balance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cash Amount', fontsize=12)
    ax2.grid(True, alpha=0.3)

    from matplotlib.ticker import FuncFormatter
    def format_currency(x, pos):
        if x >= 1e6:
            return '${:.1f}M'.format(x*1e-6)
        elif x >= 1e3:
            return '${:.1f}K'.format(x*1e-3)
        else:
            return '${:.0f}'.format(x)

    ax1.yaxis.set_major_formatter(FuncFormatter(format_currency))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_currency))

    plt.tight_layout()
    plt.savefig(f'{results_dir}/portfolio_analysis.png', dpi=300, bbox_inches='tight')

    # =========================================================================
    # 终极审计：合并全局大盘指标、筛选阈值与每日策略决策缓存
    # =========================================================================
    try:
        logging.info("正在合并全局审计数据...")

        # 1. 转换 GLOBAL_SCREEN_THRESHOLDS (str key dict -> DataFrame)
        thresholds_df = pd.DataFrame.from_dict(GLOBAL_SCREEN_THRESHOLDS, orient='index')
        thresholds_df.index = pd.to_datetime(thresholds_df.index)
        # 为列名增加前缀避免冲突
        thresholds_df = thresholds_df.add_prefix('thresh_')

        # 2. 转换 _daily_market_cache (Timestamp key dict -> DataFrame)
        # 注意：top_x_buys 是 set 对象，需要转为字符串才能存入 CSV
        cache_list = []
        for dt, vals in _daily_market_cache.items():
            row = vals.copy()
            if 'top_x_buys' in row:
                row['top_x_buys'] = "|".join(list(row['top_x_buys'])) # 转为字符串
            row['date_idx'] = dt
            cache_list.append(row)
        
        cache_df = pd.DataFrame(cache_list)
        if not cache_df.empty:
            cache_df = cache_df.set_index('date_idx').sort_index()
            cache_df = cache_df.add_prefix('strat_')

        # 3. 合并：以 GLOBAL_MARKET_STATS (基础行情统计) 为主表
        # 使用 outer join 确保即便某天没交易也能对齐大盘数据
        audit_merged = GLOBAL_MARKET_STATS.join(thresholds_df, how='outer')
        audit_merged = audit_merged.join(cache_df, how='outer')
        audit_merged = audit_merged.reset_index() 

        # 4. 导出到文件
        audit_merged.to_csv(f"global_strategy_audit.csv")
        
        logging.info(f"终极审计文件已生成: global_strategy_audit.csv")

    except Exception as e:
        logging.error(f"合并审计数据失败: {e}")

    # =========================================================================
    # 导出个股筛选细节审计 (海选明细)
    # =========================================================================

    try:
        logging.info("开始多源数据归因对齐...")
        trades = result.trades.copy()
        
        # --- 1. 强化版统一格式化函数 ---
        def normalize_symbol(s):
            import re
            s_str = str(s).strip()
            # 使用正则表达式提取前/后 6 位数字，兼容 300347, 300347.SZ, sz300347 等各类格式
            match = re.search(r'\d{6}', s_str)
            if match:
                return match.group(0)
            return s_str.split('.')[0][-6:].zfill(6)
        
        def normalize_date(d):
            if pd.isna(d):
                return pd.NaT
            dt = pd.to_datetime(d)
            # 移除时区信息，避免 merge_asof 时因时区不一致报错
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                dt = dt.tz_localize(None)
            return dt.normalize()

        # 格式化交易表键值
        trades['entry_date'] = trades['entry_date'].apply(normalize_date)
        trades['exit_date'] = trades['exit_date'].apply(normalize_date)
        trades['symbol'] = trades['symbol'].apply(normalize_symbol)

        # --- 2. 关联买入快照 (Entry) ---
        if BUY_ELIGIBILITY_DETAILS and len(BUY_ELIGIBILITY_DETAILS) > 0:
            entry_audit = pd.DataFrame(BUY_ELIGIBILITY_DETAILS)
            entry_audit['date'] = entry_audit['date'].apply(normalize_date)
            entry_audit['symbol'] = entry_audit['symbol'].apply(normalize_symbol)
            
            # 清洗并重命名列
            entry_audit = entry_audit[['date', 'symbol', 'ml_rank', 'entry_bias', 'gain_90d']].rename(
                columns={'ml_rank': 'entry_ml_rank', 'date': 'entry_decision_date'}
            )
            entry_audit = entry_audit.dropna(subset=['entry_decision_date', 'symbol'])
            entry_audit = entry_audit.drop_duplicates(subset=['entry_decision_date', 'symbol'], keep='last')
            
            # merge_asof 要求两表必须对排序键（日期）进行升序排序
            trades = trades.sort_values('entry_date')
            entry_audit = entry_audit.sort_values('entry_decision_date')
            
            # 模糊匹配：寻找在 entry_date 当天或之前（backward）最临近的买入决策快照
            # tolerance=pd.Timedelta(days=5) 确保能自动跨越周末或节假日
            trades = pd.merge_asof(
                trades,
                entry_audit,
                left_on='entry_date',
                right_on='entry_decision_date',
                by='symbol',
                direction='backward',
                tolerance=pd.Timedelta(days=5)
            )

        # --- 3. 关联卖出快照 (Exit) ---
        if EXIT_SNAPSHOTS and len(EXIT_SNAPSHOTS) > 0:
            exit_audit = pd.DataFrame(EXIT_SNAPSHOTS)
            exit_audit['exit_date'] = exit_audit['exit_date'].apply(normalize_date)
            exit_audit['symbol'] = exit_audit['symbol'].apply(normalize_symbol)
            
            # 清洗并重命名列
            exit_audit = exit_audit[['exit_date', 'symbol', 'sell_reason', 'exit_risk_ml_rank', 'buy_limit_price', 'sell_limit_price']].rename(
                columns={'exit_date': 'exit_decision_date'}
            )
            exit_audit = exit_audit.dropna(subset=['exit_decision_date', 'symbol'])
            exit_audit = exit_audit.drop_duplicates(subset=['exit_decision_date', 'symbol'], keep='last')
            
            # 排序
            trades = trades.sort_values('exit_date')
            exit_audit = exit_audit.sort_values('exit_decision_date')
            
            # 模糊匹配：寻找在 exit_date 当天或之前最近的卖出决策快照
            trades = pd.merge_asof(
                trades,
                exit_audit,
                left_on='exit_date',
                right_on='exit_decision_date',
                by='symbol',
                direction='backward',
                tolerance=pd.Timedelta(days=5)
            )

        # --- 4. 关联大盘场景 (Scenario) ---
        if _daily_market_cache:
            mkt_list = []
            for dt, val in _daily_market_cache.items():
                mkt_list.append({'date': normalize_date(dt), 'primary_scenario': val['primary_scenario']})
            mkt_df = pd.DataFrame(mkt_list).set_index('date')
            
            # 大盘场景使用执行当天（entry_date）进行精确关联
            trades = pd.merge(trades, mkt_df, left_on='entry_date', right_index=True, how='left')

        # --- 5. 导出 ---
        # 恢复初始交易表的展示顺序（可选，如果您希望根据交易序号或初始顺序展示）
        if 'index' in trades.columns:
            trades = trades.sort_values('index')
            
        trades.to_excel(f'{results_dir}/ultimate_trade_audit.xlsx', index=False)
        logging.info(f"终极归因表已生成: {len(trades)} 笔交易")
        
    except Exception as e:
        logging.error(f"合并个股审计数据失败: {e}")
    # =========================================================================
    # 终极实战：回测截止日（当前最新）市场环境审计
    # =========================================================================
    if _daily_market_cache:
        # 获取最后一个交易日的日期和数据
        last_dt = max(_daily_market_cache.keys())
        last_env = _daily_market_cache[last_dt]
        
        print("\n" + "🚀" * 10 + " 当前市场即时环境报告 (Live Signal) " + "🚀" * 10)
        print(f"信号日期: {last_dt.date()}")
        print(f"大盘场景: {last_env['primary_scenario'].upper()}")
        print(f"是否允许开仓: {'✅ 允许' if last_env['is_market_ok'] else '❌ 禁入 (熔断)'}")
        print(f"决策依据: {last_env['decision_reason']}")
        print("-" * 50)
        print(f"宏观货币信号: {last_env['money_supply_signal']}")
        print(f"当日个股分值门槛 (Threshold): {last_env['daily_ml_threshold']:.4f}")
        print(f"全市场海选入围个股数: {len(last_env.get('top_x_buys', set()))}")
        
        if last_env.get('top_x_buys'):
            print(f"推荐关注标的: {' | '.join(list(last_env['top_x_buys']))}")
        
        print(f"资金分配系数 (Multiplier): {last_env['position_multiplier']}")
        print("🚀" * 35 + "\n")

        recent_dates = sorted(_daily_market_cache.keys())[-5:]
        path = " -> ".join([_daily_market_cache[d]['primary_scenario'] for d in recent_dates])
        print(f"近期场景演变: {path}")

        # 记录到日志文件，方便追踪历史变化
        logging.info(f"LIVE_SIGNAL|{last_dt.date()}|{last_env['primary_scenario']}|{last_env['is_market_ok']}|{last_env['decision_reason']}")

    return results_dir