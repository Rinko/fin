import os
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import talib
import logging
import joblib
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings

# 导入公共算子
try:
    import co_compute 
    # 与生产回测口径对齐：只使用单一 G_pca1_z 大盘特征
    co_compute.FeatureConfig.MKT_FEATURES = ['mkt_macro_regime']
    # 使用预计算单主成分表，避免 co_compute 在 1-component PCA 命名上的兼容性问题
    co_compute.FeatureConfig.PC_TABLE_PATH = '/Volumes/MAC外接/fin_data/explore_night/pca_market_features_20260817/models/g_pca1_z_table.parquet'
except ImportError:
    logging.error("无法加载 co_compute.py，请检查路径")

# 静默警告
warnings.filterwarnings('ignore', category=FutureWarning)

_industry_map_cache = None


def _load_industry_map():
    """惰性加载个股->申万一级行业映射 (读外接盘缓存, 不触发抓取)。缺失时降级为空 dict。"""
    global _industry_map_cache
    if _industry_map_cache is None:
        try:
            from industry_data import load_industry_map
            _industry_map_cache = load_industry_map()
        except Exception as e:
            logging.warning(f"行业映射加载失败, ind_inner_rank 将填中性值: {e}")
            _industry_map_cache = {}
    return _industry_map_cache


def _load_industry_rank_table(mkt_context):
    """按大盘基准计算行业排名表 (行业日K + mkt_ret_20/60)。缺失返回空 DataFrame (特征填中性)。"""
    try:
        from industry_data import load_industry_daily
        ind_daily = load_industry_daily()
        if ind_daily.empty:
            return pd.DataFrame()
        return co_compute.calculate_industry_rank_table(ind_daily, mkt_context)
    except Exception as e:
        logging.warning(f"行业排名表计算失败, ind_rank_20/60 将填中性值: {e}")
        return pd.DataFrame()


_MKT_RET_CACHE = None


def _load_mkt_ret():
    """中证全指日收益（TARGET_EXCESS=1 时用于超额收益 target）。"""
    global _MKT_RET_CACHE
    if _MKT_RET_CACHE is None:
        zz = pd.read_excel('zzqz_df.xlsx').rename(columns={'日期': 'date', '收盘': 'close'})
        zz['date'] = pd.to_datetime(zz['date'])
        _MKT_RET_CACHE = zz.set_index('date')['close'].pct_change()
    return _MKT_RET_CACHE


class AccumulationTrainer:
    def __init__(self, cache_dir='./stock_data_cache'):
        self.cache_dir = cache_dir
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_all_symbols(self):
        # 严格对齐中证全指 (000985) 口径：复用 screen.basic_screen 的统一过滤
        # (剔除 ST/*ST、B股/新股申购前缀；保留历史退市股防生存者偏差)
        from screen import basic_screen
        return basic_screen(cache_dir=self.cache_dir)
    
    def run_global_training(self, start_date, warmup_days, train_end_date, val_end_date,
                            output_pkl='chip_accumulation_v6.pkl', symbols=None, max_stocks=None,
                            feature_contri_overrides=None, skip_audit_csv=False, mkt_contri=1.0,
                            audit_dir='external_data/audit'):
        """
        全量模型训练流水线

        Args:
            output_pkl: 模型导出路径，默认覆盖现役 v6 (滚动重训时应传入独立文件避免破坏静态基准)
            symbols: 可选股票子集 (None=全市场)。用于 feature_gate 快速验证，与生产管线完全同构。
            max_stocks: 可选抽样上限，配合 symbols 使用 (None=全部)。
            feature_contri_overrides: 可选 dict {特征名: 贡献系数}，用于对特定特征降权
                                      (如 {'ema_turnover_vol_z': 0.5} 压低换手率主导地位)。
            skip_audit_csv: 跳过导出 OOS 审计 CSV (滚动 fold 训练用, 每 fold 省 ~15GB 磁盘)
            mkt_contri: 大盘特征的 feature_contri 系数。默认 1.0 (等权，不对大盘特征额外降权)，
                        让模型自主分配市场/个股特征权重。
            audit_dir: 审计 CSV 输出目录。默认 external_data/audit (外接盘)，
                       避免 ~15GB 审计 CSV 撑爆系统盘。
                       存到 pkl 的 data_file 字段为绝对路径，ml_check 据此定位。
        """
        # 1. 准备大盘环境快照 (MarketData)
        # 这里会扫描所有 Network 计算广度、拥挤度，并合并指数环境特征
        context_path = os.path.join(self.cache_dir, 'market_context_cache.parquet')

        # 建议每次训练前同步一次，或者判断文件日期
        logging.info("Step 1: 同步大盘环境快照 (Breadth, Congestion, Mkt Factors)...")
        mkt_context = pd.read_parquet('market_context_cache.parquet')
        mkt_context = mkt_context.set_index('date')
        mkt_context.index = pd.to_datetime(mkt_context.index)

        # 确保 PCA 大盘特征存在（兼容旧 parquet，统一走 co_compute 公共组件）
        pc_cols = list(co_compute.FeatureConfig.MKT_FEATURES)
        if not all(c in mkt_context.columns for c in pc_cols):
            logging.info("PCA 大盘特征缺失，基于原始大盘列重新生成...")
            mkt_context_reset = mkt_context.reset_index()
            mkt_pc = co_compute.build_market_pca_table(mkt_context_reset, min_periods=60)
            mkt_context = mkt_context_reset.merge(mkt_pc, on='date', how='left').set_index('date')
            mkt_context.index = pd.to_datetime(mkt_context.index)

        # 2. 个股特征并行/串行提取
        if symbols is None:
            symbols = self.get_all_symbols()
        if max_stocks is not None:
            symbols = symbols[:max_stocks]
        actual_train_start = (pd.to_datetime(start_date) + timedelta(days=warmup_days)).strftime('%Y-%m-%d')
        
        logging.info(f"Step 2: 提取个股时序特征，目标股票数: {len(symbols)}...")
        all_dfs = []

        from local_data_cache import LocalDataCache
        ldc = LocalDataCache(cache_dir=self.cache_dir)

        for i, s in enumerate(symbols):
            try:
                # 读取前复权数据，与回测保持一致
                df = ldc.get_stock_data(s, start_date, '2060-01-01', adjust="hfq", mode=2)
                if df.empty or len(df) < 150: continue
                df['date'] = pd.to_datetime(df['date'])
                df['vwap'] = ((df['high'] + df['low'] + 2 * df['close']) / 4.0)
                # --- 调用公共算子：计算单股时序特征 ---
                # 传入准备好的 mkt_context，内部会自动处理动态 EMA 窗口
                df = co_compute.compute_individual_indicators(df, mkt_context, use_smooth=True)

                # --- 计算 Target (GPR: Gain-to-Pain Ratio) ---
                # 统一走 co_compute 公共入口，确保所有训练脚本口径一致
                _kw = {'entry_price': os.environ.get('ENTRY_PRICE_MODE', 'close')}
                if os.environ.get('TARGET_EXCESS', '0') == '1':
                    _kw['mkt_ret'] = _load_mkt_ret()
                df = co_compute.compute_entry_target(df, window=20, eps=0.0001, **_kw)

                # 仅保留训练实际开始日期后的数据，减少内存占用
                df = df[df['date'] >= pd.to_datetime(actual_train_start)].copy()
                all_dfs.append(df)
                
                if i % 500 == 0: logging.info(f"已处理 {i} 只股票...")
                
            except Exception as e:
                logging.error(f"处理 {s} 出错: {e}")
                continue

        # 3. 合并全市场数据
        logging.info("Step 3: 合并数据并执行截面标准化...")
        global_data = pd.concat(all_dfs, axis=0).reset_index(drop=True)
        
        # 显式清理中间列表释放内存
        del all_dfs
        import gc
        gc.collect()

        # 3.5 训练宇宙基础过滤（TRAIN_UNIVERSE_FILTER=1 启用）
        # 与回测硬门槛对齐：股价>2、流动性/活跃度(后30%成交额+后20%波动率拦截)、财务盈利。
        # 让模型只在可交易宇宙内学习排序，避免容量浪费在不可行动样本上。
        if os.environ.get('TRAIN_UNIVERSE_FILTER', '0') == '1':
            logging.info("Step 3.5: 训练宇宙基础过滤 (price>2 / liquidity / profit_ok)...")
            n0 = len(global_data)
            global_data = global_data[global_data['close'] > 2]
            # 训练侧无现成 amount_ma20/atr_ratio，按回测同口径现算
            g = global_data.sort_values(['symbol', 'date'])
            grp = g.groupby('symbol', sort=False)
            amt20 = grp['amount'].transform(lambda s: s.rolling(20, min_periods=1).mean())
            pc = grp['close'].shift(1)
            tr = np.maximum(g['high'] - g['low'],
                            np.maximum((g['high'] - pc).abs(), (g['low'] - pc).abs()))
            atr14 = tr.groupby(g['symbol'], sort=False).transform(
                lambda s: s.rolling(14, min_periods=1).mean())
            g = g.assign(_amt20=amt20.values, _atrr=(atr14 / g['close']).values)
            liq_q = g.groupby('date')['_amt20'].quantile(0.30)
            vol_q = g.groupby('date')['_atrr'].quantile(0.20)
            keep = (g['_amt20'] >= g['date'].map(liq_q)) & \
                   (g['_atrr'] >= g['date'].map(vol_q))
            global_data = g[keep].drop(columns=['_amt20', '_atrr'])
            fin_path = 'financial_reports_all.csv'
            if os.path.exists(fin_path):
                fin = pd.read_csv(
                    fin_path,
                    usecols=['股票代码', '报告日期', '净利润-净利润', '净利润-同比增长', '每股收益'],
                    parse_dates=['报告日期'], dtype={'股票代码': 'str'})
                fin = fin.dropna(subset=['报告日期'])
                ok = (fin['净利润-净利润'] > 0) & (fin['净利润-同比增长'] > 0) & (fin['每股收益'] > 0)
                fin = fin.loc[ok, ['股票代码', '报告日期']].copy()
                fin['is_profit_ok'] = True
                fin = fin.sort_values('报告日期').drop_duplicates(
                    subset=['股票代码', '报告日期'], keep='last')
                fin = fin.rename(columns={'股票代码': 'symbol', '报告日期': 'date'})
                gd = global_data.sort_values('date')
                merged = pd.merge_asof(gd, fin, on='date', by='symbol', direction='backward')
                before = len(merged)
                merged = merged[merged['is_profit_ok'].fillna(False)]
                global_data = merged.drop(columns=['is_profit_ok']).reset_index(drop=True)
                logging.info(f"  财务过滤: {before:,} -> {len(global_data):,}")
            else:
                logging.warning("  未找到 financial_reports_all.csv，跳过财务过滤")
            global_data = global_data.reset_index(drop=True)
            logging.info(f"  宇宙过滤完成: {n0:,} -> {len(global_data):,}")

        # 4. 执行截面标准化 (Z-Score)
        # 内部会自动根据 co_compute.FeatureConfig.BIZ_FEATURES 进行处理并计算背离特征
        global_data = co_compute.apply_standardization(
            global_data,
            industry_map=_load_industry_map(),
            ind_rank_table=_load_industry_rank_table(mkt_context),
            features=co_compute.FeatureConfig.BIZ_FEATURES,
        )

        # 5. 映射大盘特征
        # 大盘 PCA 特征已预计算在 mkt_context 中，直接映射到 global_data
        for m_col in co_compute.FeatureConfig.MKT_FEATURES:
            mapped = global_data['date'].map(mkt_context[m_col])
            global_data[m_col] = mapped.ffill()
            if global_data[m_col].isna().any():
                raise ValueError(f"大盘特征 {m_col} 存在无法映射的日期")

        # 6. 目标值高斯化 (Gaussian Rank)
        # 使用 method='average' 处理并列排名, 避免同分股票因 DataFrame 顺序被随机排序引入噪声
        logging.info("Step 4: 目标值高斯化变换...")
        global_data['target'] = global_data.groupby('date')['gpr_target'].transform(
            lambda x: norm.ppf((x.rank(method='average') - 0.5) / (len(x) + 1e-9))
        )
        
        # 7. 模型训练
        final_features = co_compute.FeatureConfig.get_model_input_features()
        logging.info(f"特征构建完毕。总特征数: {len(final_features)}")

        # 动态生成特征惩罚系数 (对齐对大盘特征的抑制)
        feature_contri = []
        for col in final_features:
            if col in co_compute.FeatureConfig.MKT_FEATURES:
                feature_contri.append(mkt_contri) # 抑制大盘因子
            elif feature_contri_overrides and col in feature_contri_overrides:
                feature_contri.append(feature_contri_overrides[col]) # 针对特定特征降权
            else:
                feature_contri.append(1.0)

        model = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.005,
            max_depth=6,
            num_leaves=31,
            min_child_samples=1000,
            reg_alpha=20.0,
            reg_lambda=30.0,
            colsample_bytree=0.4,
            feature_contri=feature_contri,
            deterministic=True,
            force_row_wise=True,
            random_state=42,
            importance_type='gain'
        )

        train_mask = (global_data['date'] <= train_end_date)
        val_mask = (global_data['date'] > train_end_date) & (global_data['date'] <= val_end_date)
        
        # 移除含有 NaN 的训练样本
        clean_data = global_data.dropna(subset=['target'] + final_features)

        model.fit(
            clean_data.loc[train_mask, final_features], clean_data.loc[train_mask, 'target'],
            eval_set=[(clean_data.loc[val_mask, final_features], clean_data.loc[val_mask, 'target'])],
            eval_metric='l2',
            callbacks=[lgb.early_stopping(stopping_rounds=150)]
        )
    
        # 8. 导出
        # 审计数据文件名与模型名绑定，避免多模型训练时互相覆盖（ml_check 审计对齐用）
        audit_data_path = os.path.splitext(output_pkl)[0] + '_data.csv'
        if audit_dir is not None:
            os.makedirs(audit_dir, exist_ok=True)
            audit_data_path = os.path.join(audit_dir, os.path.basename(audit_data_path))
            audit_data_path = os.path.abspath(audit_data_path)
        if not skip_audit_csv:
            clean_data.to_csv(audit_data_path, index=False)
        else:
            audit_data_path = None
        joblib.dump({
            'model': model, 
            'features': final_features,
            'biz_features': co_compute.FeatureConfig.BIZ_FEATURES, # 额外保存原始特征清单以便追溯
            'data_file': audit_data_path # 记录该模型对应的审计数据文件，ml_check 据此避免口径错配
        }, output_pkl)
        logging.info("模型训练并导出成功，v6 架构已实现 100% 逻辑对齐。")


    def run_global_sell_training(self, start_date, warmup_days, train_end_date, val_end_date,
                                 output_pkl='chip_risk_model_v1.pkl', symbols=None, max_stocks=None,
                                 feature_contri_overrides=None, skip_audit_csv=False, mkt_contri=1.0,
                                 audit_dir='external_data/audit'):
        """
        全量模型训练流水线

        Args:
            output_pkl: 模型导出路径，默认覆盖现役风控模型 (滚动重训时应传入独立文件)
            symbols: 可选股票子集 (None=全市场)。用于 feature_gate 快速验证。
            max_stocks: 可选抽样上限，配合 symbols 使用 (None=全部)。
            feature_contri_overrides: 可选 dict {特征名: 贡献系数}，用于对特定特征降权。
            mkt_contri: 大盘特征的 feature_contri 系数。默认 1.0 (等权，不对大盘特征额外降权)，
                        让模型自主分配市场/个股特征权重。
            audit_dir: 审计 CSV 输出目录。默认 external_data/audit (外接盘)，
                       避免 ~15GB 审计 CSV 撑爆系统盘。
        """
        # 1. 准备大盘环境快照 (Market Context)
        # 这一步会扫描所有 DB 计算广度、拥挤度，并合并指数环境因子
        context_path = os.path.join(self.cache_dir, 'market_context_cache.parquet')

        # 建议每次训练前同步一次，或者判断文件日期
        logging.info("Step 1: 同步大盘环境快照 (Breadth, Congestion, Mkt Factors)...")
        mkt_context = pd.read_parquet('market_context_cache.parquet')
        mkt_context = mkt_context.set_index('date')
        mkt_context.index = pd.to_datetime(mkt_context.index)

        # 确保 PCA 大盘特征存在（兼容旧 parquet，统一走 co_compute 公共组件）
        pc_cols = list(co_compute.FeatureConfig.MKT_FEATURES)
        if not all(c in mkt_context.columns for c in pc_cols):
            logging.info("PCA 大盘特征缺失，基于原始大盘列重新生成...")
            mkt_context_reset = mkt_context.reset_index()
            mkt_pc = co_compute.build_market_pca_table(mkt_context_reset, min_periods=60)
            mkt_context = mkt_context_reset.merge(mkt_pc, on='date', how='left').set_index('date')
            mkt_context.index = pd.to_datetime(mkt_context.index)

        # 2. 个股特征并行/串行提取
        if symbols is None:
            symbols = self.get_all_symbols()
        if max_stocks is not None:
            symbols = symbols[:max_stocks]
        actual_train_start = (pd.to_datetime(start_date) + timedelta(days=warmup_days)).strftime('%Y-%m-%d')
        
        logging.info(f"Step 2: 提取个股时序特征，目标股票数: {len(symbols)}...")
        all_dfs = []

        from local_data_cache import LocalDataCache
        ldc = LocalDataCache(cache_dir=self.cache_dir)

        for i, s in enumerate(symbols):
            try:
                # 读取前复权数据，与回测保持一致
                df = ldc.get_stock_data(s, start_date, '2060-01-01', adjust="hfq", mode=2)
                if df.empty or len(df) < 150: continue
                df['date'] = pd.to_datetime(df['date'])
                df['vwap'] = ((df['high'] + df['low'] + 2 * df['close']) / 4.0)
                # --- 调用公共算子：计算单股时序特征 ---
                # 传入准备好的 mkt_context，内部会自动处理动态 EMA 窗口
                df = co_compute.compute_individual_indicators(df, mkt_context, use_smooth=False)

                # 风险 Target：统一走 co_compute 公共入口
                df = co_compute.compute_risk_target(df, hold_window=5)

                # 仅保留训练实际开始日期后的数据，减少内存占用
                df = df[df['date'] >= pd.to_datetime(actual_train_start)].copy()
                all_dfs.append(df)
                
                if i % 500 == 0: logging.info(f"已处理 {i} 只股票...")
                
            except Exception as e:
                logging.error(f"处理 {s} 出错: {e}")
                continue

        # 3. 合并全市场数据
        logging.info("Step 3: 合并数据并执行截面标准化...")
        global_data = pd.concat(all_dfs, axis=0).reset_index(drop=True)
        
        # 显式清理中间列表释放内存
        del all_dfs
        import gc
        gc.collect()

        # 4. 执行截面标准化 (Z-Score)
        global_data = co_compute.apply_standardization(
            global_data,
            industry_map=_load_industry_map(),
            ind_rank_table=_load_industry_rank_table(mkt_context),
            features=co_compute.FeatureConfig.BIZ_RISK_FEATURES,
        )

        # 5. 映射大盘特征
        # 大盘 PCA 特征已预计算在 mkt_context 中，直接映射到 global_data
        for m_col in co_compute.FeatureConfig.MKT_FEATURES:
            mapped = global_data['date'].map(mkt_context[m_col])
            global_data[m_col] = mapped.ffill()
            if global_data[m_col].isna().any():
                raise ValueError(f"大盘特征 {m_col} 存在无法映射的日期")

        # 6. 目标值高斯化 (Gaussian Rank)
        # 使用 method='average' 处理并列排名, 避免大量无风险股票因 DataFrame 顺序被随机排序引入噪声
        logging.info("Step 4: 目标值高斯化变换...")
        global_data['target'] = global_data.groupby('date')['risk_score'].transform(
            lambda x: norm.ppf((x.rank(method='average', ascending=False) - 0.5) / (len(x) + 1e-9))
        )

        # 7. 模型训练
        final_features = co_compute.FeatureConfig.get_risk_model_input_features()
        logging.info(f"特征构建完毕。总特征数: {len(final_features)}")

        # 动态生成特征惩罚系数 (对齐对大盘特征的抑制)
        feature_contri = []
        for col in final_features:
            if col in co_compute.FeatureConfig.MKT_FEATURES:
                feature_contri.append(mkt_contri) # 抑制大盘因子
            elif feature_contri_overrides and col in feature_contri_overrides:
                feature_contri.append(feature_contri_overrides[col]) # 针对特定特征降权
            else:
                feature_contri.append(1.0)

        risk_model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01, # 稍微快一点
            max_depth=5,
            reg_alpha=10.0,
            reg_lambda=10.0,
            colsample_bytree=0.6,
            feature_contri=feature_contri,
            importance_type='gain'
        )

        train_mask = (global_data['date'] <= train_end_date)
        val_mask = (global_data['date'] > train_end_date) & (global_data['date'] <= val_end_date)
        
        # 移除含有 NaN 的训练样本
        clean_data = global_data.dropna(subset=['target'] + final_features)

        risk_model.fit(
            clean_data.loc[train_mask, final_features], clean_data.loc[train_mask, 'target'],
            eval_set=[(clean_data.loc[val_mask, final_features], clean_data.loc[val_mask, 'target'])],
            eval_metric='l2',
            callbacks=[lgb.early_stopping(stopping_rounds=150)]
        )
    
        # 8. 导出
        # 审计数据文件名与模型名绑定，避免多模型训练时互相覆盖（ml_check 审计对齐用）
        audit_data_path = os.path.splitext(output_pkl)[0] + '_data.csv'
        if audit_dir is not None:
            os.makedirs(audit_dir, exist_ok=True)
            audit_data_path = os.path.join(audit_dir, os.path.basename(audit_data_path))
            audit_data_path = os.path.abspath(audit_data_path)
        if not skip_audit_csv:
            clean_data.to_csv(audit_data_path, index=False)
        else:
            audit_data_path = None
        joblib.dump({
            'model': risk_model, 
            'features': final_features,
            'biz_features': co_compute.FeatureConfig.BIZ_RISK_FEATURES, # 额外保存原始特征清单以便追溯
            'data_file': audit_data_path # 记录该模型对应的审计数据文件，ml_check 据此避免口径错配
        }, output_pkl)
        logging.info("模型风险训练并导出成功")

if __name__ == "__main__":
    trainer = AccumulationTrainer()
    trainer.run_global_training(
        start_date='2012-03-12',
        warmup_days=400,
        train_end_date='2019-12-31',
        val_end_date='2020-12-31',
        output_pkl=os.environ.get('TRAIN_OUTPUT_PKL', 'chip_accumulation_v6_open_entry.pkl'),
        skip_audit_csv=os.environ.get('SKIP_AUDIT_CSV', '0') == '1'
    )
    # 风控模型重训会覆盖生产 chip_risk_model_v1.pkl（经软链接），仅在显式要求时执行
    if os.environ.get('TRAIN_SELL', '0') == '1':
        trainer.run_global_sell_training(
            start_date='2012-03-12',
            warmup_days=400,
            train_end_date='2019-12-31',
            val_end_date='2020-12-31',
            output_pkl=os.environ.get('TRAIN_SELL_OUTPUT_PKL', 'chip_risk_model_v1.pkl'),
            skip_audit_csv=os.environ.get('SKIP_AUDIT_CSV', '0') == '1'
        )