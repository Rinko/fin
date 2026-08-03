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
except ImportError:
    logging.error("无法加载 co_compute.py，请检查路径")

# 静默警告
warnings.filterwarnings('ignore', category=FutureWarning)


class AccumulationTrainer:
    def __init__(self, cache_dir='./stock_data_cache'):
        self.cache_dir = cache_dir
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_all_symbols(self):
        # 严格过滤：文件名必须是数字或大写字母开头，排除 metadata 库和 journaling 文件
        return [f[:-3] for f in os.listdir(self.cache_dir) 
                if f.endswith('.db') and not f.startswith('stock_data') and '-' not in f]
    
    # def prepare_global_factors(self, index_file='zzqz_df.xlsx'):
        """计算大盘 5 维环境因子 + 全市场广度/拥挤度"""
        logging.info("Step 1: 计算大盘 5 维度因子 (Trend, Vol, Liq, Bias, Pos)...")
        self.mkt_factors = co_compute.calculate_global_mkt_factors(index_file)
        
        logging.info("Step 2: 采样个股数据计算大盘广度与拥挤度...")
        all_basic_data = []
        potential_dbs = self.get_all_symbols()
        
        
        for s in potential_dbs:
            db_path = os.path.join(self.cache_dir, f"{s}.db")
            try:
                with sqlite3.connect(db_path) as conn:
                    # 检查表是否存在
                    check_table = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_data'").fetchone()
                    if check_table:
                        tmp = pd.read_sql("SELECT date, symbol, close, amount FROM stock_data", conn)
                        all_basic_data.append(tmp)
            except Exception:
                continue # 跳过异常数据库

        if not all_basic_data:
            raise ValueError("未能从缓存目录读取到任何有效股票数据，请检查缓存路径。")
            
        full_market = pd.concat(all_basic_data)
        full_market['date'] = pd.to_datetime(full_market['date'])
        
        # 调用公共脚本：计算 20日新高占比、拥挤度等
        self.mkt_breadth = co_compute.calculate_high_low_stats(full_market).set_index('date')
        logging.info(f"大盘因子准备完毕，共计 {len(self.mkt_breadth)} 个交易日。")

    # def process_single_stock(self, symbol, data_start_date, train_actual_start):
        db_path = os.path.join(self.cache_dir, f"{symbol}.db")
        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql(f"SELECT * FROM stock_data WHERE date >= '{data_start_date}' ORDER BY date", conn)
            
            if len(df) < 150: return None

            # --- 基础清洗 ---
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open','high','low','close','amount','volume','turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['vwap'] = ((df['high'] + df['low'] + 2 * df['close']) / 4.0)

            df = co_compute.compute_individual_indicators(df, self.mkt_factors, use_smooth=True)


            # --- 合并全维度大盘特征 ---
            df = df.merge(self.mkt_factors, left_on='date', right_index=True, how='left')
            df = df.merge(self.mkt_breadth[['up_ratio', 'congestion', 'congestion_bias', 'high20_ratio', 'low20_ratio']], 
                         left_on='date', right_index=True, how='left')
            breadth_cols = ['up_ratio', 'congestion', 'congestion_bias', 'high20_ratio', 'low20_ratio']
            df[breadth_cols] = df[breadth_cols].ffill().fillna(0.0)
            df = df.ffill().fillna(0.5)

            # --- Numba 核心算子 ---
            c, v, h, l, t = df['close'].values, df['vwap'].values, df['high'].values, df['low'].values, df['turnover'].values
            chips = co_compute.calculate_chip_metrics_numba(c, v, h, l, t)
            p_pos, p_neg = co_compute.calculate_chip_penetration_numba(c, chips[0], t)

            # --- 特征工程：残差化逻辑 (EMA + Residual) ---
            vol_adj = (1.5 - df['mkt_vol'].values).clip(0.6, 1.4)
            w_slow, w_mid, w_fast = self.cfg.base_ema_slow * vol_adj, self.cfg.base_ema_mid * vol_adj, self.cfg.base_ema_fast * vol_adj

            # 1. 筹码获利盘与单峰形态
            df['ema_profit'] = co_compute.dynamic_ema(chips[0], w_slow)
            df['res_profit'] = chips[0] - df['ema_profit']
            
            # 【新增新特征 A】：筹码单峰峰值高度 (Peak Density) -> 衡量筹码最重仓价格段锁定力度
            df['ema_peak_density'] = co_compute.dynamic_ema(chips[8], w_slow)  # chips[8] 在 co_compute 中输出
            
            # 2. 筹码集中度 (90% & 70%) 残差与变动速度
            df['ema_conc_90'] = co_compute.dynamic_ema(chips[4], w_mid)
            df['res_conc_90'] = chips[4] - df['ema_conc_90']
            df['ema_conc_70'] = co_compute.dynamic_ema(chips[7], w_mid)
            df['res_conc_70'] = chips[7] - df['ema_conc_70']

            # 【新增新特征 B】 筹码集中度 90% 的时序收缩速度 (Velocity of Concentration)
            # 通过 np.clip 防御任何因数据微小不对齐产生的瞬时脉冲噪声
            conc_90_velocity = pd.Series(chips[4]).pct_change(5).fillna(0).values.astype(np.float64)
            conc_90_velocity = np.clip(conc_90_velocity, -0.20, 0.20)
            df['ema_conc_90_v'] = co_compute.dynamic_ema(conc_90_velocity, w_mid)
            
            # 3. 价格相对于筹码峰的位置 (支撑/压力)
            df['dist_to_avg'] = (c / (chips[1] + 1e-9)) - 1.0
            df['dist_to_low90'] = (c / (chips[2] + 1e-9)) - 1.0
            df['dist_to_high90'] = (c / (chips[3] + 1e-9)) - 1.0

            # 4. 穿透力残差
            df['ema_penetrate_up'] = co_compute.dynamic_ema(p_pos, w_slow)
            df['res_penetrate_up'] = p_pos - df['ema_penetrate_up']
            df['ema_decay_dn'] = co_compute.dynamic_ema(p_neg, w_slow)

            # 5. 核心：乖离率残差
            bias_raw = (c / (talib.SMA(c, 20) + 1e-9)) - 1.0
            bias_std = pd.Series(bias_raw).rolling(60).std().fillna(0.02).values
            bias_norm = (bias_raw / (bias_std + 1e-9)).astype(np.float64)

            df['ema_bias_norm'] = co_compute.dynamic_ema(bias_norm, w_fast)
            df['res_bias_norm'] = bias_norm - df['ema_bias_norm']

            # 6. 筹码重心位移速度
            avg_cost_velocity = pd.Series(chips[1]).pct_change(5).fillna(0).values.astype(np.float64)
            # 【时序防御机制】无除权日历下，直接对时序变化率进行极值 clipping，消除除权尾差跳空影响
            avg_cost_velocity = np.clip(avg_cost_velocity, -0.15, 0.15)
            df['ema_cost_v'] = co_compute.dynamic_ema(avg_cost_velocity, w_mid)

            # 7. 换手率分布熵与间歇放量特征
            t_mean_20 = pd.Series(t).rolling(20).mean().values
            turnover_rel = (t / (t_mean_20 + 1e-9)).astype(np.float64)
            df['ema_turnover_vol'] = co_compute.dynamic_ema(turnover_rel, w_slow)

            # 【新增新特征 C】：间歇式放量最大残差值（反映吸筹期脉冲大买）
            t_mean_60 = pd.Series(t).rolling(60).mean().values
            t_ratio_60 = (t / (t_mean_60 + 1e-9)).astype(np.float64)
            t_max_res = pd.Series(t_ratio_60).rolling(20).max().fillna(1.0).values
            df['ema_turnover_max_res'] = co_compute.dynamic_ema(t_max_res, w_slow)

            # 8. 量能平稳性残差与量价背离特征
            vol_stab = pd.Series(t).rolling(20).std() / (pd.Series(t).rolling(20).mean() + 1e-9)
            df['ema_vol_stab'] = co_compute.dynamic_ema(vol_stab.fillna(0).values, w_mid)
            df['res_vol_stab'] = vol_stab.values - df['ema_vol_stab']
            
            # 【新增新特征 D】：20日时序量价相关性（背离度）
            c_pct = pd.Series(c).pct_change().fillna(0)
            v_pct = pd.Series(df['volume'].values).pct_change().fillna(0)
            vp_corr = c_pct.rolling(20).corr(v_pct).fillna(0.0).values.astype(np.float64)
            df['ema_vp_corr'] = co_compute.dynamic_ema(vp_corr, w_slow)

            # --- 目标值：风险调整后收益 ---
            # ret_f20 = pd.Series(c).pct_change(20).shift(-20)
            # f_low = pd.Series(l).rolling(20).min().shift(-20)
            # drawdown = (pd.Series(c) - f_low) / (pd.Series(c) + 1e-9)
            # # 【优化】放宽最大回撤容忍至 10%（A股极易触碰5%），同时减轻惩罚系数至 4.0，保留黑马妖股样本
            # df['target_val'] = ret_f20 / (1.0 + np.power(np.maximum(0, drawdown - 0.10), 1.2) * 4.0)

            # =====================================================================
            # 优化点：前瞻索提诺比率 (Forward Sortino Ratio)
            # 无任何主观硬编码参数，不惩罚向上爆发的波动，仅惩罚向下破位的下行风险
            # =====================================================================
            # daily_ret = pd.Series(c).pct_change(1)
            
            # # 1. 均值收益：未来 20 日的每日对数/百分比收益率均值
            # f_mean = daily_ret.rolling(20).mean().shift(-20)
            
            # # 2. 下行偏差：计算未来 20 日中，每日收益率为负数时的均方根 (RMS)
            # # 将正收益全部 clip 归零，仅保留负收益平方，然后滚动求均值再开方
            # downside_sq = daily_ret.clip(upper=0.0) ** 2
            # f_downside_std = np.sqrt(downside_sq.rolling(20).mean().shift(-20))
            
            # # =====================================================================
            # # 优化：训练目标（Sortino）与评估收益（Raw Return）完美解耦
            # # =====================================================================
            # # 1. 真实收益率（用于回测、审计 PnL 和夏普计算）
            # ret_f20 = pd.Series(c).pct_change(20).shift(-20)
            # df['target_val'] = ret_f20 # 必须保持为真实百分比收益率！

            # # 2. 索提诺比率（仅用于生成模型的训练 Target）
            # daily_ret = pd.Series(c).pct_change(1)
            # f_mean = daily_ret.rolling(20).mean().shift(-20)
            
            # downside_sq = daily_ret.clip(upper=0.0) ** 2
            # # 引入 0.001 (0.1%日波动) 作为贝叶斯先验，防止单边无下跌牛股的分母爆炸
            # f_downside_std = np.sqrt(downside_sq.rolling(20).mean().shift(-20))
            
            # df['sortino_target'] = f_mean / (f_downside_std + 0.001)
            # =====================================================================
            # 【修改点 1】: 替换 Target 逻辑为 Gain-to-Pain Ratio (GPR)
            # =====================================================================
            daily_ret = pd.Series(c).pct_change(1)
        
            # 1. 记录原始收益率 (用于审计 PnL，不直接参与训练)
            df['target_val'] = pd.Series(c).pct_change(self.cfg.target_long_window).shift(-self.cfg.target_long_window)

            # 2. 计算未来 20 日的正收益之和 与 负收益绝对值之和
            # 我们使用 clip 来分离正负收益
            pos_rets = daily_ret.clip(lower=0)
            neg_rets = daily_ret.clip(upper=0).abs()
            
            # 滚动求和并向前移动窗口 (look-forward)
            window = self.cfg.target_long_window
            f_pos_sum = pos_rets.rolling(window).sum().shift(-window)
            f_neg_sum = neg_rets.rolling(window).sum().shift(-window)
            
            # 计算 GPR: 分母加入 0.0001 (0.01%波动) 仅作为数学防御，无业务主观含义
            # 它完美捕捉了“上行厚度”与“下行痛感”的比值
            df['gpr_target'] = f_pos_sum / (f_neg_sum + 0.0001)

            # 过滤无效行
            df = df[df['date'] >= pd.to_datetime(train_actual_start)].copy()
            return df.dropna(subset=['target_val', 'gpr_target'])
            # return df.dropna(subset=['target_val', 'sortino_target'])
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
            return None

    def run_global_training(self, start_date, warmup_days, train_end_date, val_end_date):
        """
        全量模型训练流水线
        """
        # 1. 准备大盘环境快照 (Market Context)
        # 这一步会扫描所有 DB 计算广度、拥挤度，并合并指数环境因子
        context_path = os.path.join(self.cache_dir, 'market_context_cache.parquet')
        
        # 建议每次训练前同步一次，或者判断文件日期
        logging.info("Step 1: 同步大盘环境快照 (Breadth, Congestion, Mkt Factors)...")
        mkt_context = pd.read_parquet('market_context_cache.parquet')
        mkt_context = mkt_context.set_index('date')
        mkt_context.index = pd.to_datetime(mkt_context.index)

        # 2. 个股特征并行/串行提取
        symbols = self.get_all_symbols()
        actual_train_start = (pd.to_datetime(start_date) + timedelta(days=warmup_days)).strftime('%Y-%m-%d')
        
        logging.info(f"Step 2: 提取个股时序特征，目标股票数: {len(symbols)}...")
        all_dfs = []
        
        for i, s in enumerate(symbols):
            try:
                # 读取原始数据
                db_path = os.path.join(self.cache_dir, f"{s}.db")
                with sqlite3.connect(db_path) as conn:
                    df = pd.read_sql(f"SELECT * FROM stock_data WHERE date >= '{start_date}'", conn)
                
                if len(df) < 150: continue
                df['date'] = pd.to_datetime(df['date']) 
                df['vwap'] = ((df['high'] + df['low'] + 2 * df['close']) / 4.0)
                # --- 调用公共算子：计算单股时序特征 ---
                # 传入准备好的 mkt_context，内部会自动处理动态 EMA 窗口
                df = co_compute.compute_individual_indicators(df, mkt_context, use_smooth=True)
                
                # --- 计算 Target (GPR: Gain-to-Pain Ratio) ---
                # 注意：Target 必须在截面标准化之前算好
                daily_ret = df['close'].pct_change(1)
                window = 20
                f_pos_sum = daily_ret.clip(lower=0).rolling(window).sum().shift(-window-1)
                f_neg_sum = daily_ret.clip(upper=0).abs().rolling(window).sum().shift(-window-1)
                df['gpr_target'] = f_pos_sum / (f_neg_sum + 0.0001)
                
                # 保存用于审计的真实收益
                df['target_val'] = df['close'].pct_change(window).shift(-window-1)

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
        # 内部会自动根据 co_compute.FeatureConfig.BIZ_FEATURES 进行处理并计算背离特征
        global_data = co_compute.apply_standardization(global_data)
        
        # 5. 映射大盘特征
        # 之前 compute_individual_indicators 只是用了 mkt_vol 算窗口，
        # 现在需要把大盘因子正式作为模型输入列映射进来
        for m_col in co_compute.FeatureConfig.MKT_FEATURES:
            global_data[m_col] = global_data['date'].map(mkt_context[m_col]).ffill().fillna(0.5)

        # 6. 目标值高斯化 (Gaussian Rank)
        logging.info("Step 4: 目标值高斯化变换...")
        global_data['target'] = global_data.groupby('date')['gpr_target'].transform(
            lambda x: norm.ppf((x.rank(method='first') - 0.5) / (len(x) + 1e-9))
        )
        
        # 7. 模型训练
        final_features = co_compute.FeatureConfig.get_model_input_features()
        logging.info(f"特征构建完毕。总特征数: {len(final_features)}")

        # 动态生成特征惩罚系数 (对齐对大盘特征的抑制)
        feature_contri = []
        for col in final_features:
            if col in co_compute.FeatureConfig.MKT_FEATURES:
                feature_contri.append(0.45) # 抑制大盘因子
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
        joblib.dump({
            'model': model, 
            'features': final_features,
            'biz_features': co_compute.FeatureConfig.BIZ_FEATURES # 额外保存原始特征清单以便追溯
        }, 'chip_accumulation_v6.pkl')
        
        # 保存一份样例数据用于 ml_check 审计
        clean_data.to_csv('model_data.csv', index=False)
        logging.info("模型训练并导出成功，v6 架构已实现 100% 逻辑对齐。")


    def run_global_sell_training(self, start_date, warmup_days, train_end_date, val_end_date):
        """
        全量模型训练流水线
        """
        # 1. 准备大盘环境快照 (Market Context)
        # 这一步会扫描所有 DB 计算广度、拥挤度，并合并指数环境因子
        context_path = os.path.join(self.cache_dir, 'market_context_cache.parquet')
        
        # 建议每次训练前同步一次，或者判断文件日期
        logging.info("Step 1: 同步大盘环境快照 (Breadth, Congestion, Mkt Factors)...")
        mkt_context = pd.read_parquet('market_context_cache.parquet')
        mkt_context = mkt_context.set_index('date')
        mkt_context.index = pd.to_datetime(mkt_context.index)

        # 2. 个股特征并行/串行提取
        symbols = self.get_all_symbols()
        actual_train_start = (pd.to_datetime(start_date) + timedelta(days=warmup_days)).strftime('%Y-%m-%d')
        
        logging.info(f"Step 2: 提取个股时序特征，目标股票数: {len(symbols)}...")
        all_dfs = []
        
        for i, s in enumerate(symbols):
            try:
                # 读取原始数据
                db_path = os.path.join(self.cache_dir, f"{s}.db")
                with sqlite3.connect(db_path) as conn:
                    df = pd.read_sql(f"SELECT * FROM stock_data WHERE date >= '{start_date}'", conn)
                
                if len(df) < 150: continue
                df['date'] = pd.to_datetime(df['date']) 
                df['vwap'] = ((df['high'] + df['low'] + 2 * df['close']) / 4.0)
                # --- 调用公共算子：计算单股时序特征 ---
                # 传入准备好的 mkt_context，内部会自动处理动态 EMA 窗口
                df = co_compute.compute_individual_indicators(df, mkt_context, use_smooth=False)
                
                daily_ret = df['close'].pct_change(1)
                sigma = daily_ret.rolling(20).std()
                # 风险 Target：从 T+1 后的价格波动中寻找最低点
                f_max_loss = daily_ret.rolling(5).min().shift(-6) 

                # 增加分母防御，并对最终得分进行截断
                # 我们认为超过 5 倍标准差的下跌统一视为“极端崩溃”
                raw_risk = f_max_loss / (sigma + 0.005) # 增加 epsilon 到 0.5% 波动
                df['risk_score'] = raw_risk.clip(-5.0, 0.0)

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
        # 内部会自动根据 co_compute.FeatureConfig.BIZ_FEATURES 进行处理并计算背离特征
        global_data = co_compute.apply_standardization(global_data)
        
        # 5. 映射大盘特征
        # 之前 compute_individual_indicators 只是用了 mkt_vol 算窗口，
        # 现在需要把大盘因子正式作为模型输入列映射进来
        for m_col in co_compute.FeatureConfig.MKT_FEATURES:
            global_data[m_col] = global_data['date'].map(mkt_context[m_col]).ffill().fillna(0.5)

        # 6. 目标值高斯化 (Gaussian Rank)
        logging.info("Step 4: 目标值高斯化变换...")
        global_data['target'] = global_data.groupby('date')['risk_score'].transform(
            lambda x: norm.ppf((x.rank(method='first', ascending=False) - 0.5) / (len(x) + 1e-9))
        )
        
        # 7. 模型训练
        final_features = co_compute.FeatureConfig.get_risk_model_input_features()
        logging.info(f"特征构建完毕。总特征数: {len(final_features)}")

        # 动态生成特征惩罚系数 (对齐对大盘特征的抑制)
        feature_contri = []
        for col in final_features:
            if col in co_compute.FeatureConfig.MKT_FEATURES:
                feature_contri.append(0.45) # 抑制大盘因子
            else:
                feature_contri.append(1.0)

        risk_model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01, # 稍微快一点
            max_depth=5,
            reg_alpha=10.0,
            reg_lambda=10.0,
            colsample_bytree=0.6,
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
        joblib.dump({
            'model': risk_model, 
            'features': final_features,
            'biz_features': co_compute.FeatureConfig.BIZ_RISK_FEATURES # 额外保存原始特征清单以便追溯
        }, 'chip_risk_model_v1.pkl')
        
        # 保存一份样例数据用于 ml_check 审计
        clean_data.to_csv('model_risk_data.csv', index=False)
        logging.info("模型风险训练并导出成功")

if __name__ == "__main__":
    trainer = AccumulationTrainer()
    trainer.run_global_training(
        start_date='2012-03-12', 
        warmup_days=400, 
        train_end_date='2019-12-31', 
        val_end_date='2020-12-31'
    )
    # trainer.run_global_sell_training(
    #     start_date='2012-03-12', 
    #     warmup_days=400, 
    #     train_end_date='2019-12-31', 
    #     val_end_date='2020-12-31'
    # )