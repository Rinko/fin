import sqlite3
import os
import pandas as pd
from datetime import datetime

class LocalDataCache:
    def __init__(self, symbol_fetcher=None, code_fetcher=None, cache_dir='./stock_data_cache'):
        """
        升级版数据缓存层：
        1. 线上数据拉取强制保存为不复权
        2. 全局主库维护因子表
        3. 读取时利用 pd.merge_asof 动态复权并剥离 raw_ 前缀
        """
        self.symbol_fetcher = symbol_fetcher
        self.code_fetcher = code_fetcher
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, 'stock_data.db')
        
        # 1. 初始化元数据表与集中因子表
        self._init_main_db()

        self._code_map = {}
        # 自动迁移检查（向个股库中追加 raw_ 字段）
        self._migrate_existing_cache_files()

    def _init_main_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_meta (
                    symbol TEXT PRIMARY KEY,
                    code TEXT,
                    last_updated TEXT,
                    min_date TEXT,
                    max_date TEXT
                )
            ''')
            cursor.execute("PRAGMA table_info(cache_meta)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'code' not in columns:
                cursor.execute("ALTER TABLE cache_meta ADD COLUMN code TEXT")
                
            # 建立主库全局因子存储表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS adjust_factors (
                    code TEXT,
                    dividOperateDate TEXT,
                    foreAdjustFactor REAL,
                    backAdjustFactor REAL,
                    adjustFactor REAL,
                    PRIMARY KEY (code, dividOperateDate)
                )
            ''')
            conn.commit()

    def _resolve_ids(self, stock_id):
        stock_id = str(stock_id).strip()
        if '.' in stock_id:
            code = stock_id
            symbol = stock_id.split('.')[1]
        else:
            symbol = stock_id.zfill(6)
            code = self._get_code_from_symbol(symbol)
        return symbol, code

    def _get_code_from_symbol(self, symbol):
        symbol_str = str(symbol).strip().zfill(6)
        if symbol_str in self._code_map:
            return self._code_map[symbol_str]
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT code FROM cache_meta WHERE symbol = ? AND code IS NOT NULL", (symbol_str,))
                row = cursor.fetchone()
                if row and row[0]:
                    self._code_map[symbol_str] = row[0]
                    return row[0]
        except Exception:
            pass

        if symbol_str.startswith(('60', '68', '90', '73', '78', '50', '51', '52', '58', '110', '113', '118')):
            code = f"sh.{symbol_str}"
        elif symbol_str.startswith(('00', '30', '20', '39', '15', '16', '18', '123', '127', '128')):
            code = f"sz.{symbol_str}"
        elif symbol_str.startswith(('43', '83', '87', '88', '92')):
            code = f"bj.{symbol_str}"
        else:
            code = f"sh.{symbol_str}"
        self._code_map[symbol_str] = code
        return code

    def _get_cached_date_range(self, symbol):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT min_date, max_date FROM cache_meta WHERE symbol = ?", (symbol,))
                row = cursor.fetchone()
                if row and row[0] and row[1]:
                    return row[0], row[1]
        except Exception:
            pass
        return None, None

    def _check_coverage(self, symbol, s_dt, e_dt):
        min_date, max_date = self._get_cached_date_range(symbol)
        if min_date and max_date:
            return s_dt >= min_date and e_dt <= max_date
        return False

    def _get_symbol_db_path(self, symbol):
        return os.path.join(self.cache_dir, f'{symbol}.db')

    def _migrate_existing_cache_files(self):
        if not os.path.exists(self.cache_dir):
            return
        for file in os.listdir(self.cache_dir):
            if file.endswith('.db') and file != 'stock_data.db':
                symbol = file.replace('.db', '')
                db_path = self._get_symbol_db_path(symbol)
                try:
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_data'")
                        if not cursor.fetchone():
                            continue
                        cursor.execute("PRAGMA table_info(stock_data)")
                        existing_cols = [col[1] for col in cursor.fetchall()]
                        
                        raw_cols_to_add = {
                            'raw_open': 'REAL', 'raw_high': 'REAL', 'raw_low': 'REAL',
                            'raw_close': 'REAL', 'raw_preclose': 'REAL', 'raw_volume': 'REAL', 'raw_amount': 'REAL'
                        }
                        mutated = False
                        for col_name, col_type in raw_cols_to_add.items():
                            if col_name not in existing_cols:
                                cursor.execute(f"ALTER TABLE stock_data ADD COLUMN {col_name} {col_type}")
                                mutated = True
                        if mutated:
                            # 历史存量数据填充：如果数据库中只有 open/close，将其先默认赋给 raw_ 列
                            for col_name in raw_cols_to_add.keys():
                                origin_name = col_name.replace('raw_', '')
                                if origin_name in existing_cols:
                                    cursor.execute(f"UPDATE stock_data SET {col_name} = {origin_name} WHERE {col_name} IS NULL")
                            if 'raw_preclose' in raw_cols_to_add and 'close' in existing_cols:
                                cursor.execute("UPDATE stock_data SET raw_preclose = close WHERE raw_preclose IS NULL")
                            conn.commit()
                except Exception as e:
                    print(f"老版文件 [{file}] 迁移升级失败: {e}")

    def _get_from_cache(self, symbol, code, start_date, end_date):
        db_path = self._get_symbol_db_path(symbol)
        if not os.path.exists(db_path):
            return pd.DataFrame()
        with sqlite3.connect(db_path) as conn:
            query = f"SELECT * FROM stock_data WHERE date BETWEEN '{start_date}' AND '{end_date} 23:59:59' ORDER BY date"
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df['symbol'] = symbol
                df['code'] = code
        return df

    def _get_last_unadjusted_close_from_db(self, symbol):
        """
        获取本地最新一天的历史不复权收盘价（用于除权检测）
        """
        db_path = self._get_symbol_db_path(symbol)
        if not os.path.exists(db_path):
            return None
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT raw_close FROM stock_data WHERE raw_close IS NOT NULL ORDER BY date DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    return float(row[0])
        except Exception:
            pass
        return None

    def sync_adjust_factors_from_api(self, code):
        """
        从 API 同步并追加历史复权因子数据
        """
        if self.code_fetcher is None:
            return
        print(f"[{code}] 因子缺失或检测到除权，开始从 API 同步完整历史复权因子...")
        # 默认同步 1990 年至今
        df_factors = self.code_fetcher.query_adjust_factor(code, "1990-01-01", datetime.now().strftime("%Y-%m-%d"))
        if df_factors is not None and not df_factors.empty:
            df_factors['dividOperateDate'] = pd.to_datetime(df_factors['dividOperateDate']).dt.strftime('%Y-%m-%d')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for _, row in df_factors.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO adjust_factors 
                        (code, dividOperateDate, foreAdjustFactor, backAdjustFactor, adjustFactor)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        row['code'], row['dividOperateDate'], 
                        float(row['foreAdjustFactor']), float(row['backAdjustFactor']), float(row['adjustFactor'])
                    ))
                conn.commit()
                print(f"[{code}] 因子同步成功，已写入 {len(df_factors)} 条因子记录。")

    def _get_adjust_factors_from_db(self, code):
        """
        从本地主库中读取历史复权因子
        """
        query = "SELECT dividOperateDate, foreAdjustFactor, backAdjustFactor FROM adjust_factors WHERE code = ? ORDER BY dividOperateDate ASC"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(code,))
        return df

    def _store_to_cache(self, symbol, code, df, s_dt=None, e_dt=None):
        if df is None or df.empty: return
        
        # 🌟 写入前的物理除权检测
        if 'raw_preclose' in df.columns:
            try:
                last_raw_close = self._get_last_unadjusted_close_from_db(symbol)
                if last_raw_close is not None:
                    df_sorted = df.sort_values('date')
                    first_row = df_sorted.iloc[0]
                    first_preclose = float(first_row['raw_preclose'])
                    
                    if abs(first_preclose - last_raw_close) > 0.005:
                        print(f"⚠️ [除权检测拦截] {code} 本地历史昨收: {last_raw_close}, 今日昨收: {first_preclose}。检测到历史发生除权！")
                        self.sync_adjust_factors_from_api(code)
            except Exception as e:
                print(f"除权监测异常: {e}")

        db_path = self._get_symbol_db_path(symbol)
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
            df['symbol'] = symbol
            df['code'] = code
            
            df.to_sql('stock_data', conn, if_exists='replace', index=False)

            min_date = df['date'].min()
            max_date = df['date'].max()

            if s_dt is not None:
                min_date = min(min_date, s_dt)
            if e_dt is not None:
                max_date = max(max_date, e_dt)

        today = datetime.now().strftime('%Y-%m-%d')
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT OR REPLACE INTO cache_meta (symbol, code, last_updated, min_date, max_date) VALUES (?, ?, ?, ?, ?)', 
                           (symbol, code, today, min_date, max_date))
            conn.commit()

    # ==================== [核心实现：1.0 内存向量化复权计算与双轨对比验证] ====================
    def _apply_dynamic_adjust(self, df_raw, symbol, code, adjust):
        """
        利用 pandas 向量化计算 QFQ/HFQ，计算完毕后移除 `raw_` 前缀列
        """
        if df_raw.empty:
            return df_raw
            
        df = df_raw.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. 纯不复权直接剥离 raw_
        if adjust in ["none", "3"]:
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if f'raw_{col}' in df.columns:
                    df[col] = df[f'raw_{col}']
            if 'raw_preclose' in df.columns:
                df['preclose'] = df['raw_preclose']
            df['change'] = df['close'] - df['preclose']
            df['change_pct'] = (df['change'] / df['preclose'] * 100).round(4)
            df['amplitude'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(4)
            
            # 移除所有不复权的前缀物理列，防止对外泄漏
            raw_cols = [c for c in df.columns if c.startswith('raw_')]
            return df.drop(columns=raw_cols)

        # 2. 读取复权因子表
        df_factors = self._get_adjust_factors_from_db(code)
        
        # 🌟 Lazy-load：因子缺失时自动向 API 调取
        if df_factors.empty and adjust in ["qfq", "hfq"]:
            self.sync_adjust_factors_from_api(code)
            df_factors = self._get_adjust_factors_from_db(code)

        # 若仍无因子记录（从未除权过或拉取失败），直接按不复权处理
        if df_factors.empty:
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if f'raw_{col}' in df.columns:
                    df[col] = df[f'raw_{col}']
            if 'raw_preclose' in df.columns:
                df['preclose'] = df['raw_preclose']
            df['change'] = df['close'] - df['preclose']
            df['change_pct'] = (df['change'] / df['preclose'] * 100).round(4)
            df['amplitude'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(4)
            raw_cols = [c for c in df.columns if c.startswith('raw_')]
            return df.drop(columns=raw_cols)

        df_factors['dividOperateDate'] = pd.to_datetime(df_factors['dividOperateDate'])
        
        # 向量化对齐前，必须确保日期升序
        df = df.sort_values('date')
        df_factors = df_factors.sort_values('dividOperateDate')

        # 利用 merge_asof 对齐最近除权因子
        merged = pd.merge_asof(
            df,
            df_factors,
            left_on='date',
            right_on='dividOperateDate',
            direction='backward'
        )

        # 根据复权要求选择对应列和填充策略
        if adjust == "qfq":
            # QFQ 填充策略：最早除权日之前的历史数据填充为“最早的那个前复权因子 (iloc[0])”
            earliest_factor = df_factors['foreAdjustFactor'].iloc[0]
            merged['factor'] = merged['foreAdjustFactor'].fillna(earliest_factor)
        else: # hfq
            # HFQ 填充策略：早于最早除权日的数据，因子填充为 1.0 (未除权前状态)
            merged['factor'] = merged['backAdjustFactor'].fillna(1.0)

        # 最终安全兜底
        merged['factor'] = merged['factor'].fillna(1.0)

        # 动态计算 OHLCV 和 preclose (成交量 volume 必须除以因子以保持乘积等价)
        merged['open'] = (merged['raw_open'] * merged['factor']).round(4)
        merged['high'] = (merged['raw_high'] * merged['factor']).round(4)
        merged['low'] = (merged['raw_low'] * merged['factor']).round(4)
        merged['close'] = (merged['raw_close'] * merged['factor']).round(4)
        merged['preclose'] = (merged['raw_preclose'] * merged['factor']).round(4)
        merged['volume'] = (merged['raw_volume'] / merged['factor']).round(2)
        merged['amount'] = merged['raw_amount']

        # 重新计算技术指标
        merged['change'] = (merged['close'] - merged['preclose']).round(4)
        merged['change_pct'] = (merged['change'] / merged['preclose'] * 100).round(4)
        merged['amplitude'] = ((merged['high'] - merged['low']) / merged['preclose'] * 100).round(4)

        # =============================================================
        # 🧪 【临时双轨对比校验代码】
        # 对比动态计算出来的 QFQ 收盘价与数据库里原有的 QFQ 真实拉取值，证明 iloc[0] 修复的正确性
        # 确认完全一致后，可将本段 if 块全部安全删除
        # =============================================================
        if adjust == "qfq" and 'close' in df_raw.columns:
            # 库原存 close（拉自 BaoStock 官方 QFQ K线数据）
            orig_close_sorted = df_raw.sort_values('date').reset_index(drop=True)['close']
            calc_close = merged['close']
            diff = (orig_close_sorted - calc_close).abs()
            max_diff = diff.max()
            if pd.notna(max_diff):
                print(f"🧪 [临时校验] 股票 {symbol} 动态 QFQ 算法最大绝对偏离度: {max_diff:.6f}")
                if max_diff > 0.01:
                    print(f"⚠️ 校验提示：在以下行发现精度或逻辑偏差：")
                    bad_indices = diff[diff > 0.01].index
                    for idx in bad_indices[:5]:
                        row = merged.iloc[idx]
                        print(f"  日期: {row['date'].strftime('%Y-%m-%d')} | 库中原值: {orig_close_sorted.iloc[idx]:.4f} | 动态计算: {row['close']:.4f} | 因子: {row['factor']:.6f}")
            else:
                print(f"🧪 [临时校验] {symbol} 暂无对比数据。")
        # =============================================================

        # 清除 raw_ 系列列以及多余临时合并字段
        drop_cols = ['dividOperateDate', 'foreAdjustFactor', 'backAdjustFactor', 'factor'] + [c for c in merged.columns if c.startswith('raw_')]
        merged = merged.drop(columns=[col for col in drop_cols if col in merged.columns])
        
        # 格式化日期返回
        merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')
        return merged

    def get_stock_data(self, stock_id, start_date, end_date, adjust="qfq", mode=1, online=None):
        if online is not None:
            mode = 1 if online else 2

        symbol, code = self._resolve_ids(stock_id)
        s_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        e_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        # ==================== [MODE 2: 纯本地模式] ====================
        if mode == 2:
            min_date, max_date = self._get_cached_date_range(symbol)
            if not min_date or not max_date:
                print(f"[{symbol} / {code}] 提示: 本地暂无任何缓存数据。")
            else:
                missing_info = []
                if s_dt < min_date:
                    missing_info.append(f"前段缺失 [{s_dt} 至 {min_date}]")
                if e_dt > max_date:
                    missing_info.append(f"后段缺失 [{max_date} 至 {e_dt}]")
                if missing_info:
                    print(f"[{symbol} / {code}] 提示: 本地数据未完全覆盖。")
            
            df_raw = self._get_from_cache(symbol, code, s_dt, e_dt)
            return self._apply_dynamic_adjust(df_raw, symbol, code, adjust)

        # ==================== [MODE 1: 智能混合模式] ====================
        if mode == 1:
            is_covered = self._check_coverage(symbol, s_dt, e_dt)
            if is_covered:
                print(f"[{symbol} / {code}] 本地数据覆盖请求区间，直接动态复权并返回...")
                df_raw = self._get_from_cache(symbol, code, s_dt, e_dt)
                return self._apply_dynamic_adjust(df_raw, symbol, code, adjust)
            else:
                print(f"[{symbol} / {code}] 本地未完全覆盖，从线上同步不复权底稿数据...")

        # ==================== [线上拉取流程 (MODE 1 未覆盖 或 MODE 3)] ====================
        df_api = None
        
        # 优先使用 code_fetcher 🌟 强制拉取 "none" (不复权) 以保存增量原始数据
        if self.code_fetcher is not None:
            try:
                df_api = self.code_fetcher.fetch_single_stock(code, s_dt, e_dt, adjust="none")
            except Exception as e:
                print(f"[{code}] 通过 code_fetcher 拉取线上数据失败: {e}")
                
        # 降级使用 symbol_fetcher
        if (df_api is None or df_api.empty) and self.symbol_fetcher is not None:
            try:
                df_api = self.symbol_fetcher.fetch_single_stock(symbol, s_dt, e_dt, adjust="none")
            except Exception as e:
                print(f"[{symbol}] 通过 symbol_fetcher 拉取线上数据失败: {e}")
        
        if df_api is None or df_api.empty:
            print(f"[{symbol}] 线上拉取失败，降级使用本地已有数据。")
            df_raw = self._get_from_cache(symbol, code, s_dt, e_dt)
            return self._apply_dynamic_adjust(df_raw, symbol, code, adjust)

        # 合流
        df_api['date'] = pd.to_datetime(df_api['date'], format='mixed').dt.strftime('%Y-%m-%d')
        df_db = self._get_from_cache(symbol, code, '1900-01-01', '2100-01-01')
        
        if not df_db.empty:
            df_db['date'] = pd.to_datetime(df_db['date'], format='mixed').dt.strftime('%Y-%m-%d')
            
            # 补齐字段
            all_cols = list(df_api.columns)
            for col in all_cols:
                if col not in df_db.columns:
                    df_db[col] = pd.NA
                    
            df_api['symbol'] = symbol
            df_api['code'] = code
            df_db['symbol'] = symbol
            df_db['code'] = code

            # 合并去重（信任最新的 API 数据）
            df_final = pd.concat([df_db, df_api]).drop_duplicates(subset=['date'], keep='last')
        else:
            df_final = df_api.copy()
            df_final['symbol'] = symbol
            df_final['code'] = code

        df_final = df_final.sort_values('date').reset_index(drop=True)

        try:
            self._store_to_cache(symbol, code, df_final, s_dt, e_dt)
        except Exception as e:
            print(f"写入本地数据库失败 [{symbol} / {code}]: {e}")

        # 切片，截取用户所需区间，并应用动态复权计算
        mask = (df_final['date'] >= s_dt) & (df_final['date'] <= e_dt)
        df_raw_slice = df_final.loc[mask].copy()
        
        return self._apply_dynamic_adjust(df_raw_slice, symbol, code, adjust)

def update_daily_market_data(self, date_str=None):
        """
        🚀 全市场每日数据一键极速增量更新
        :param date_str: 格式为 'YYYY-MM-DD'。如果为 None，则自动使用今天日期。
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
            
        print(f"\n==================== [开始全市场每日增量同步: {date_str}] ====================")
        
        # 1. 获取今日全市场复权因子变更名单 (只需 1 次 API 请求)
        print("步骤 [1/4]: 正在获取今日复权因子变更数据...")
        df_factors_today = self.code_fetcher.query_daily_adjust_factor(date_str)
        
        ex_dividend_codes = set()
        if df_factors_today is not None and not df_factors_today.empty:
            ex_dividend_codes = set(df_factors_today['code'].tolist())
            print(f"👉 今日全市场共有 {len(ex_dividend_codes)} 只股票发生除权除息。")
        else:
            print("👉 今日全市场无除权除息事件。")
            
        # 2. 获取今日全市场 A股 日K线数据 (只需 1 次 API 请求)
        print("步骤 [2/4]: 正在获取今日全市场股票日K线数据...")
        df_astock = self.code_fetcher.query_daily_history_k_AStock(date_str)
        
        # 3. 获取今日全市场 ETF 日K线数据 (只需 1 次 API 请求)
        print("步骤 [3/4]: 正在获取今日全市场 ETF 日K线数据...")
        df_etf = self.code_fetcher.query_daily_history_k_ETF(date_str)
        
        # 合并K线
        df_kline_today = pd.DataFrame()
        if df_astock is not None and not df_astock.empty:
            df_kline_today = pd.concat([df_kline_today, df_astock])
        if df_etf is not None and not df_etf.empty:
            df_kline_today = pd.concat([df_kline_today, df_etf])
            
        if df_kline_today.empty:
            print(f"❌ {date_str} 未获取到任何交易K线数据（可能是非交易日或数据未发布）。更新终止。")
            return
            
        # 过滤停牌或非正常交易日（tradestatus == '1' 代表正常交易）
        if 'tradestatus' in df_kline_today.columns:
            df_kline_today = df_kline_today[df_kline_today['tradestatus'] == '1'].copy()
            
        if df_kline_today.empty:
            print(f"⚠️ {date_str} 全市场无正常交易标的（全部停牌或非交易日）。")
            return
            
        print(f"步骤 [4/4]: 准备增量写入 {len(df_kline_today)} 只标的的今日数据...")
        
        # 确保日期格式规范
        df_kline_today['date'] = pd.to_datetime(df_kline_today['date']).dt.strftime('%Y-%m-%d')
        
        chip_cols = [
            'profit_ratio', 'avg_cost', 'cost_90_low', 'cost_90_high', 
            'concentration_90', 'cost_70_low', 'cost_70_high', 'concentration_70'
        ]
        
        success_count = 0
        new_stock_count = 0
        
        # 逐个标的增量写入
        for _, row in df_kline_today.iterrows():
            code = row['code']
            symbol = code.split('.')[1] if '.' in code else code
            db_path = self._get_symbol_db_path(symbol)
            
            # (A) 判断是否是新股/新标的（本地无历史数据库文件）
            if not os.path.exists(db_path):
                print(f"🆕 发现新上市或新加入标的: {code}，正在进行历史数据初始化拉取...")
                try:
                    # 强拉 2010 年至今的历史不复权数据完成建库
                    self.get_stock_data(code, "2010-01-01", date_str, mode=3)
                    new_stock_count += 1
                    success_count += 1
                except Exception as e:
                    print(f"初始化新标的 [{code}] 失败: {e}")
                continue
                
            # (B) 如果是今天除权的股票，优先下载并覆写其历史因子（全市场仅对除权个股做此操作）
            if code in ex_dividend_codes:
                try:
                    self.sync_adjust_factors_from_api(code)
                except Exception as e:
                    print(f"更新除权股票 [{code}] 因子失败: {e}")
            
            # (C) 转换该行数据并写入
            try:
                row_dict = {}
                row_dict['date'] = date_str
                row_dict['code'] = code
                row_dict['symbol'] = symbol
                
                # 转换数值类型
                for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount']:
                    row_dict[col] = float(row[col]) if row[col] else None
                    # 双轨不复权字段
                    row_dict[f'raw_{col}'] = row_dict[col]
                    
                row_dict['turnover'] = float(row['turn']) if row['turn'] else None
                row_dict['change_pct'] = float(row['pctChg']) if row['pctChg'] else None
                
                # 重新计算 amplitude 和 change
                if row_dict['preclose'] and row_dict['preclose'] > 0:
                    row_dict['amplitude'] = round((row_dict['high'] - row_dict['low']) / row_dict['preclose'] * 100, 4)
                    row_dict['change'] = round(row_dict['close'] - row_dict['preclose'], 4)
                else:
                    row_dict['amplitude'] = 0.0
                    row_dict['change'] = 0.0
                    
                # 筹码字段置空
                for col in chip_cols:
                    row_dict[col] = None
                    
                # 安全写入：先清除可能存在的今日旧数据，防止脚本重复运行产生重复行，然后 append 插入
                with sqlite3.connect(db_path) as conn:
                    conn.execute("DELETE FROM stock_data WHERE date = ?", (date_str,))
                    
                    cols = list(row_dict.keys())
                    placeholders = ', '.join(['?'] * len(cols))
                    sql = f"INSERT INTO stock_data ({', '.join(cols)}) VALUES ({placeholders})"
                    conn.execute(sql, tuple(row_dict[c] for c in cols))
                    
                # (D) 同步更新 cache_meta 表中的 max_date
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE cache_meta SET max_date = ?, last_updated = ? WHERE symbol = ?", 
                                   (date_str, datetime.now().strftime('%Y-%m-%d'), symbol))
                    conn.commit()
                    
                success_count += 1
            except Exception as e:
                # 打印单只股票的错误，但不要阻断整个循环
                print(f"写入 [{code}] 每日增量失败: {e}")
                
        print(f"\n==================== [每日增量同步完成] ====================")
        print(f"🎯 成功同步标的: {success_count} 只 | 自动建库新股: {new_stock_count} 只 | 同步日期: {date_str}")