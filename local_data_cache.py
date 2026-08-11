import sqlite3
import os
import logging
import pandas as pd
from datetime import datetime

class LocalDataCache:
    def __init__(self, symbol_fetcher=None, code_fetcher=None, cache_dir='./stock_data_cache'):
        self.symbol_fetcher = symbol_fetcher
        self.code_fetcher = code_fetcher
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, 'stock_data.db')
        
        # 1. 初始化元数据表与集中因子表
        self._init_main_db()

        self._code_map = {}
        # 自动迁移检查
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
                    max_date TEXT,
                    ipo_date TEXT,
                    out_date TEXT
                )
            ''')
            # 兼容表结构升级
            cursor.execute("PRAGMA table_info(cache_meta)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'code' not in columns:
                cursor.execute("ALTER TABLE cache_meta ADD COLUMN code TEXT")
            if 'ipo_date' not in columns:
                cursor.execute("ALTER TABLE cache_meta ADD COLUMN ipo_date TEXT")
            if 'out_date' not in columns:
                cursor.execute("ALTER TABLE cache_meta ADD COLUMN out_date TEXT")
            if 'code_name' not in columns:
                cursor.execute("ALTER TABLE cache_meta ADD COLUMN code_name TEXT")
                
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
                            for col_name in raw_cols_to_add.keys():
                                origin_name = col_name.replace('raw_', '')
                                if origin_name in existing_cols:
                                    cursor.execute(f"UPDATE stock_data SET {col_name} = {origin_name} WHERE {col_name} IS NULL")
                            if 'raw_preclose' in raw_cols_to_add and 'close' in existing_cols:
                                cursor.execute("UPDATE stock_data SET raw_preclose = close WHERE raw_preclose IS NULL")
                            conn.commit()
                except Exception as e:
                    print(f"老版文件 [{file}] 迁移升级失败: {e}")

    # 🌟 修正 1：防止 SQL 注入，将 _get_from_cache 重构为标准的参数化查询
    def _get_from_cache(self, symbol, code, start_date, end_date):
        db_path = self._get_symbol_db_path(symbol)
        if not os.path.exists(db_path):
            return pd.DataFrame()
        with sqlite3.connect(db_path) as conn:
            query = "SELECT * FROM stock_data WHERE date BETWEEN ? AND ? ORDER BY date"
            df = pd.read_sql_query(query, conn, params=(start_date, f"{end_date} 23:59:59"))
            if not df.empty:
                df['symbol'] = symbol
                df['code'] = code
        return df

    def _get_last_unadjusted_close_from_db(self, symbol):
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
        if self.code_fetcher is None:
            return
        print(f"[{code}] 开始同步完整历史复权因子...")
        df_factors = self.code_fetcher.query_adjust_factor(code, "1990-01-01", datetime.now().strftime("%Y-%m-%d"))
        
        # 🌟 核心改进：网络连接成功且返回不为 None 时才写入（排除网络超时导致的误判）
        if df_factors is not None:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if not df_factors.empty:
                    # (A) 正常写入真实的除权除息因子
                    df_factors['dividOperateDate'] = pd.to_datetime(df_factors['dividOperateDate']).dt.strftime('%Y-%m-%d')
                    for _, row in df_factors.iterrows():
                        cursor.execute('''
                            INSERT OR REPLACE INTO adjust_factors 
                            (code, dividOperateDate, foreAdjustFactor, backAdjustFactor, adjustFactor)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            row['code'], row['dividOperateDate'], 
                            float(row['foreAdjustFactor']), float(row['backAdjustFactor']), float(row['adjustFactor'])
                        ))
                else:
                    # (B) 🌟 成功获取但数据为空（说明该股从未发生过除权除息）
                    # 写入 1900-01-01 的 1.0 虚拟占位因子。既标记了“同步完毕”，又彻底阻断了后续回测无尽的 lazy-load 循环！
                    cursor.execute('''
                        INSERT OR REPLACE INTO adjust_factors 
                        (code, dividOperateDate, foreAdjustFactor, backAdjustFactor, adjustFactor)
                        VALUES (?, '1900-01-01', 1.0, 1.0, 1.0)
                    ''', (code,))
                conn.commit()

    def _get_adjust_factors_from_db(self, code):
        query = "SELECT dividOperateDate, foreAdjustFactor, backAdjustFactor FROM adjust_factors WHERE code = ? ORDER BY dividOperateDate ASC"
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(code,))
        return df

    def _store_to_cache(self, symbol, code, df, s_dt=None, e_dt=None):
        if df is None or df.empty: return
        
        df_sorted = df.sort_values('date').copy()

        # 精准向量化检测：检查拉回来的整段 df 内部是否存在任何除权点
        need_sync_factors = False
        if 'raw_preclose' in df_sorted.columns:
            try:
                # 1. 检查新拉回数据内部的相邻行是否发生除权
                yesterday_close = df_sorted['raw_close'].shift(1)
                mismatch_series = (df_sorted['raw_preclose'] - yesterday_close).abs() > 0.005
                mismatch_series.iloc[0] = False # 排除首行
                
                # 2. 检查与本地数据库“连接处”的物理变化
                last_raw_close = self._get_last_unadjusted_close_from_db(symbol)
                connection_mismatch = False
                if last_raw_close is not None:
                    first_preclose = float(df_sorted['raw_preclose'].iloc[0])
                    connection_mismatch = abs(first_preclose - last_raw_close) > 0.005
                
                if mismatch_series.any() or connection_mismatch:
                    need_sync_factors = True
                    
            except Exception as e:
                print(f"向量化除权判定异常: {e}")

        if need_sync_factors:
            print(f"⚠️ [除权检测拦截] {code} 扫描到历史未同步的除权事件，触发因子同步...")
            self.sync_adjust_factors_from_api(code)

        db_path = self._get_symbol_db_path(symbol)
        
        # 🌟 核心：过滤出我们要写入 SQLite 的列（仅限 meta 映射列 + raw_ 物理列）
        raw_cols_to_write = [
            'date', 'symbol', 'code', 
            'raw_open', 'raw_high', 'raw_low', 'raw_close', 'raw_preclose', 
            'raw_volume', 'raw_amount', 'raw_change_pct', 'raw_turnover'
        ]
        # 只保留这些列，不向 SQLite 写入任何没有前缀的重复列和空占位列
        df_sorted_raw_only = df_sorted[[c for c in raw_cols_to_write if c in df_sorted.columns]].copy()

        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            df_sorted_raw_only['date'] = pd.to_datetime(df_sorted_raw_only['date'], format='mixed').dt.strftime('%Y-%m-%d')
            df_sorted_raw_only['symbol'] = symbol
            df_sorted_raw_only['code'] = code
            
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_data'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                dates_list = df_sorted_raw_only['date'].tolist()
                if len(dates_list) == 1:
                    conn.execute("DELETE FROM stock_data WHERE date = ?", (dates_list[0],))
                elif len(dates_list) > 1:
                    placeholders = ', '.join(['?'] * len(dates_list))
                    conn.execute(f"DELETE FROM stock_data WHERE date IN ({placeholders})", tuple(dates_list))
            
            df_sorted_raw_only.to_sql('stock_data', conn, if_exists='append', index=False)

            min_date = df_sorted_raw_only['date'].min()
            max_date = df_sorted_raw_only['date'].max()

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
    def _get_ipo_out_dates(self, symbol):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ipo_date, out_date FROM cache_meta WHERE symbol = ?", (symbol,))
                row = cursor.fetchone()
                if row:
                    return row[0], row[1]
        except Exception:
            pass
        return None, None

    def _apply_dynamic_adjust(self, df_raw, symbol, code, adjust):
        if df_raw.empty:
            return df_raw
            
        df = df_raw.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. 纯不复权：直接将 raw_ 物理列拷贝给无前缀列返回
        if adjust in ["none", "3"]:
            df['open'] = df['raw_open']
            df['high'] = df['raw_high']
            df['low'] = df['raw_low']
            df['close'] = df['raw_close']
            df['preclose'] = df['raw_preclose']
            df['volume'] = df['raw_volume']
            df['amount'] = df['raw_amount']
            df['change_pct'] = df['raw_change_pct']
            df['turnover'] = df['raw_turnover']
            
            df['change'] = df['close'] - df['preclose']
            df['amplitude'] = ((df['high'] - df['low']) / df['preclose'].replace(0, float('nan')) * 100).round(4)
            
            raw_cols = [c for c in df.columns if c.startswith('raw_')]
            return df.drop(columns=raw_cols)

        # 2. 读取复权因子表
        df_factors = self._get_adjust_factors_from_db(code)
        
        if df_factors.empty and adjust in ["qfq", "hfq"]:
            self.sync_adjust_factors_from_api(code)
            df_factors = self._get_adjust_factors_from_db(code)

        # 若无因子（未除权过），等同于不复权
        if df_factors.empty:
            df['open'] = df['raw_open']
            df['high'] = df['raw_high']
            df['low'] = df['raw_low']
            df['close'] = df['raw_close']
            df['preclose'] = df['raw_preclose']
            df['volume'] = df['raw_volume']
            df['amount'] = df['raw_amount']
            df['change_pct'] = df['raw_change_pct']
            df['turnover'] = df['raw_turnover']
            
            df['change'] = df['close'] - df['preclose']
            df['amplitude'] = ((df['high'] - df['low']) / df['preclose'].replace(0, float('nan')) * 100).round(4)
            
            raw_cols = [c for c in df.columns if c.startswith('raw_')]
            return df.drop(columns=raw_cols)

        df_factors['dividOperateDate'] = pd.to_datetime(df_factors['dividOperateDate'])
        
        df = df.sort_values('date')
        df_factors = df_factors.sort_values('dividOperateDate')

        merged = pd.merge_asof(
            df,
            df_factors,
            left_on='date',
            right_on='dividOperateDate',
            direction='backward'
        )

        if adjust == "qfq":
            merged['factor'] = merged['foreAdjustFactor'].bfill()
            latest_factor = df_factors['foreAdjustFactor'].iloc[-1] if not df_factors.empty else 1.0
            if latest_factor != 0 and latest_factor != 1.0:
                merged['factor'] = merged['factor'] / latest_factor
        else: # hfq
            merged['factor'] = merged['backAdjustFactor'].fillna(1.0)

        merged['factor'] = merged['factor'].fillna(1.0)

        # 3. 动态复权换算
        merged['open'] = (merged['raw_open'] * merged['factor']).round(4)
        merged['high'] = (merged['raw_high'] * merged['factor']).round(4)
        merged['low'] = (merged['raw_low'] * merged['factor']).round(4)
        merged['close'] = (merged['raw_close'] * merged['factor']).round(4)
        merged['preclose'] = (merged['raw_preclose'] * merged['factor']).round(4)
        merged['volume'] = (merged['raw_volume'] / merged['factor']).round(2)
        merged['amount'] = merged['raw_amount']
        
        # 复权重算 change_pct：基于复权后的 close 和 preclose 计算
        merged['change_pct'] = ((merged['close'] / merged['preclose'].replace(0, float('nan'))) - 1) * 100
        merged['turnover'] = merged['raw_turnover']

        merged['change'] = (merged['close'] - merged['preclose']).round(4)
        merged['amplitude'] = ((merged['high'] - merged['low']) / merged['preclose'].replace(0, float('nan')) * 100).round(4)

        # 移除 raw_ 系列列以及多余临时对齐列
        drop_cols = ['dividOperateDate', 'foreAdjustFactor', 'backAdjustFactor', 'factor'] + [c for c in merged.columns if c.startswith('raw_')]
        merged = merged.drop(columns=[col for col in drop_cols if col in merged.columns])
        
        merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')
        return merged

    def get_stock_meta(self):
        """通过统一入口返回股票元数据(名称/生命周期)，供筛股与训练侧使用

        所有访问 stock_data.db / cache_meta 的行为都应经由此方法，
        禁止外部直接 sqlite3.connect 到缓存数据库。
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    "SELECT symbol, code, code_name, ipo_date, out_date, last_updated "
                    "FROM cache_meta", conn)
            return df
        except Exception as e:
            logging.error(f"读取股票元数据失败: {e}")
            return pd.DataFrame()

    def update_cache_meta_code(self):
        if self.code_fetcher is None:
            print("❌ 错误: 未配置 code_fetcher")
            return

        print(f"[{datetime.now()}] 开始同步全历史股票资料（含退市股）...")
        df = self.code_fetcher.query_stock_basic()

        if df is None or df.empty:
            print("❌ 未获取到任何有效的证券数据。")
            return

        df_stocks = df[df['type'] == '1'].copy()
        if df_stocks.empty:
            print("⚠️ 未筛选出有效的股票数据。")
            return

        df_stocks['symbol'] = df_stocks['code'].apply(lambda x: x.split('.')[1] if '.' in x else x)
        write_data = df_stocks[['symbol', 'code', 'ipoDate', 'outDate', 'code_name']].values.tolist()


        print(f"正在写入 {len(write_data)} 只股票生命周期映射至 {self.db_path}...")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                upsert_query = '''
                    INSERT INTO cache_meta (symbol, code, ipo_date, out_date, code_name)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET 
                        code = excluded.code,
                        ipo_date = excluded.ipo_date,
                        out_date = excluded.out_date,
                        code_name = excluded.code_name
                '''
                cursor.executemany(upsert_query, write_data)
                conn.commit()
                print(f"[{datetime.now()}] 全量股票生命周期表映射成功。")
        except Exception as e:
            print(f"❌ 写入 cache_meta 失败: {e}")

    # 🌟 修正 5：在接口层显式申明对 mode=3 (强制网络模式) 的分流支持
    def get_stock_data(self, stock_id, start_date, end_date, adjust="qfq", mode=1, online=None):
        if online is not None:
            mode = 1 if online else 2

        symbol, code = self._resolve_ids(stock_id)
        s_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        e_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        ipo_date, out_date = self._get_ipo_out_dates(symbol)
        
        effective_s_dt = s_dt
        effective_e_dt = e_dt
        
        if ipo_date and ipo_date != "":
            effective_s_dt = max(s_dt, ipo_date)
        if out_date and out_date != "":
            effective_e_dt = min(e_dt, out_date)

        if effective_s_dt > effective_e_dt:
            return pd.DataFrame()

        # ==================== [MODE 2: 纯本地模式] ====================
        if mode == 2:
            df_raw = self._get_from_cache(symbol, code, effective_s_dt, effective_e_dt)
            return self._apply_dynamic_adjust(df_raw, symbol, code, adjust)

        # ==================== [MODE 1: 智能混合模式] ====================
        if mode == 1:
            is_covered = self._check_coverage(symbol, effective_s_dt, effective_e_dt)
            if is_covered:
                df_raw = self._get_from_cache(symbol, code, effective_s_dt, effective_e_dt)
                return self._apply_dynamic_adjust(df_raw, symbol, code, adjust)
            else:
                print(f"[{symbol} / {code}] 本地数据未完全覆盖实际交易区间，从网络增量同步...")

        # ==================== [MODE 3: 强制网络模式 (显式声明分支)] ====================
        if mode == 3:
            print(f"[{symbol} / {code}] 强制网络同步启动...")

        # ==================== [线上数据拉取与合流逻辑] ====================
        df_api = None
        if self.code_fetcher is not None:
            try:
                df_api = self.code_fetcher.fetch_single_stock(code, effective_s_dt, effective_e_dt, adjust="none")
            except Exception as e:
                print(f"[{code}] fetch 失败: {e}")
                
        if (df_api is None or df_api.empty) and self.symbol_fetcher is not None:
            try:
                df_api = self.symbol_fetcher.fetch_single_stock(symbol, effective_s_dt, effective_e_dt, adjust="none")
            except Exception as e:
                print(f"[{symbol}] fetch 失败: {e}")
        
        if df_api is None or df_api.empty:
            df_raw = self._get_from_cache(symbol, code, effective_s_dt, effective_e_dt)
            return self._apply_dynamic_adjust(df_raw, symbol, code, adjust)

        df_api['date'] = pd.to_datetime(df_api['date'], format='mixed').dt.strftime('%Y-%m-%d')
        df_db = self._get_from_cache(symbol, code, '1900-01-01', '2100-01-01')
        
        if not df_db.empty:
            df_db['date'] = pd.to_datetime(df_db['date'], format='mixed').dt.strftime('%Y-%m-%d')
            all_cols = list(df_api.columns)
            for col in all_cols:
                if col not in df_db.columns:
                    df_db[col] = pd.NA
                    
            df_api['symbol'] = symbol
            df_api['code'] = code
            df_db['symbol'] = symbol
            df_db['code'] = code

            dfs = [df for df in [df_db, df_api] if not df.empty]
            if len(dfs) > 1:
                df_final = pd.concat(dfs).drop_duplicates(subset=['date'], keep='last')
            elif len(dfs) == 1:
                df_final = dfs[0].copy()
            else:
                df_final = pd.DataFrame()
        else:
            df_final = df_api.copy()
            df_final['symbol'] = symbol
            df_final['code'] = code

        df_final = df_final.sort_values('date').reset_index(drop=True)

        try:
            self._store_to_cache(symbol, code, df_final, effective_s_dt, effective_e_dt)
        except Exception as e:
            print(f"写入失败 [{symbol}]: {e}")

        mask = (df_final['date'] >= effective_s_dt) & (df_final['date'] <= effective_e_dt)
        df_raw_slice = df_final.loc[mask].copy()
        
        return self._apply_dynamic_adjust(df_raw_slice, symbol, code, adjust)
    

    def update_daily_market_data(self, date_str=None):
        """
        🚀 全市场每日数据一键极速增量更新（纯物理底稿版 + 股票纯化版）
        :param date_str: 格式为 'YYYY-MM-DD'。如果为 None，则自动使用今天日期。
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
            
        print(f"\n==================== [开始全市场每日增量同步: {date_str}] ====================")
        
        # 1. 自动利用统一 Session 执行器获取今日全市场复权因子变更名单
        print("步骤 [1/3]: 正在获取今日复权因子变更数据...")
        df_factors_today = self.code_fetcher.query_daily_adjust_factor(date_str)
        
        ex_dividend_codes = set()
        if df_factors_today is not None and not df_factors_today.empty:
            ex_dividend_codes = set(df_factors_today['code'].tolist())
            print(f"👉 今日全市场共有 {len(ex_dividend_codes)} 只股票发生除权除息。")
        else:
            print("👉 今日全市场无除权除息事件。")
            
        # 2. 获取今日全市场 A 股日 K 线数据
        print("步骤 [2/3]: 正在获取今日全市场股票日K线数据...")
        df_astock = self.code_fetcher.query_daily_history_k_AStock(date_str)
        
        # 🌟 方案 A 修正：为了保持个股库纯净、排除生存者偏差并杜绝 ETF (如 510010) 混入，
        # 我们在这里直接注释/删掉对 ETF 日K线数据的拉取。
        df_kline_today = pd.DataFrame()
        if df_astock is not None and not df_astock.empty:
            df_kline_today = df_astock.copy()
            
        if df_kline_today.empty:
            print(f"❌ {date_str} 未获取到任何股票K线数据（可能是非交易日或数据未发布）。更新终止。")
            return
            
        # 过滤停牌或非正常交易日（tradestatus == '1' 代表正常交易）
        if 'tradestatus' in df_kline_today.columns:
            df_kline_today = df_kline_today[df_kline_today['tradestatus'] == '1'].copy()
            
        if df_kline_today.empty:
            print(f"⚠️ {date_str} 全市场无正常交易标的（全部停牌或非交易日）。")
            return
            
        print(f"步骤 [3/3]: 准备增量写入 {len(df_kline_today)} 只股票的今日不复权数据...")
        
        # 确保日期格式规范
        df_kline_today['date'] = pd.to_datetime(df_kline_today['date']).dt.strftime('%Y-%m-%d')
        
        success_count = 0
        new_stock_count = 0
        
        # 逐个标的增量写入
        for _, row in df_kline_today.iterrows():
            code = row['code']
            symbol = code.split('.')[1] if '.' in code else code
            db_path = self._get_symbol_db_path(symbol)
            
            # (A) 判断是否是新上市的股票（本地无历史数据库文件）
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
                
            # (B) 如果是今天除权的股票，优先下载并覆写其历史因子
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
                
                # 🌟 核心修正：只向物理底稿库写入带 raw_ 前缀的不复权字段，极致压缩磁盘空间
                row_dict['raw_open'] = float(row['open']) if row['open'] else None
                row_dict['raw_high'] = float(row['high']) if row['high'] else None
                row_dict['raw_low'] = float(row['low']) if row['low'] else None
                row_dict['raw_close'] = float(row['close']) if row['close'] else None
                row_dict['raw_preclose'] = float(row['preclose']) if row['preclose'] else None
                row_dict['raw_volume'] = float(row['volume']) if row['volume'] else None
                row_dict['raw_amount'] = float(row['amount']) if row['amount'] else None
                
                # 写入对应的原始比例列
                row_dict['raw_turnover'] = float(row['turn']) if row['turn'] else None
                row_dict['raw_change_pct'] = float(row['pctChg']) if row['pctChg'] else None
                
                # 安全写入：先清除可能存在的今日旧数据，然后以 append 方式安全插入
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    # 检查表是否存在（极安全保护）
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_data'")
                    table_exists = cursor.fetchone() is not None
                    
                    if table_exists:
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
                print(f"写入 [{code}] 每日增量失败: {e}")
                
        print(f"\n==================== [每日增量同步完成] ====================")
        print(f"🎯 成功同步股票: {success_count} 只 | 自动建库新股: {new_stock_count} 只 | 同步日期: {date_str}")