import sqlite3
import os
import pandas as pd
from datetime import datetime

class LocalDataCache:
    def __init__(self, symbol_fetcher=None, code_fetcher=None, cache_dir='./stock_data_cache'):
        """
        升级版本地行情缓存层：
        1. 采用全局 `stock_data.db` 集中存储复权因子。
        2. 个股 `.db` 采用双轨制，保留原 `open/close` 列，新增 `raw_` 不复权列。
        """
        self.symbol_fetcher = symbol_fetcher
        self.code_fetcher = code_fetcher
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, 'stock_data.db')  # 主配置库
        
        # 1. 初始化或升级主配置表 cache_meta & 集中存储的因子表 adjust_factors
        self._init_main_db()
        
        self._code_map = {}
        # 自动迁移与升级已有的个股缓存文件
        self._migrate_existing_cache_files()

    def _init_main_db(self):
        """
        初始化全局主数据库中的元数据表和复权因子表
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 创建/兼容元数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_meta (
                    symbol TEXT PRIMARY KEY,
                    code TEXT,
                    last_updated TEXT,
                    min_date TEXT,
                    max_date TEXT
                )
            ''')
            # 确保 code 列存在
            cursor.execute("PRAGMA table_info(cache_meta)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'code' not in columns:
                cursor.execute("ALTER TABLE cache_meta ADD COLUMN code TEXT")
                
            # 🌟 创建全局复权因子表 (方案 B)
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

    def _migrate_existing_cache_files(self):
        """
        老版缓存兼容性无损升级：
        自动检测个股数据库，并安全追加 `raw_` 不复权系列列，确保旧代码不崩溃。
        """
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
                        
                        # 检查并自动追加 `raw_` 列（如果不存在的话）
                        raw_cols_to_add = {
                            'raw_open': 'REAL',
                            'raw_high': 'REAL',
                            'raw_low': 'REAL',
                            'raw_close': 'REAL',
                            'raw_preclose': 'REAL', # ⚠️ 除权检测核心依赖
                            'raw_volume': 'REAL',
                            'raw_amount': 'REAL'
                        }
                        
                        mutated = False
                        for col_name, col_type in raw_cols_to_add.items():
                            if col_name not in existing_cols:
                                print(f"正在对 [{file}] 自动热升级，追加不复权字段: {col_name}...")
                                cursor.execute(f"ALTER TABLE stock_data ADD COLUMN {col_name} {col_type}")
                                mutated = True
                        
                        if mutated:
                            # 兼容性填充：如果是历史老数据库，由于当时没有保存不复权数据
                            # 我们可以默认将当时的 `open` 等前复权数据先复制给 `raw_open` 作为历史占位符
                            # 确保后续做 `raw_` 比较时不为 NULL
                            for col_name in raw_cols_to_add.keys():
                                origin_name = col_name.replace('raw_', '')
                                if origin_name in existing_cols:
                                    cursor.execute(f"UPDATE stock_data SET {col_name} = {origin_name} WHERE {col_name} IS NULL")
                            
                            # 针对特殊的 preclose 字段，默认等于 close（安全假设）
                            if 'raw_preclose' in raw_cols_to_add and 'close' in existing_cols:
                                cursor.execute("UPDATE stock_data SET raw_preclose = close WHERE raw_preclose IS NULL")
                                
                            conn.commit()
                            
                except Exception as e:
                    print(f"老版文件 [{file}] 兼容性热升级失败: {e}")

    def _resolve_ids(self, stock_id):
        """
        统一解析输入的标识符（无论是 "600000" 还是 "sh.600000"）
        返回标准元组 (symbol, code)
        """
        stock_id = str(stock_id).strip()
        if '.' in stock_id:
            code = stock_id
            symbol = stock_id.split('.')[1]
        else:
            symbol = stock_id.zfill(6)
            code = self._get_code_from_symbol(symbol)
        return symbol, code

    def _get_code_from_symbol(self, symbol):
        """
        查找 code，优先从内存、数据库映射，最后用规则兜底
        """
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

        # 规则兜底
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
        """
        查询当前股票在 cache_meta 中记录的数据日期跨度
        """
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
        """
        检查本地已缓存区间是否能【完全覆盖】请求区间 [s_dt, e_dt]
        """
        min_date, max_date = self._get_cached_date_range(symbol)
        if min_date and max_date:
            # 请求的开始和结束日期都在本地已缓存范围之内，视为完全覆盖
            return s_dt >= min_date and e_dt <= max_date
        return False

    def _get_symbol_db_path(self, symbol):
        return os.path.join(self.cache_dir, f'{symbol}.db')

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
    
    def _store_to_cache(self, symbol, code, df, s_dt=None, e_dt=None):
        if df is None or df.empty: return
        
        db_path = self._get_symbol_db_path(symbol)
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
            
            df['symbol'] = symbol
            df['code'] = code
            
            df.to_sql('stock_data', conn, if_exists='replace', index=False)

            min_date = df['date'].min()
            max_date = df['date'].max()

            # --- 核心修正：利用请求的上下界拓宽记录，完美解决 IPO 与周末/非交易日引起的“未完全覆盖”问题 ---
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

    def get_stock_data(self, stock_id, start_date, end_date, adjust="qfq", mode=1, online=None):
        """
        获取日K及筹码等综合数据。
        :param stock_id: 支持 '600000' 或 'sh.600000' 等任意格式
        :param start_date: 开始日期 (如 '2025-01-01')
        :param end_date: 结束日期 (如 '2025-12-31')
        :param adjust: 复权方式，默认 'qfq'
        :param mode: 读取模式选择：
                     1 - 智能混合模式：本地完全覆盖则直接读取本地；若未覆盖或部分覆盖，则通过网络拉取缺失区间并同步本地。
                     2 - 纯本地模式：完全读取本地。若数据不能完全覆盖，输出提示信息，并返回本地已有部分。
                     3 - 强制网络模式：直接访问网络拉取并覆盖/更新本地。
        :param online: 旧版参数兼容。如果 online 传入 True 映射为 mode=1，False 映射为 mode=2。
        """
        # 0. 兼容旧接口的 online 传入
        if online is not None:
            mode = 1 if online else 2

        symbol, code = self._resolve_ids(stock_id)
        s_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        e_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        # ==================== [MODE 2: 纯本地模式] ====================
        if mode == 2:
            min_date, max_date = self._get_cached_date_range(symbol)
            if not min_date or not max_date:
                print(f"[{symbol} / {code}] 提示: 本地暂无任何缓存数据，缺少请求区间 [{s_dt} 至 {e_dt}]。")
            else:
                missing_info = []
                if s_dt < min_date:
                    missing_info.append(f"前段缺失 [{s_dt} 至 {min_date}]")
                if e_dt > max_date:
                    missing_info.append(f"后段缺失 [{max_date} 至 {e_dt}]")
                if missing_info:
                    print(f"[{symbol} / {code}] 提示: 本地数据不能完全覆盖请求区间！目前" + "，".join(missing_info))
            
            return self._get_from_cache(symbol, code, s_dt, e_dt)

        # ==================== [MODE 1: 智能混合模式 - 覆盖性检查] ====================
        if mode == 1:
            is_covered = self._check_coverage(symbol, s_dt, e_dt)
            if is_covered:
                # 完全覆盖，无需请求网络，直接返回本地缓存
                print(f"[{symbol} / {code}] 本地数据覆盖请求区间，直接返回数据")
                return self._get_from_cache(symbol, code, s_dt, e_dt)
            
            else:
                print(f"[{symbol} / {code}] 本地数据未能完全覆盖请求区间，切换至网络拉取模式...")

        # ==================== [线上数据拉取流程 (MODE 1 未覆盖 或 MODE 3)] ====================
        print(f"[{symbol} / {code}] 正在从线上同步并更新数据: {s_dt} 至 {e_dt}...")
        df_api = None
        
        # 优先使用 code_fetcher (带前缀)
        if self.code_fetcher is not None:
            try:
                df_api = self.code_fetcher.fetch_single_stock(code, s_dt, e_dt, adjust)
            except Exception as e:
                print(f"[{code}] 通过 code_fetcher 拉取线上数据失败: {e}")
                
        # 降级使用 symbol_fetcher (纯数字)
        if (df_api is None or df_api.empty) and self.symbol_fetcher is not None:
            try:
                df_api = self.symbol_fetcher.fetch_single_stock(symbol, s_dt, e_dt, adjust)
            except Exception as e:
                print(f"[{symbol}] 通过 symbol_fetcher 拉取线上数据失败: {e}")
        
        # 如果线上拉取失败（如网络中断），降级读取本地缓存并警示
        if df_api is None or df_api.empty:
            print(f"[{symbol}] 线上网络拉取失败，降级使用本地已有缓存。")
            return self._get_from_cache(symbol, code, s_dt, e_dt)

        # 格式化 API 日期
        df_api['date'] = pd.to_datetime(df_api['date'], format='mixed').dt.strftime('%Y-%m-%d')

        # 读取本地全量，准备执行拼合
        df_db = self._get_from_cache(symbol, code, '1900-01-01', '2100-01-01')
        
        if not df_db.empty:
            df_db['date'] = pd.to_datetime(df_db['date'], format='mixed').dt.strftime('%Y-%m-%d')
            
            kline_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'change_pct', 'change', 'turnover']
            chip_cols = ['profit_ratio', 'avg_cost', 'cost_90_low', 'cost_90_high', 
                         'concentration_90', 'cost_70_low', 'cost_70_high', 'concentration_70']
            
            # 补足各列并强制指定为 float64 类型，防止 all-NA 引起 Pandas 警告
            for col in kline_cols:
                if col not in df_api.columns:
                    df_api[col] = pd.to_numeric(pd.Series([pd.NA] * len(df_api)), errors='coerce')
                else:
                    df_api[col] = pd.to_numeric(df_api[col], errors='coerce')
                    
                if col not in df_db.columns:
                    df_db[col] = pd.to_numeric(pd.Series([pd.NA] * len(df_db)), errors='coerce')
                else:
                    df_db[col] = pd.to_numeric(df_db[col], errors='coerce')

            for col in chip_cols:
                if col not in df_api.columns:
                    df_api[col] = pd.to_numeric(pd.Series([pd.NA] * len(df_api)), errors='coerce')
                else:
                    df_api[col] = pd.to_numeric(df_api[col], errors='coerce')
                    
                if col not in df_db.columns:
                    df_db[col] = pd.to_numeric(pd.Series([pd.NA] * len(df_db)), errors='coerce')
                else:
                    df_db[col] = pd.to_numeric(df_db[col], errors='coerce')

            # 注入标准化标识列
            df_api['symbol'] = symbol
            df_api['code'] = code
            df_db['symbol'] = symbol
            df_db['code'] = code

            # --- 外科手术融合逻辑 ---
            # (1) K线拼合并去重（keep='last' 代表信任线上最新数据，保证复权修正生效）
            df_kline_db = df_db[['date', 'symbol', 'code'] + kline_cols]
            df_kline_api = df_api[['date', 'symbol', 'code'] + kline_cols]
            df_kline_merged = pd.concat([df_kline_db, df_kline_api]).drop_duplicates(subset=['date'], keep='last')
            
            # (2) 筹码拼接（只追溯超出本地最大日期之后的 API 筹码增量）
            df_chip_db = df_db[['date'] + chip_cols]
            max_db_date = df_db['date'].max()
            df_chip_api_new = df_api[df_api['date'] > max_db_date][['date'] + chip_cols]
            
            if df_chip_api_new.empty:
                df_chip_merged = df_chip_db
            elif df_chip_db.empty:
                df_chip_merged = df_chip_api_new
            else:
                df_chip_merged = pd.concat([df_chip_db, df_chip_api_new]).drop_duplicates(subset=['date'], keep='last')
            
            # (3) 再次大表合流
            df_final = pd.merge(df_kline_merged, df_chip_merged, on='date', how='left')
            # --- 拼合结束 ---
        else:
            df_final = df_api.copy()
            df_final['symbol'] = symbol
            df_final['code'] = code

        # 清洗重整并去重
        df_final = df_final.sort_values('date').drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)

        try:
            self._store_to_cache(symbol, code, df_final, s_dt, e_dt)
        except Exception as e:
            print(f"写入本地数据库失败 [{symbol} / {code}]: {e}")

        # 截取用户请求的范围返回
        mask = (df_final['date'] >= s_dt) & (df_final['date'] <= e_dt)
        return df_final.loc[mask].copy()

    def get_data_at_date(self, stock_id, query_date, lookback=True):
        """
        单点特定日期数据提取
        """
        symbol, code = self._resolve_ids(stock_id)
        if hasattr(query_date, 'strftime'):
            date_str = query_date.strftime('%Y-%m-%d')
        else:
            date_str = str(query_date)[:10]

        db_path = self._get_symbol_db_path(symbol)
        if not os.path.exists(db_path):
            return None

        try:
            with sqlite3.connect(db_path) as conn:
                if lookback:
                    query = f"SELECT * FROM stock_data WHERE date <= ? ORDER BY date DESC LIMIT 1"
                    df = pd.read_sql(query, conn, params=(f"{date_str} 23:59:59",))
                else:
                    query = f"SELECT * FROM stock_data WHERE date LIKE ?"
                    df = pd.read_sql(query, conn, params=(f"{date_str}%",))

                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'], format='mixed')
                    df['symbol'] = symbol
                    df['code'] = code
                    return df.iloc[0]
                
                return None
        except Exception as e:
            print(f"读取单日快照异常 [{symbol} @ {date_str}]: {e}")
            return None



from stock_fetcher_bao import BaostockCodeFetcher
if __name__ == "__main__":
    fetcher = BaostockCodeFetcher()
    stock_data_cache = LocalDataCache(code_fetcher=fetcher, cache_dir="./stock_data_cache")