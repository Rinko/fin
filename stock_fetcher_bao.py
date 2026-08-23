import pandas as pd
import numpy as np
import baostock as bs
import time
import logging

class BaostockCodeFetcher:
    _active_instance = None

    def __init__(self):
        if BaostockCodeFetcher._active_instance is not None:
            logging.warning(
                "已有 BaostockCodeFetcher 实例存活，新实例可能引发 Session 冲突"
            )
        BaostockCodeFetcher._active_instance = self
        self.is_logged_in = False

    def ensure_login(self):
        """
        延迟加载保护：确保处于安全登录状态
        """
        if not self.is_logged_in:
            print("[API] 正在建立与 BaoStock 的安全连接并执行鉴权...")
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    self.is_logged_in = True
                else:
                    print(f"[API] BaoStock 登录失败: {lg.error_msg}")
                    self.is_logged_in = False
            except Exception as e:
                print(f"[API] 登录时发生网络连接异常: {e}")
                self.is_logged_in = False
        return self.is_logged_in

    def logout(self):
        """
        主动断开服务器连接，释放资源
        """
        if self.is_logged_in:
            try:
                bs.logout()
                print("[API] 成功断开与 BaoStock 服务器的连接。")
            except Exception as e:
                logging.warning(f"[API] logout 异常（可忽略）: {e}")
            self.is_logged_in = False

    def __enter__(self):
        self.ensure_login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logout()
        return False

    def _execute_api_with_retry(self, bs_func, *args, **kwargs):
        """
        🌟 核心守护执行器：通用带自适应重连、报错重试、Session 恢复的底层执行代理
        """
        retry_limit = 3
        for attempt in range(retry_limit):
            if not self.ensure_login():
                time.sleep(3)
                continue
            try:
                rs = bs_func(*args, **kwargs)
                if rs.error_code == '0':
                    return rs
                elif rs.error_code in ['10001001', '10001002']: # BaoStock 经典的 Session 过期状态码
                    print("[API] 您的登录凭证（Session）已过期，正在清除本地连接状态并自动重新申请...")
                    self.is_logged_in = False
                else:
                    print(f"[API] 接口返回非0错误码 ({rs.error_code}): {rs.error_msg}")
                    time.sleep(2)  # 业务错误码也需冷却，避免高频冲击 API
            except Exception as e:
                # 捕获 Errno 54 (Connection reset)、Errno 32 (Broken pipe) 等网络重置异常
                print(f"[API] 捕获网络通信/套接字中断异常 ({e})。正在自动重置连接并进行第 {attempt+1}/{retry_limit} 次重试...")
                self.is_logged_in = False
                try:
                    bs.logout()
                except Exception:
                    pass
                time.sleep(5)  # 冷却 5 秒，让服务器释放 socket 连接池

        return None

    def fetch_single_stock(self, code, start_date, end_date, adjust="qfq"):
        """
        通过 BaoStock 接口获取单只股票的日K线数据（已接入统一 Session 治理）。
        :param code: 带市场缩写前缀的代码 (如 'sh.600000')
        :param start_date: 开始日期 (格式 'YYYY-MM-DD')
        :param end_date: 结束日期 (格式 'YYYY-MM-DD')
        :param adjust: 复权类型，支持 'qfq' (前复权), 'hfq' (后复权), 'none' 或 '3' (不复权)
        :return: pandas.DataFrame
        """
        # 1. 映射复权标志（BaoStock: 1 后复权、2 前复权、3 不复权）
        if adjust in ["qfq", "2"]:
            adjustflag = "2"
        elif adjust in ["hfq", "1"]:
            adjustflag = "1"
        else:
            adjustflag = "3"

        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus"

        # 🌟 统一使用底层重连守护器执行 API
        rs = self._execute_api_with_retry(
            bs.query_history_k_data_plus,
            code=code,
            fields=fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag
        )

        if rs is None:
            return None

        try:
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            # 3. 构建 DataFrame
            df = pd.DataFrame(data_list, columns=rs.fields)

            # 4. 过滤停牌或非正常交易日（tradestatus == '1' 代表正常交易）
            if "tradestatus" in df.columns:
                df = df[df["tradestatus"] == "1"].copy()
                if df.empty:
                    return pd.DataFrame()

            # 5. 数据类型转换
            numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 6. 计算补充指标
            df['symbol'] = df['code'].apply(lambda x: x.split('.')[1] if '.' in str(x) else str(x))

            # 振幅 (%) 公式: (最高价 - 最低价) / 昨收价 * 100
            df['amplitude'] = ((df['high'] - df['low']) / df['preclose'].replace(0, np.nan) * 100).round(4)
            # 涨跌额 公式: 今日收盘价 - 昨收价
            df['change'] = (df['close'] - df['preclose']).round(4)

            # 列名重映射
            df = df.rename(columns={
                'pctChg': 'change_pct',
                'turn': 'turnover'
            })

            # 7. 为筹码相关指标列置空，保持与本地缓存结构一致
            chip_cols = [
                'profit_ratio', 'avg_cost', 'cost_90_low', 'cost_90_high',
                'concentration_90', 'cost_70_low', 'cost_70_high', 'concentration_70'
            ]
            for col in chip_cols:
                df[col] = pd.NA

            # 8. 🌟 核心改进：当请求为“不复权（none或3）”时，双轨制生成 `raw_` 列
            raw_cols = []
            if adjustflag == "3":
                df['raw_open'] = df['open']
                df['raw_high'] = df['high']
                df['raw_low'] = df['low']
                df['raw_close'] = df['close']
                df['raw_preclose'] = df['preclose']  # 核心：用于离线除权比较的物理列
                df['raw_volume'] = df['volume']
                df['raw_amount'] = df['amount']
                df['raw_change_pct'] = df['change_pct']
                df['raw_turnover'] = df['turnover']

                raw_cols = ['raw_open', 'raw_high', 'raw_low', 'raw_close', 'raw_preclose', 'raw_volume', 'raw_amount', 'raw_change_pct', 'raw_turnover']

            # 9. 整理并规范输出列，确保前复权/后复权和不复权的完美兼容
            final_cols = [
                'date', 'symbol', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount',
                'amplitude', 'change_pct', 'change', 'turnover'
            ] + chip_cols + raw_cols

            df = df[final_cols].copy()
            return df

        except Exception as e:
            print(f"[{code}] 获取K线数据时发生异常: {e}")
            time.sleep(5)
            return None

    # ==================== [新增：1.1 每日 A股 K线批量更新接口] ====================
    def query_daily_history_k_AStock(self, date):
        """
        获取某日所有股票不复权日K线数据 (BaoStock v0.9.3+)
        :param date: 目标日期 (格式 'YYYY-MM-DD')
        :return: pandas.DataFrame
        """
        rs = self._execute_api_with_retry(bs.query_daily_history_k_AStock, date=date)
        if rs is None:
            return None
        try:
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return pd.DataFrame()
            return pd.DataFrame(data_list, columns=rs.fields)
        except Exception as e:
            print(f"[AStock批量] 数据解析异常: {e}")
            return None

    # ==================== [新增：1.15 交易日历查询接口] ====================
    def query_trade_dates(self, start_date=None, end_date=None):
        """
        查询区间内交易日历 (BaoStock query_trade_dates)。
        :return: pandas.DataFrame [calendar_date, is_trading_day]；失败返回 None
        """
        rs = self._execute_api_with_retry(bs.query_trade_dates, start_date=start_date, end_date=end_date)
        if rs is None:
            return None
        try:
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return pd.DataFrame()
            return pd.DataFrame(data_list, columns=rs.fields)
        except Exception as e:
            print(f"[交易日历] 解析异常: {e}")
            return None

    # ==================== [新增：1.2 每日 ETF K线批量更新接口] ====================
    def query_daily_history_k_ETF(self, date):
        rs = self._execute_api_with_retry(bs.query_daily_history_k_ETF, date=date)
        if rs is None:
            return None
        try:
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return pd.DataFrame()
            return pd.DataFrame(data_list, columns=rs.fields)
        except Exception as e:
            print(f"[ETF批量] 数据解析异常: {e}")
            return None

    # ==================== [新增：1.3 每日复权因子批量更新接口] ====================
    def query_daily_adjust_factor(self, date):
        rs = self._execute_api_with_retry(bs.query_daily_adjust_factor, date=date)
        if rs is None:
            return None
        try:
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return pd.DataFrame()
            return pd.DataFrame(data_list, columns=rs.fields)
        except Exception as e:
            print(f"[因子批量] 数据解析异常: {e}")
            return None

    # ==================== [新增：2.0 单股历史复权因子接口] ====================
    def query_adjust_factor(self, code, start_date, end_date):
        rs = self._execute_api_with_retry(bs.query_adjust_factor, code=code, start_date=start_date, end_date=end_date)
        if rs is None:
            return None
        try:
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return pd.DataFrame()
            return pd.DataFrame(data_list, columns=rs.fields)
        except Exception as e:
            print(f"[{code}] 历史因子解析异常: {e}")
            return None

    def query_stock_basic(self, code=None):
        rs = self._execute_api_with_retry(bs.query_stock_basic, code=code)
        if rs is None:
            return None
        try:
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return pd.DataFrame()
            return pd.DataFrame(data_list, columns=rs.fields)
        except Exception as e:
            print(f"[基本资料] 解析异常: {e}")
            return None