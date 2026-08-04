import pandas as pd
import baostock as bs
import time

class BaostockCodeFetcher:
    def __init__(self):
        pass

    def fetch_single_stock(self, code, start_date, end_date, adjust="qfq"):
        """
        通过 BaoStock 接口获取单只股票的日K线数据。
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

        try:
            # 2. 指定拉取的日线指标字段
            fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus"
            
            rs = bs.query_history_k_data_plus(
                code=code,
                fields=fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjustflag
            )
            time.sleep(1) # 根据高频访问限流进行微调
            
            if rs.error_code != '0':
                print(f"[{code}] 历史K线数据拉取失败: {rs.error_msg}")
                return None

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
            df['amplitude'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(4)
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
                df['raw_preclose'] = df['preclose'] # 核心：用于离线除权比较的物理列
                df['raw_volume'] = df['volume']
                df['raw_amount'] = df['amount']
                
                raw_cols = ['raw_open', 'raw_high', 'raw_low', 'raw_close', 'raw_preclose', 'raw_volume', 'raw_amount']

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
        try:
            rs = bs.query_daily_history_k_AStock(date=date)
            if rs.error_code != '0':
                print(f"[AStock批量] {date} 获取失败: {rs.error_msg}")
                return None

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=rs.fields)
            return df
        except Exception as e:
            print(f"[AStock批量] {date} 获取时发生异常: {e}")
            return None

    # ==================== [新增：1.2 每日 ETF K线批量更新接口] ====================
    def query_daily_history_k_ETF(self, date):
        """
        获取某日所有ETF不复权日K线数据 (BaoStock v0.9.3+)
        :param date: 目标日期 (格式 'YYYY-MM-DD')
        :return: pandas.DataFrame
        """
        try:
            rs = bs.query_daily_history_k_ETF(date=date)
            if rs.error_code != '0':
                print(f"[ETF批量] {date} 获取失败: {rs.error_msg}")
                return None

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=rs.fields)
            return df
        except Exception as e:
            print(f"[ETF批量] {date} 获取时发生异常: {e}")
            return None

    # ==================== [新增：1.3 每日复权因子批量更新接口] ====================
    def query_daily_adjust_factor(self, date):
        """
        获取某日全市场发生除权除息的复权因子变更信息 (BaoStock v0.9.3+)
        :param date: 目标日期 (格式 'YYYY-MM-DD')
        :return: pandas.DataFrame
        """
        try:
            rs = bs.query_daily_adjust_factor(date=date)
            if rs.error_code != '0':
                print(f"[因子批量] {date} 获取失败: {rs.error_msg}")
                return None

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=rs.fields)
            return df
        except Exception as e:
            print(f"[因子批量] {date} 获取时发生异常: {e}")
            return None

    # ==================== [新增：2.0 单股历史复权因子接口] ====================
    def query_adjust_factor(self, code, start_date, end_date):
        """
        获取单只股票历史时间跨度内的复权因子表。
        :param code: 带市场缩写前缀的代码 (如 'sh.600000')
        :param start_date: 开始日期 (格式 'YYYY-MM-DD')
        :param end_date: 结束日期 (格式 'YYYY-MM-DD')
        :return: pandas.DataFrame
        """
        try:
            rs = bs.query_adjust_factor(code=code, start_date=start_date, end_date=end_date)
            if rs.error_code != '0':
                print(f"[{code}] 获取历史复权因子失败: {rs.error_msg}")
                return None

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=rs.fields)
            return df
        except Exception as e:
            print(f"[{code}] 获取历史复权因子发生异常: {e}")
            return None

    def login(self):
        return bs.login()

    def logout(self):
        return bs.logout()