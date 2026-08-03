import pandas as pd
import baostock as bs
import time

class BaostockCodeFetcher:
    def __init__(self):
        pass

    def fetch_single_stock(self, code, start_date, end_date, adjust="qfq"):
        """
        通过 BaoStock 接口获取单只股票的日线交易数据。
        :param code: 带市场缩写前缀的代码 (如 'sh.600000')
        :param start_date: 开始日期 (格式 'YYYY-MM-DD')
        :param end_date: 结束日期 (格式 'YYYY-MM-DD')
        :param adjust: 复权类型，支持 'qfq' (前复权), 'hfq' (后复权), 'none' 或 None (不复权)
        :return: pandas.DataFrame
        """
        # 1. 映射复权标志（BaoStock: 1 后复权、2 前复权、3 不复权）
        if adjust == "qfq":
            adjustflag = "2"
        elif adjust == "hfq":
            adjustflag = "1"
        else:
            adjustflag = "3"

        # 2. 登录系统（API 限制要求调用前必须保持登录）
        # lg = bs.login()
        # if lg.error_code != '0':
        #     print(f"BaoStock 登录失败: {lg.error_msg}")
        #     return None

        try:
            # 3. 指定拉取的日线指标字段
            # turn 表示换手率，pctChg 表示涨跌幅 (百分比)
            fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus"
            
            rs = bs.query_history_k_data_plus(
                code=code,
                fields=fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjustflag
            )
            time.sleep(5)
            
            if rs.error_code != '0':
                print(f"[{code}] 历史K线数据拉取失败: {rs.error_msg}")
                return None

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            # 4. 构建 DataFrame
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 5. 过滤停牌或非正常交易日（tradestatus == '1' 代表正常交易）
            if "tradestatus" in df.columns:
                df = df[df["tradestatus"] == "1"].copy()
                if df.empty:
                    return pd.DataFrame()

            # 6. 数据类型转换：将所有交易指标字符串转换为数值类型
            numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 7. 计算和补充与现有库列名一致的指标
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

            # 8. 为筹码相关的指标列置空，使其与缓存层中的数据结构一致
            chip_cols = [
                'profit_ratio', 'avg_cost', 'cost_90_low', 'cost_90_high', 
                'concentration_90', 'cost_70_low', 'cost_70_high', 'concentration_70'
            ]
            for col in chip_cols:
                df[col] = pd.NA

            # 9. 整理并规范输出列，确保字段排列顺序与 LocalDataCache 一致
            final_cols = [
                'date', 'symbol', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount',
                'amplitude', 'change_pct', 'change', 'turnover'
            ] + chip_cols
            
            df = df[final_cols].copy()
            return df
            

        except Exception as e:
            print(f"[{code}] 获取K线数据时发生异常: {e}")
            time.sleep(10)
            return None
            
        # finally:
            # 10. 登出系统以释放服务器连接
            # bs.logout()

    def login(self):
        return bs.login()

    def logout(self):
        return bs.logout()