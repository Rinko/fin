# -*- coding: utf-8 -*-
import akshare as ak
import pandas as pd

def basic_screen():
    try:
        stocks_df = pd.read_excel( 
            "sector2stocks_list.xlsx",
            dtype = {'代码': str}
        )
    except FileNotFoundError:
        print("未找到 sector2stocks_list.xlsx 文件，请先运行 get_base_data.py 获取基础数据。")
        return []

    common_screened = stocks_df[~stocks_df['名称'].str.contains('ST', case=False, na=True)].copy()

    # 提取前缀用于判断板块
    common_screened['prefix'] = common_screened['代码'].str[:3]  # 取前3位
    # 剔除不需要的股票板块
    exclude_prefixes = ['900', '200', '730']  # 沪市B股、深市B股、新股申购
    common_screened = common_screened[~common_screened['prefix'].isin(exclude_prefixes)]

    # common_screened = common_screened[~common_screened['代码'].astype(str).str.startswith('0', na=False)]

    print(f"公共参数筛选后剩余 {len(common_screened)} 支股票")
    stocks_list = common_screened['代码'].tolist()

    return stocks_list