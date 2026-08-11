# -*- coding: utf-8 -*-
import os
import pandas as pd

def basic_screen(cache_dir='./stock_data_cache'):
    """
    100% 离线基础筛选函数：读取中证全指 (000985) 口径范围内的正股列表

    统一经 LocalDataCache 读取元数据，禁止直连 SQLite。
    """
    from local_data_cache import LocalDataCache

    ldc = LocalDataCache(cache_dir=cache_dir)
    df = ldc.get_stock_meta()

    if df.empty:
        print("⚠️ 股票元数据表为空，请确认是否成功同步。")
        return []

    # 1. 剔除 code_name 为空的记录
    df = df[df['code_name'].notna()].copy()

    # 2. 剔除带有 "ST" 或 "*ST" 标记的股票（大小写不敏感，与中证全指 000985 口径对齐）
    common_screened = df[~df['code_name'].str.contains('ST', case=False, na=True)].copy()

    # 3. 提取前缀用于判断板块，剔除不需要的板块
    # 900: 沪市B股, 200: 深市B股, 730: 沪市新股申购
    common_screened['prefix'] = common_screened['symbol'].str[:3]  # 取前3位
    exclude_prefixes = ['900', '200', '730']
    common_screened = common_screened[~common_screened['prefix'].isin(exclude_prefixes)]

    # 4. 兼容性输出：原 excel 的"代码"列为不带前缀的 6 位数字
    # 返回 symbol (如 ['600000', '000001'])，完美契合回测依赖，防止下游指标计算崩溃
    stocks_list = common_screened['symbol'].tolist()
    print(f"✨ 数据库筛选后剩余 {len(stocks_list)} 支有效 A 股股票")

    return stocks_list