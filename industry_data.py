# -*- coding: utf-8 -*-
"""
申万一级行业数据同步模块 (数据源: swsresearch.com, 验证稳定)

提供:
  1. fetch_industry_daily(): 31个申万一级行业日K (1999~今), 增量更新
  2. fetch_industry_components(): 个股->申万一级行业成分映射 (全量刷新)
  3. load_industry_map(): 返回 {symbol: 行业代码} 映射
  4. load_industry_daily(): 返回行业指数日K DataFrame

缓存位置: external_data/industry/ (外接盘)
设计: 行业指数每日增量, 成分映射低频全量(新股才变, 150日过滤保证不滞后于模型可用)
"""
import os
import time
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _ak():
    """惰性导入 akshare，仅抓取功能需要；纯读缓存不依赖。"""
    import akshare as ak
    return ak

# 缓存目录 (外接盘, AGENTS.md 磁盘管理规则)
BASE_DIR = os.path.join('external_data', 'industry')
DAILY_FILE = os.path.join(BASE_DIR, 'industry_daily.parquet')
COMPONENTS_FILE = os.path.join(BASE_DIR, 'industry_components.parquet')


def _ensure_dir():
    os.makedirs(BASE_DIR, exist_ok=True)


def get_industry_list():
    """获取申万一级行业代码列表 (31个)。返回 DataFrame: 行业代码/行业名称。"""
    ak = _ak()
    info = ak.sw_index_first_info()
    codes = info['行业代码'].str.replace('.SI', '', regex=False)
    return info.assign(行业代码=info['行业代码'].astype(str))


def fetch_industry_daily(force_full=False):
    """
    抓取/更新 31 个申万一级行业日K。
    首次全量 (1999~今, ~40s)；之后增量 (仅拉最近 60 交易日覆盖, 对齐到已有数据尾部)。
    返回 DataFrame: 代码/日期/收盘/开盘/最高/最低/成交量/成交额
    """
    _ensure_dir()
    existing = None
    last_date = None
    if os.path.exists(DAILY_FILE) and not force_full:
        try:
            existing = pd.read_parquet(DAILY_FILE)
            last_date = pd.to_datetime(existing['日期']).max()
            logger.info(f"行业日K缓存已存在: {len(existing)}行, 最新 {last_date.date()}")
        except Exception as e:
            logger.warning(f"读取行业缓存失败, 全量重抓: {e}")
            existing = None

    info = get_industry_list()
    codes = info['行业代码'].astype(str).str.replace('.SI', '', regex=False).tolist()

    all_frames = []
    for i, code in enumerate(codes):
        try:
            ak = _ak()
            df = ak.index_hist_sw(symbol=code, period='day')
            df = df[['代码', '日期', '收盘', '开盘', '最高', '最低', '成交量', '成交额']].copy()
            df['代码'] = code
            df['日期'] = pd.to_datetime(df['日期'])
            # 增量: 保留 last_date 之后的数据 (含当天空档重放)
            if existing is not None and not force_full:
                df = df[df['日期'] > last_date]
            if len(df) > 0:
                all_frames.append(df)
        except Exception as e:
            logger.error(f"行业 [{code}] 日K抓取失败: {e}")
        time.sleep(0.15)  # 防频控

    if not all_frames:
        logger.info("无新增行业数据 (已是最新)")
        return existing

    new_df = pd.concat(all_frames, ignore_index=True)
    merged = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df
    merged = merged.drop_duplicates(subset=['代码', '日期']).sort_values(['代码', '日期'])
    merged.to_parquet(DAILY_FILE)
    logger.info(f"行业日K已更新: {len(merged)}行, 最新 {pd.to_datetime(merged['日期']).max().date()}")
    return merged


def fetch_industry_components(force_full=False):
    """
    全量刷新 申万一级行业成分映射 (个股 -> 行业代码)。
    31 行业 ~16s。仅需低频刷新 (新股纳入时才变)。
    返回 DataFrame: 证券代码/证券名称/行业代码
    """
    _ensure_dir()
    if os.path.exists(COMPONENTS_FILE) and not force_full:
        df = pd.read_parquet(COMPONENTS_FILE)
        logger.info(f"行业成分映射已存在: {len(df)}只")
        return df

    info = get_industry_list()
    codes = info['行业代码'].astype(str).str.replace('.SI', '', regex=False).tolist()
    names = dict(zip(codes, info['行业名称'].astype(str)))

    all_frames = []
    for code in codes:
        try:
            ak = _ak()
            df = ak.index_component_sw(symbol=code)
            df['行业代码'] = code
            df['行业名称'] = names[code]
            all_frames.append(df[['证券代码', '证券名称', '行业代码', '行业名称']])
        except Exception as e:
            logger.error(f"成分 [{code}] 抓取失败: {e}")
        time.sleep(0.15)

    comp = pd.concat(all_frames, ignore_index=True)
    comp['证券代码'] = comp['证券代码'].astype(str).str.zfill(6)
    comp.to_parquet(COMPONENTS_FILE)
    logger.info(f"行业成分映射已刷新: {len(comp)}只, {comp['行业代码'].nunique()}个行业")
    return comp


def load_industry_map():
    """返回 {symbol_6位: 行业代码} 映射。文件不存在返回空 dict (调用方降级为中性值)。"""
    if not os.path.exists(COMPONENTS_FILE):
        return {}
    comp = pd.read_parquet(COMPONENTS_FILE)
    return dict(zip(comp['证券代码'].astype(str), comp['行业代码'].astype(str)))


def load_industry_daily():
    """返回行业指数日K DataFrame (含 代码/日期/收盘...)。文件不存在返回空 DataFrame。"""
    if not os.path.exists(DAILY_FILE):
        return pd.DataFrame()
    return pd.read_parquet(DAILY_FILE)


def sync_industry_data(force_full=False):
    """每日任务入口: 行业日K增量 + 成分映射低频刷新。返回状态 dict。"""
    daily = fetch_industry_daily(force_full=force_full)
    components = fetch_industry_components(force_full=force_full)
    return {
        'daily_rows': len(daily) if daily is not None else 0,
        'components': len(components),
        'daily_file': DAILY_FILE,
        'components_file': COMPONENTS_FILE,
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    result = sync_industry_data()
    print(result)
