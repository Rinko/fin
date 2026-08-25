"""price_display.py — hfq→qfq 展示层换算

架构约定：系统内部全链路统一后复权(hfq)；仅对用户展示的绝对价格字段
换算回前复权(qfq)以符合使用习惯。比率类字段(pnl_pct/atr_ratio等)不受口径影响。

换算原理：close_qfq(t) = close_hfq(t) × F_q(t)/F_h(t)，因子比在非除权日恒定，
日度信号取最新交易日比值即为精确解。
"""
import pandas as pd
from local_data_cache import LocalDataCache


def latest_qfq_ratio(symbols):
    """返回 {symbol: close_qfq/close_hfq}（最新共同交易日比值）。"""
    ldc = LocalDataCache(cache_dir='./stock_data_cache')
    y = pd.Timestamp.now().year
    out = {}
    for s in symbols:
        try:
            dq = ldc.get_stock_data(s, f'{y}-01-01', '2100-01-01', adjust='qfq', mode=2)
            dh = ldc.get_stock_data(s, f'{y}-01-01', '2100-01-01', adjust='hfq', mode=2)
            if dq is None or dh is None or len(dq) == 0 or len(dh) == 0:
                continue
            if str(dq['date'].iloc[-1])[:10] != str(dh['date'].iloc[-1])[:10]:
                continue
            cq, ch = float(dq['close'].iloc[-1]), float(dh['close'].iloc[-1])
            if ch > 0 and cq > 0:
                out[s] = cq / ch
        except Exception:
            continue
    return out


def convert_close_column(df, symbol_col='symbol', price_cols=('close',)):
    """就地换算 df 中指定价格列 hfq→qfq（缺失比值的行保持原值）。"""
    r = latest_qfq_ratio(df[symbol_col].dropna().unique())
    for c in price_cols:
        if c in df.columns:
            df[c] = df[c] * df[symbol_col].map(r).fillna(1.0)
    return df
