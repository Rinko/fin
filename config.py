# -*- coding: utf-8 -*-
"""
config.py — 全系统唯一参数源

规则：
1. 除本文件外，任何模块禁止 os.environ[...]=... 写入（读取 .get 允许，但值必须由这里盖章）。
2. apply(line) 会把 MANAGED_KEYS 全量重盖为该业务线的确定性状态——未列出的键一律回落 DEFAULTS，
   保证每次调用后环境完全可预测、无历史泄漏。
3. 新增参数：先加进 DEFAULTS，再按需在各 PROFILES 里覆盖。
"""
import os

# ---- 生产现役模型（ALIGN 四件套）----
MODELS = {
    'ENTRY_MODEL_PKL':  'chip_accumulation_v6_g_pca1_z_hfq.pkl',
    'RISK_MODEL_PKL':   'chip_risk_model_v1_g_pca1_z_hfq.pkl',
    'OPPORT_PKL':       'chip_opport_magnitude_excess_for_g_hfq.pkl',
    'RISKMAG_PKL':      'chip_risk_magnitude_for_g_hfq.pkl',
}

# ---- 全量受管键与生产默认值 ----
DEFAULTS = {
    # 仓位 sizing
    'BASE_TARGET_SIZE': '0.04',
    'POS_MULT_WEIGHT':  '0.5',
    'POS_MULT_BIAS':    '0.5',
    'OPPORT_SIZING_COEFF': '0.30',
    'OPPORT_SIZING_MIN':   '0.5',   # 生产锚点口径(main直跑引擎默认)
    'OPPORT_SIZING_MAX':   '1.5',
    'OPPORT_HURDLE':       '0.02',
    # 风控退出
    'RISK_MAG_SELL_THRESHOLD': '-0.05',
    # ml_rank floor（默认全关）
    'ML_RANK_FLOOR_BOTTOM': '0.0', 'ML_RANK_FLOOR_OPPORTUNITY': '0.0',
    'ML_RANK_FLOOR_NORMAL': '0.0', 'ML_RANK_FLOOR_CAUTION': '0.0',
    'ML_RANK_FLOOR_RISK': '0.0',
    # 场景化 quota
    'BUY_QUOTA_BOTTOM': '5', 'BUY_QUOTA_OPPORTUNITY': '5', 'BUY_QUOTA_NORMAL': '2',
    'BUY_QUOTA_CAUTION': '3', 'BUY_QUOTA_RISK': '0',
    # 业务开关
    'MODERATE_BUSINESS_RULES': '0',
    'USE_PROFIT_RATIO_CON': '0',
    'CO_COMBO_FEATURES': '0',
    # 指标预热（语义参数）：所有入口的 fetch 起点由 padded_start() 按交易日历精确回推，
    # 保证 EMA 等递归特征与抓取起点无关
    'WARMUP_TRADING_DAYS': '600',   # 深预热：EMA 路径依赖归零，跨入口打分逐位一致
    # 回测控制
    'BUY_QUOTA_OVERRIDE': '',          # 空串=删除该 env，走场景化 quota
    'BASELINE_END': '2026-08-17',
    'INITIAL_CASH': '1000000',
    'RESULTS_DIR_SUFFIX': '',
    'DEBUG_INFERENCE': '0',
    # 训练线
    'TRAIN_UNIVERSE_FILTER': '0', 'TARGET_EXCESS': '0', 'ENTRY_PRICE_MODE': 'close',
    'SKIP_AUDIT_CSV': '0', 'TRAIN_SELL': '0', 'TRAIN_OUTPUT_PKL': '',
}
DEFAULTS.update(MODELS)
MANAGED_KEYS = list(DEFAULTS)

# ---- 业务线 profile（只写与默认值不同的键）----
PROFILES = {
    'prod':     {},
    'backtest': {'BASELINE_END': '2026-08-21'},
    'signals':  {'BUY_QUOTA_OVERRIDE': '0'},   # 只出候选不下单
    'bench':    {'MODERATE_BUSINESS_RULES': '1', 'RESULTS_DIR_SUFFIX': '_bench'},
    'audit':    {},
    'daily':    {},   # 轻量每日信号（买入+卖出）
    'train':    {k: v for k, v in DEFAULTS.items() if k.startswith(('TRAIN_',)) },
}


def apply(line):
    """把环境重盖为指定业务线的确定性状态。返回生效差异摘要。"""
    if line not in PROFILES:
        raise SystemExit(f"[config] 未知业务线 '{line}'，可选: {sorted(PROFILES)}")
    changed = []
    for k in MANAGED_KEYS:
        want = str(PROFILES[line].get(k, DEFAULTS[k]))
        cur = os.environ.get(k)
        if k == 'BUY_QUOTA_OVERRIDE' and want == '':
            if cur is not None:
                os.environ.pop(k, None); changed.append(f'-{k}')
        else:
            if cur != want:
                os.environ[k] = want
                changed.append(f'{k}={want}')
    return changed


def padded_start(end, warmup=None):
    """按中证全指交易日历，从 end 精确回推 warmup 个交易日。"""
    import pandas as pd
    w = int(warmup) if warmup is not None else int(os.environ.get('WARMUP_TRADING_DAYS', '270'))
    z = pd.read_excel('zzqz_df.xlsx').rename(columns={'日期': 'date'})
    d = pd.to_datetime(z['date'])
    d = d[d <= pd.Timestamp(end)].sort_values().reset_index(drop=True)
    return d.iloc[max(len(d) - 1 - w, 0)].strftime('%Y-%m-%d')


def summary(line):
    return {k: os.environ.get(k, '<unset>') for k in MANAGED_KEYS}
