#!/usr/bin/env python3
"""
signal_engine.py — 纯信号规则层（无 PyBroker、无现金约束）

把买入资格判断、卖出判断、目标仓位计算从 backtest.py 中抽离出来，
供 PyBroker 组合回测、信号级回测、实盘信号生成共用同一套规则。
"""
import os
import numpy as np

# =========================================================================
# 业务规则开关（通过环境变量调整，保持与 backtest.py 历史默认一致）
# =========================================================================
MODERATE_BUSINESS_RULES = os.environ.get('MODERATE_BUSINESS_RULES', '0') == '1'
RELAXED_EXIT_RULES = os.environ.get('RELAXED_EXIT_RULES', '0') == '1'

# 止损 / 止盈参数
NEG_PNL_RISK_THRESHOLD = float(os.environ.get('NEG_PNL_RISK_THRESHOLD', '0.10'))
PROFIT_RISK_THRESHOLD = float(os.environ.get('PROFIT_RISK_THRESHOLD', '0.05'))
RISK_DETERIORATION_THRESHOLD = float(os.environ.get('RISK_DETERIORATION_THRESHOLD', '-0.25'))
TIME_EXIT_BARS = int(os.environ.get('TIME_EXIT_BARS', '8'))
TIME_EXIT_PNL = float(os.environ.get('TIME_EXIT_PNL', '0.015'))
TRAILING_PROFIT_THRESHOLD = float(os.environ.get('TRAILING_PROFIT_THRESHOLD', '0.06'))
ENABLE_HARD_ATR_STOP = os.environ.get('ENABLE_HARD_ATR_STOP', '0') == '1'
HARD_ATR_STOP_MULT = float(os.environ.get('HARD_ATR_STOP_MULT', '2.0'))

# 幅度模型阈值（可选，未加载时不使用）
try:
    RISK_MAG_SELL_THRESHOLD = float(os.environ.get('RISK_MAG_SELL_THRESHOLD', '-0.03'))
except Exception:
    RISK_MAG_SELL_THRESHOLD = -0.03

# 场景化 bias 阈值（探索用，单位与 ctx.bias_20 一致）
BIAS_BOTTOM_THRESHOLD = float(os.environ.get('BIAS_BOTTOM_THRESHOLD', '0.0'))      # bottom: bias < threshold
BIAS_OPPORTUNITY_THRESHOLD = float(os.environ.get('BIAS_OPPORTUNITY_THRESHOLD', '-0.05'))  # opportunity: bias > threshold
BIAS_NORMAL_THRESHOLD = float(os.environ.get('BIAS_NORMAL_THRESHOLD', '0.05'))     # normal: bias > threshold
USE_PROFIT_RATIO_CON = os.environ.get('USE_PROFIT_RATIO_CON', '0') == '1'

# ml_rank 下限：在 opportunity / caution 场景下，剔除模型排名最靠前的 0.5% 候选，
# 避免买入过度自信/隔夜跳空变差的边际头部。经验验证 floor=0.005 在 2021-2026 回测中
# 将 opport/caution 平均收益从 1.07%/0.28% 提升到 1.21%/0.45%。
ML_RANK_FLOOR_BOTTOM = float(os.environ.get('ML_RANK_FLOOR_BOTTOM', '0.0'))
ML_RANK_FLOOR_OPPORTUNITY = float(os.environ.get('ML_RANK_FLOOR_OPPORTUNITY', '0.0'))
ML_RANK_FLOOR_NORMAL = float(os.environ.get('ML_RANK_FLOOR_NORMAL', '0.0'))
ML_RANK_FLOOR_CAUTION = float(os.environ.get('ML_RANK_FLOOR_CAUTION', '0.0'))
ML_RANK_FLOOR_RISK = float(os.environ.get('ML_RANK_FLOOR_RISK', '0.0'))


# =========================================================================
# 买入资格判断
# =========================================================================
def check_buy_eligibility_and_score(ctx, daily_env):
    """
    判断单只股票当日是否具备买入资格，并返回模型排名。

    参数:
    - ctx: 个股上下文，需包含 close/ml_rank/risk_ml_rank/bias_20/profit_ratio/
           amount_ma20/atr_ratio/concentration_70/indicator 等属性。
    - daily_env: 每日市场环境字典，包含 primary_scenario/day_limit/daily_ml_threshold/
                 is_market_ok/congestion_too_high/money_supply_signal 等。

    返回:
    - is_eligible: bool
    - ml_rank: float (越低代表模型打分越高)
    - audit: dict (用于信号审计)
    """
    if len(ctx.close) < 90:
        return False, 1.0, {}

    scenario = daily_env['primary_scenario']
    day_limit = daily_env['day_limit']
    liq_limit = day_limit['amount_ma20']
    vol_limit = day_limit['atr_ratio']

    # 拦截成交额后 30% 且波动率比例后 20% 的僵尸死盘股
    is_active = (ctx.amount_ma20[-1] >= liq_limit) and (ctx.atr_ratio[-1] >= vol_limit)

    close = ctx.close[-1]
    ml_threshold = daily_env.get('daily_ml_threshold', 0.15)
    ml_rank = ctx.ml_rank[-1]
    risk_ml_rank = ctx.risk_ml_rank[-1]

    floor_map = {
        'bottom': ML_RANK_FLOOR_BOTTOM,
        'opportunity': ML_RANK_FLOOR_OPPORTUNITY,
        'normal': ML_RANK_FLOOR_NORMAL,
        'caution': ML_RANK_FLOOR_CAUTION,
        'risk': ML_RANK_FLOOR_RISK,
    }
    ml_rank_floor = floor_map.get(scenario, 0.0)

    profit_ratio_ma3 = np.mean(ctx.profit_ratio[-3:]) if len(ctx.profit_ratio) >= 3 else ctx.profit_ratio[-1]

    # 位置判断
    close_min_90d = np.min(ctx.close[-90:])
    gain_90d = (close / close_min_90d) - 1.0
    close_too_high = gain_90d > 0.50

    close_min_5d = np.min(ctx.close[-5:])
    gain_5d = (close / close_min_5d) - 1.0
    recent_surged = gain_5d > 0.10

    highest_price = np.max(ctx.close[-5:])
    drop_from_top = (close / highest_price) - 1.0
    is_crashing = drop_from_top < -0.1

    # 场景化条件（保留 profit_ratio_con 以便后续微调）
    profit_ratio_con = True
    bias_con = True
    profit_ratio_q20 = ctx.indicator('profit_ratio_q20')[-1]
    profit_ratio_q50 = ctx.indicator('profit_ratio_q50')[-1]
    current_bias = ctx.bias_20[-1]

    if MODERATE_BUSINESS_RULES:
        # 适度基线：不强制场景化 bias 条件，由模型自己学习
        profit_ratio_con = True
        bias_con = True
    else:
        if 'bottom' in scenario:
            bias_con = current_bias < BIAS_BOTTOM_THRESHOLD
        elif 'opportunity' in scenario:
            profit_ratio_con = profit_ratio_ma3 > profit_ratio_q50
            bias_con = current_bias > BIAS_OPPORTUNITY_THRESHOLD
        elif 'normal' in scenario:
            profit_ratio_con = profit_ratio_ma3 > profit_ratio_q50
            bias_con = current_bias > BIAS_NORMAL_THRESHOLD
        elif 'caution' in scenario:
            # 谨慎场景暂不做额外约束
            pass

    # 是否启用 profit_ratio_con（默认不启用，通过环境变量开启探索）
    final_profit_ratio_con = profit_ratio_con if USE_PROFIT_RATIO_CON else True

    is_chip_ready = (
        getattr(ctx, 'is_profit_ok', False)[-1] and
        ml_rank < ml_threshold and
        ml_rank >= ml_rank_floor and
        bias_con and
        final_profit_ratio_con and
        not daily_env['congestion_too_high']
    )

    # 大盘流动性环境强拦截
    money_sig = daily_env.get('money_supply_signal', 1.0)
    can_buy_in_this_market = True
    if not daily_env['is_market_ok'] or money_sig < 0.3:
        can_buy_in_this_market = False

    is_eligible = is_chip_ready and can_buy_in_this_market and is_active

    audit = {
        'date': getattr(ctx, 'dt', None),
        'symbol': getattr(ctx, 'symbol', None),
        'ml_rank': ml_rank,
        'ml_threshold': round(ml_threshold, 4),
        'entry_bias': round(current_bias, 4),
        'profit_ratio_ma3': round(profit_ratio_ma3, 4),
        'amount_ma20': round(ctx.amount_ma20[-1], 0),
        'liq_limit': round(liq_limit, 0),
        'atr_ratio': round(ctx.atr_ratio[-1], 4),
        'vol_limit': round(vol_limit, 4),
        'gain_90d': round(gain_90d, 4),
        'is_profit_ok': getattr(ctx, 'is_profit_ok', False)[-1],
        'is_active': is_active,
        'actual_liq_limit': round(liq_limit, 0),
        'money_sig': money_sig,
        'can_buy_final': can_buy_in_this_market,
        'is_chip_ready': is_chip_ready,
        'close_too_high': close_too_high,
        'is_crashing': is_crashing,
        'market_ok': daily_env['is_market_ok'],
        'is_eligible': is_eligible,
        'risk_ml_rank': risk_ml_rank,
        'primary_scenario': scenario,
        'opport_mag_z': _scalar(getattr(ctx, 'opport_mag_z', None)),
        'risk_mag': _scalar(getattr(ctx, 'risk_mag', None)),
        'close': round(close, 4),
        'atr': round(ctx.atr[-1], 4),
    }

    return is_eligible, ml_rank, audit


def _scalar(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(x[-1]) if len(x) > 0 else None
    return float(x)


# =========================================================================
# 目标仓位计算（无现金约束）
# =========================================================================
def compute_target_size(ctx, daily_env,
                        base_target_size=0.05,
                        pos_mult_weight=1.0,
                        pos_mult_bias=0.0,
                        opport_sizing_coeff=0.15,
                        opport_sizing_min=0.5,
                        opport_sizing_max=1.5,
                        trained_opport_mag_lgbm=None):
    """
    计算单只股票的目标仓位比例（0~1）。
    不包含任何现金约束，调用方自行决定是否截断。
    """
    position_multiplier = daily_env.get('position_multiplier', 1.0)
    target_size = base_target_size * (pos_mult_bias + pos_mult_weight * position_multiplier)

    # 机会幅度模型调整仓位（方案B）：直接消费原始超额收益(小数)，按 hurdle 归一，
    # 预测越超门槛仓位越大、为负则缩仓——保留绝对幅度语义，不做 z 化。
    opport_mag = getattr(ctx, 'opport_mag', None)
    if trained_opport_mag_lgbm is not None and opport_mag is not None:
        val = float(opport_mag[-1]) if isinstance(opport_mag, (list, tuple, np.ndarray)) else float(opport_mag)
        hurdle = max(float(os.environ.get('OPPORT_HURDLE', '0.02')), 1e-4)
        sizing_factor = 1.0 + opport_sizing_coeff * (val / hurdle)
        sizing_factor = max(opport_sizing_min, min(opport_sizing_max, sizing_factor))
        target_size *= sizing_factor

    return float(target_size)


# =========================================================================
# 卖出判断
# =========================================================================
def evaluate_sell_signal(ctx, daily_env, position,
                         trained_risk_mag_lgbm=None):
    """
    判断持仓是否应该卖出。

    参数:
    - ctx: 个股上下文（需包含 close/atr/ml_rank/risk_ml_rank 等时序）。
    - daily_env: 每日市场环境。
    - position: 持仓对象，需包含 bars/market_value/pnl/shares。
    - trained_risk_mag_lgbm: 可选风险幅度模型。

    返回:
    - should_sell: bool
    - sell_reason: str
    """
    close = ctx.close[-1]
    atr = ctx.atr[-1]
    bars_held = position.bars
    if bars_held < 1:
        return False, ""

    total_cost = float(position.market_value - position.pnl)
    shares = float(position.shares)
    entry_price = total_cost / shares if shares > 0 else close
    curr_pnl = float(position.pnl) / total_cost if total_cost > 0 else 0.0

    held_period_closes = ctx.close[-bars_held:]
    highest_close = np.max(held_period_closes) if len(held_period_closes) > 0 else close

    should_sell = False
    sell_reason = ""

    ml_rank = ctx.ml_rank[-1]
    risk_ml_rank = ctx.risk_ml_rank[-1]

    # --- 0. 硬性 ATR 止损 ---
    if ENABLE_HARD_ATR_STOP and close < entry_price - HARD_ATR_STOP_MULT * atr:
        return True, "Hard_ATR_Stop_Loss"

    # --- 1. 风险模型：动态安全水位 ---
    if RELAXED_EXIT_RULES:
        neg_pnl_risk_threshold = 0.07
        profit_risk_threshold = 0.03
        risk_deterioration_threshold = -0.40
        time_exit_bars = 15
        time_exit_pnl = 0.00
    else:
        neg_pnl_risk_threshold = NEG_PNL_RISK_THRESHOLD
        profit_risk_threshold = PROFIT_RISK_THRESHOLD
        risk_deterioration_threshold = RISK_DETERIORATION_THRESHOLD
        time_exit_bars = TIME_EXIT_BARS
        time_exit_pnl = TIME_EXIT_PNL

    if curr_pnl < 0:
        if risk_ml_rank < neg_pnl_risk_threshold:
            should_sell = True
            sell_reason = "Negative_PnL_Risk_Preemption"
    else:
        if risk_ml_rank < profit_risk_threshold:
            should_sell = True
            sell_reason = "Profit_Protection_Risk"

    # --- 2. 风险模型：边际恶化审计 ---
    if not should_sell and bars_held >= 2:
        risk_change = risk_ml_rank - ctx.risk_ml_rank[-2]
        if risk_change < risk_deterioration_threshold:
            should_sell = True
            sell_reason = "Risk_Sudden_Deterioration"

        if not should_sell:
            if bars_held >= time_exit_bars and curr_pnl < time_exit_pnl:
                should_sell = True
                sell_reason = "Time_Efficiency_Exit"

    # --- 3. 移动止盈 ---
    if not should_sell and curr_pnl > TRAILING_PROFIT_THRESHOLD:
        mult = 2.5 if curr_pnl > 0.10 else 1.8
        if ml_rank < 0.01:
            mult = 3
        if close < highest_close - mult * atr:
            should_sell = True
            sell_reason = "Trailing_Stop_Profit"

    # --- 4. 风险幅度模型补丁 ---
    if not should_sell and trained_risk_mag_lgbm is not None:
        risk_mag = getattr(ctx, 'risk_mag', None)
        if risk_mag is not None:
            risk_mag_val = risk_mag[-1] if isinstance(risk_mag, (list, np.ndarray)) else risk_mag
            if risk_mag_val < RISK_MAG_SELL_THRESHOLD:
                should_sell = True
                sell_reason = "Risk_Mag_Exit"

    # --- 5. 极端场景补丁 ---
    if not should_sell and daily_env['primary_scenario'] == 'risk':
        if ml_rank > 0.05:
            should_sell = True
            sell_reason = "Market_Risk_Clearance"

    # --- 6. 熔断持续性持仓上限 (试点, env 门控默认关) ---
    # 与模型零交叉设计: 仅作用于 Market_Risk_Clearance 的豁免区 (ml_rank<=0.05,
    # 即模型仍看好的持仓) —— 模型分本身不参与触发, 触发依据是熔断的"持续性"
    # (连续 risk_run_days >= N) 与亏损状态, 补足模型对系统性状态持续度的盲区。
    if (not should_sell and os.environ.get('RISK_HOLD_CAP') == '1'
            and daily_env['primary_scenario'] == 'risk'):
        cap_n = int(os.environ.get('RISK_HOLD_CAP_N', '3'))
        if int(daily_env.get('risk_run_days', 0)) >= cap_n and ml_rank <= 0.05 and curr_pnl < 0:
            should_sell = True
            sell_reason = "Risk_Regime_Hold_Cap"

    return should_sell, sell_reason
