import pandas as pd
import numpy as np

# ==========================================
# 阈值开关：False = 固定阈值 (历史校准，回测更优) / True = 动态 v5 (双重确认，泛化更稳)
# 由 REPORT_breadth_threshold_dynamic.md 五章结论支持，二者可并存切换。
# ==========================================
USE_DYNAMIC_THRESHOLD = True


# ==========================================
# 核心逻辑：四象限检测 (严格 1 日延迟、全火力武器系统、安全闭环版)
# ==========================================

# ---------------- 动态 v5 阈值版 (默认) ----------------

def detect_bottom_divergence_dynamic(zzqz_row, breadth_row, available_breadth, _total_stocks=None):
    """
    【底背离检测 - 抄底信号检测端】(动态 v5)
    """
    divergence_signals = []
    curr_price = zzqz_row['close']

    # 广度指标（Day T 收盘数据）
    breadth_recovery_zone_20 = breadth_row['low20'] < breadth_row['low20_q30']
    breadth_recovery_zone_60 = breadth_row['low60'] < breadth_row['low60_q30']
    momentum_exhausted = breadth_row['low_v'] < 0

    # 稳定点火判定 (动态阈值: q10低门槛保持易触发, 与原固定0.05的~89%触发率语义一致)
    avg_high_5d = available_breadth['high10'].tail(5).mean()
    spark_floor = max(breadth_row.get('high10_ratio_q10', 0.05), 0.02)
    has_spark = (breadth_row['high10'] > avg_high_5d * 1.1) or (breadth_row['high10_ratio'] > spark_floor)

    # 1. 20日底背离
    is_20d_div = False
    price_low_20 = curr_price <= zzqz_row['close_q30_w20']
    if price_low_20 and breadth_recovery_zone_20 and momentum_exhausted and has_spark:
        is_20d_div = True
        divergence_signals.append("强20日动态底背离")

    # 2. 60日底背离
    is_60d_div = False
    price_low_60 = curr_price <= zzqz_row['close_q30_w60']
    if price_low_60 and breadth_recovery_zone_60 and momentum_exhausted and has_spark:
        is_60d_div = True
        divergence_signals.append("强60日动态底背离")

    # 3. 极端超跌V型修复 (动态阈值: q80高分位保持难触发+0.20绝对上限, 匹配原0.20的q80语义)
    panic_floor = max(breadth_row.get('low20_ratio_q80', 0.20), 0.10)
    panic_high_level = breadth_row['low20_ratio'] > min(0.20, panic_floor)
    new_high_power = breadth_row['high20'] > available_breadth['high20'].tail(5).mean() * 1.2
    breadth_turn_confirmed = breadth_row['low20'] < max(available_breadth['low20'].tail(3)) * 0.5

    if panic_high_level and momentum_exhausted and new_high_power and breadth_turn_confirmed:
        divergence_signals.append("极端压抑V型修复")

    return divergence_signals


def detect_risk_avoidance_dynamic(zzqz_row, breadth_row, _total_stocks=None):
    """
    【风险回避检测】（前置安全熔断红线，动态 v5）
    """
    risk_signals = []

    # 绝对流动性崩溃硬熔断线 (双重确认: 绝对18%线 + 近期q80高位, 只熔断真危机)
    meltdown_floor = max(breadth_row.get('low20_ratio_q80', 0.18), 0.18)
    if breadth_row['low20_ratio'] > meltdown_floor:
        risk_signals.append("绝对流动性崩溃_熔断(双重确认:>18%且近期高位)")

    return risk_signals


def detect_opportunity_environment_dynamic(zzqz_row, breadth_row, available_breadth, _total_stocks=None):
    """
    【机会环境检测】（动态 v5）
    """
    opportunity_signals = []

    # 1. 新高动能爆发 (牛市爆发点)
    limit_q85 = max(breadth_row['high_ratio_q85'], 0.05) if 'high_ratio_q85' in breadth_row else 0.05
    if breadth_row['high_ratio'] > limit_q85 and breadth_row['high_v'] > 0:
        opportunity_signals.append("新高动能爆发(>Q85)")

    # 2. 趋势广度共振
    price_trend = zzqz_row['close'] > zzqz_row['close_ma10']
    breadth_healthy = breadth_row['high20'] > breadth_row['high20_q50']
    if price_trend and breadth_healthy:
        opportunity_signals.append("趋势广度共振(>Q50)")

    # 3. 隐性广度走强 (动态阈值: q10低门槛保持易触发, 匹配原0.03的~90%触发率)
    index_quiet = abs(zzqz_row['close'] / zzqz_row['close_prev'] - 1) < 0.005 if 'close_prev' in zzqz_row else False
    breadth_floor = max(breadth_row.get('high20_ratio_q10', 0.03), 0.02)
    if index_quiet and breadth_row['high_v'] > 0 and breadth_row['high20_ratio'] > breadth_floor:
        opportunity_signals.append("隐性广度走强")

    # 4. 稳健市右侧点火
    limit_q30 = max(breadth_row['low20_q30'], 2.0) if 'low20_q30' in breadth_row else 2.0
    breadth_stable = breadth_row['low20'] <= limit_q30  # 解决 low20_q30=0 导致的零值死锁
    momentum_exhausted = breadth_row['low_v'] <= 0

    avg_high_5d = available_breadth['high10'].tail(5).mean()
    limit_spark = max(avg_high_5d * 1.1, 2.0) if not np.isnan(avg_high_5d) else 2.0
    spark_floor = max(breadth_row.get('high10_ratio_q10', 0.04), 0.02)
    has_spark = (breadth_row['high10'] > limit_spark) or (breadth_row['high10_ratio'] > spark_floor)

    if breadth_stable and momentum_exhausted and has_spark:
        opportunity_signals.append("稳健市右侧点火")

    return opportunity_signals


def detect_caution_environment_dynamic(zzqz_row, breadth_row, _total_stocks=None):
    """
    【谨慎环境检测】（动态 v5）
    注: 原"系统性趋势走坏"信号要求 close<MA60, 但决策树 P4(MA60下方)优先级
    更高、永远先行触发, 该分支自上线以来从未生效(死代码), 已于 2026-08-26 清理。
    """
    caution_signals = []

    price_high = zzqz_row['close'] >= zzqz_row['close_prev'] * 0.98 if 'close_prev' in zzqz_row else True
    breadth_weak = breadth_row['high20'] < breadth_row['high20_q40']
    if price_high and breadth_weak:
        caution_signals.append("高位动能枯竭(<Q40)")

    return caution_signals


# ---------------- 固定阈值版 (历史原生) ----------------

def detect_bottom_divergence_fixed(zzqz_row, breadth_row, available_breadth, total_stocks):
    """
    【底背离检测 - 抄底信号检测端】（固定阈值原版）
    """
    divergence_signals = []
    curr_price = zzqz_row['close']

    # 广度指标（Day T 收盘数据）
    breadth_recovery_zone_20 = breadth_row['low20'] < breadth_row['low20_q30']
    breadth_recovery_zone_60 = breadth_row['low60'] < breadth_row['low60_q30']
    momentum_exhausted = breadth_row['low_v'] < 0

    # 稳定点火判定
    avg_high_5d = available_breadth['high10'].tail(5).mean()
    has_spark = (breadth_row['high10'] > avg_high_5d * 1.1) or (breadth_row['high10'] > total_stocks * 0.05)

    # 1. 20日底背离
    is_20d_div = False
    price_low_20 = curr_price <= zzqz_row['close_q30_w20']
    if price_low_20 and breadth_recovery_zone_20 and momentum_exhausted and has_spark:
        is_20d_div = True
        divergence_signals.append("强20日动态底背离")

    # 2. 60日底背离
    is_60d_div = False
    price_low_60 = curr_price <= zzqz_row['close_q30_w60']
    if price_low_60 and breadth_recovery_zone_60 and momentum_exhausted and has_spark:
        is_60d_div = True
        divergence_signals.append("强60日动态底背离")

    # 3. 极端超跌V型修复
    panic_high_level = breadth_row['low20'] > total_stocks * 0.20
    new_high_power = breadth_row['high20'] > available_breadth['high20'].tail(5).mean() * 1.2
    breadth_turn_confirmed = breadth_row['low20'] < max(available_breadth['low20'].tail(3)) * 0.5

    if panic_high_level and momentum_exhausted and new_high_power and breadth_turn_confirmed:
        divergence_signals.append("极端压抑V型修复")

    return divergence_signals


def detect_risk_avoidance_fixed(zzqz_row, breadth_row, total_stocks):
    """
    【风险回避检测】（前置安全熔断红线，固定阈值原版）
    """
    risk_signals = []

    # 绝对流动性崩溃硬熔断线：大盘有超过 18% 个股创 20 日新低
    ABSOLUTE_CRASH_THRESHOLD = total_stocks * 0.18
    if breadth_row['low20'] > ABSOLUTE_CRASH_THRESHOLD:
        risk_signals.append(f"绝对流动性崩溃_熔断(>{int(ABSOLUTE_CRASH_THRESHOLD)}家)")

    return risk_signals


def detect_opportunity_environment_fixed(zzqz_row, breadth_row, available_breadth, total_stocks):
    """
    【机会环境检测】（固定阈值原版）
    """
    opportunity_signals = []

    # 1. 新高动能爆发 (牛市爆发点)
    limit_q85 = max(breadth_row['high_ratio_q85'], 0.05) if 'high_ratio_q85' in breadth_row else 0.05
    if breadth_row['high_ratio'] > limit_q85 and breadth_row['high_v'] > 0:
        opportunity_signals.append("新高动能爆发(>Q85)")

    # 2. 趋势广度共振
    price_trend = zzqz_row['close'] > zzqz_row['close_ma10']
    breadth_healthy = breadth_row['high20'] > breadth_row['high20_q50']
    if price_trend and breadth_healthy:
        opportunity_signals.append("趋势广度共振(>Q50)")

    # 3. 隐性广度走强
    index_quiet = abs(zzqz_row['close'] / zzqz_row['close_prev'] - 1) < 0.005 if 'close_prev' in zzqz_row else False
    limit_high20 = max(total_stocks * 0.03, 3.0)
    if index_quiet and breadth_row['high_v'] > 0 and breadth_row['high20'] > limit_high20:
        opportunity_signals.append("隐性广度走强")

    # 4. 稳健市右侧点火
    limit_q30 = max(breadth_row['low20_q30'], 2.0) if 'low20_q30' in breadth_row else 2.0
    breadth_stable = breadth_row['low20'] <= limit_q30  # 解决 low20_q30=0 导致的零值死锁
    momentum_exhausted = breadth_row['low_v'] <= 0

    avg_high_5d = available_breadth['high10'].tail(5).mean()
    limit_spark = max(avg_high_5d * 1.1, 2.0) if not np.isnan(avg_high_5d) else 2.0
    has_spark = (breadth_row['high10'] > limit_spark) or (breadth_row['high10'] > total_stocks * 0.04)

    if breadth_stable and momentum_exhausted and has_spark:
        opportunity_signals.append("稳健市右侧点火")

    return opportunity_signals


def detect_caution_environment_fixed(zzqz_row, breadth_row, total_stocks):
    """
    【谨慎环境检测】（固定阈值原版）
    注: "系统性趋势走坏"为死代码(被 P4 MA60分支遮蔽), 已清理, 见动态版注释。
    """
    caution_signals = []

    price_high = zzqz_row['close'] >= zzqz_row['close_prev'] * 0.98 if 'close_prev' in zzqz_row else True
    breadth_weak = breadth_row['high20'] < breadth_row['high20_q40']
    if price_high and breadth_weak:
        caution_signals.append("高位动能枯竭(<Q40)")

    return caution_signals


# ==========================================
# 决策树总控：防死锁、零未来函数、严格对齐闭环版
# ==========================================

def scenario_based_market_judgment(date, zzqz_df, breadth_df, total_stocks=None, use_dynamic_threshold=None):
    """
    预热双阈值开关：use_dynamic_threshold 优先于模块级 USE_DYNAMIC_THRESHOLD。
    若传 None 则使用模块默认开关。fixed 模式依赖 total_stocks 绝对计数。
    """
    if use_dynamic_threshold is None:
        use_dynamic_threshold = USE_DYNAMIC_THRESHOLD

    available_zzqz = zzqz_df[zzqz_df.index <= date]
    available_breadth = breadth_df[breadth_df.index <= date]

    if len(available_zzqz) < 5 or len(available_breadth) < 5:
        return {
            'is_market_ok': True, 'position_multiplier': 0.5,
            'primary_scenario': 'normal', 'decision_reason': "冷启动期默认"
        }

    zzqz_row = available_zzqz.iloc[-1].copy()
    breadth_row = available_breadth.iloc[-1].copy()
    zzqz_row['close_prev'] = available_zzqz['close'].iloc[-2]

    # 阈值模式分支
    if use_dynamic_threshold:
        mode_tag = "动态v5"
        if total_stocks is None:
            total_stocks = 0
        bottom_signals = detect_bottom_divergence_dynamic(zzqz_row, breadth_row, available_breadth, total_stocks)
        risk_signals = detect_risk_avoidance_dynamic(zzqz_row, breadth_row, total_stocks)
        opportunity_signals = detect_opportunity_environment_dynamic(zzqz_row, breadth_row, available_breadth, total_stocks)
        caution_signals = detect_caution_environment_dynamic(zzqz_row, breadth_row, total_stocks)
    else:
        mode_tag = "固定阈值"
        if total_stocks is None:
            total_stocks = 5000  # 固定版 fallback
        bottom_signals = detect_bottom_divergence_fixed(zzqz_row, breadth_row, available_breadth, total_stocks)
        risk_signals = detect_risk_avoidance_fixed(zzqz_row, breadth_row, total_stocks)
        opportunity_signals = detect_opportunity_environment_fixed(zzqz_row, breadth_row, available_breadth, total_stocks)
        caution_signals = detect_caution_environment_fixed(zzqz_row, breadth_row, total_stocks)

    is_above_ma60 = zzqz_row['close'] >= zzqz_row['close_ma60']

    # --- 柔性"防飞刀安全带"（加入低基数安全垫） ---
    is_falling_knife = False
    if len(available_breadth) >= 2:
        mkt_daily_drop = (zzqz_row['close'] / zzqz_row['close_prev'] - 1.0) < -0.015  # 单日大跌 [1]

        if use_dynamic_threshold:
            # 踩踏判定 (双重确认: 绝对2%线 + 近期q20, 减平静期噪声)
            knife_floor = max(breadth_row.get('low20_ratio_q20', 0.02), 0.02)
            low20_surge = (breadth_row['low20_ratio'] > knife_floor) and (breadth_row['low20'] > available_breadth['low20'].iloc[-2] * 1.25)
        else:
            # 优化：只有当新低个股占比大于 2% 全市场（真正有踩踏苗头）且增幅超 25% 时，才启动踩踏判定
            low20_surge = (breadth_row['low20'] > total_stocks * 0.02) and (breadth_row['low20'] > available_breadth['low20'].iloc[-2] * 1.25)
        is_falling_knife = mkt_daily_drop or low20_surge

    # =========================================================================
    # 【决策树金标准】：前置熔断避险，安全抄底/主升完全释放（修复不阻断 bug）
    # =========================================================================

    # 优先级 1：绝对系统风险（硬熔断，拦截 2024 年 1 月踩踏） [1]
    if len(risk_signals) > 0:
        return {
            'is_market_ok': False,
            'position_multiplier': 0.0,
            'primary_scenario': 'risk',
            'decision_reason': f"【铁血熔断】触发大盘前置高危指标：{risk_signals[0]}"
        }

    # 优先级 2：安全抄底判定（底背离大面积释放，修复无 return 的漏洞） [1]
    elif len(bottom_signals) > 0:
        if is_falling_knife:
            return {
                'is_market_ok': True,
                'position_multiplier': 0.4,
                'primary_scenario': 'caution',
                'decision_reason': "【抄底拦截】检测到大盘处于急跌飞刀阶段，强制控仓防御"
            }
        else:
            return {
                'is_market_ok': True,
                'position_multiplier': 1.2,
                'primary_scenario': 'bottom',
                'decision_reason': f"【安全抄底】触发底背离：{bottom_signals[0]}"
            }

    # 优先级 3：主升/右侧进攻机会（修复无 return 的漏洞，彻底释放主升武器）
    elif len(opportunity_signals) > 0:
        return {
            'is_market_ok': True,
            'position_multiplier': 0.8 if is_above_ma60 else 0.6,
            'primary_scenario': 'opportunity',
            'decision_reason': f"【主升进攻】触发：{opportunity_signals[0]}"
        }

    # 优先级 4：均线下方 = 修复型机会环境（2026-08-26 语义归位, V4 影子仲裁中性验证）
    # 实证: 该子类 fwd20 与选股质量长期优于 normal、近窗口优于 opportunity 本尊;
    # 原先"名为谨慎、实为机会"的错配是历史共演结果, 现按真实语义归入 opportunity。
    elif not is_above_ma60:
        return {
            'is_market_ok': True,
            'position_multiplier': 0.8,
            'primary_scenario': 'opportunity',
            'decision_reason': "【修复型机会】MA60下方但广度环境成立，释放防御选股空间"
        }

    # 优先级 5：谨慎防守场景（真防守: 高位动能枯竭; 参数已降至 normal 之下）
    elif len(caution_signals) > 0:
        return {
            'is_market_ok': True,
            'position_multiplier': 0.4,
            'primary_scenario': 'caution',
            'decision_reason': f"【高位谨慎】触发：{caution_signals[0]}"
        }

    # 优先级 6：常态 fallback
    return {
        'is_market_ok': True,
        'position_multiplier': 0.5,
        'primary_scenario': 'normal',
        'decision_reason': "正常波动"
    }