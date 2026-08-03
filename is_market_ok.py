import pandas as pd
import numpy as np

# ==========================================
# 核心逻辑：四象限检测 (严格 1 日延迟、全火力武器系统、安全闭环版)
# ==========================================

def detect_bottom_divergence(zzqz_row, breadth_row, available_breadth, total_stocks):
    """
    【底背离检测 - 抄底信号检测端】
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


def detect_risk_avoidance(zzqz_row, breadth_row, total_stocks):
    """
    【风险回避检测】（前置安全熔断红线）
    """
    risk_signals = []
    
    # 绝对流动性崩溃硬熔断线：大盘有超过 18% 个股创 20 日新低
    ABSOLUTE_CRASH_THRESHOLD = total_stocks * 0.18 
    if breadth_row['low20'] > ABSOLUTE_CRASH_THRESHOLD:
        risk_signals.append(f"绝对流动性崩溃_熔断(>{int(ABSOLUTE_CRASH_THRESHOLD)}家)")
        
    return risk_signals


def detect_opportunity_environment(zzqz_row, breadth_row, available_breadth, total_stocks):
    """
    【机会环境检测】
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


def detect_caution_environment(zzqz_row, breadth_row, total_stocks):
    """
    【谨慎环境检测】
    """
    caution_signals = []
    
    price_high = zzqz_row['close'] >= zzqz_row['close_prev'] * 0.98 if 'close_prev' in zzqz_row else True
    breadth_weak = breadth_row['high20'] < breadth_row['high20_q40']
    if price_high and breadth_weak:
        caution_signals.append("高位动能枯竭(<Q40)")
        
    if zzqz_row['close'] < zzqz_row['close_ma60'] and breadth_row['low20'] > breadth_row['low20_q60']: 
        caution_signals.append("系统性趋势走坏")

    return caution_signals


# ==========================================
# 决策树总控：防死锁、零未来函数、严格对齐闭环版
# ==========================================

def scenario_based_market_judgment(date, zzqz_df, breadth_df, total_stocks=5000):
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

    # 获取分项底层特征信号
    bottom_signals = detect_bottom_divergence(zzqz_row, breadth_row, available_breadth, total_stocks)
    risk_signals = detect_risk_avoidance(zzqz_row, breadth_row, total_stocks)
    opportunity_signals = detect_opportunity_environment(zzqz_row, breadth_row, available_breadth, total_stocks)
    caution_signals = detect_caution_environment(zzqz_row, breadth_row, total_stocks)

    is_above_ma60 = zzqz_row['close'] >= zzqz_row['close_ma60']

    # --- 柔性“防飞刀安全带”（加入低基数安全垫） ---
    is_falling_knife = False
    if len(available_breadth) >= 2:
        mkt_daily_drop = (zzqz_row['close'] / zzqz_row['close_prev'] - 1.0) < -0.015  # 单日大跌 [1]
        
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
            'decision_reason': f"【铁血熔断】触发大盘前置高危指标：{risk_signals[0]}" [1]
        }
        
    # 优先级 2：安全抄底判定（底背离大面积释放，修复无 return 的漏洞） [1]
    elif len(bottom_signals) > 0:
        if is_falling_knife:
            return {
                'is_market_ok': True, 
                'position_multiplier': 0.7, 
                'primary_scenario': 'caution', 
                'decision_reason': "【抄底拦截】检测到大盘处于急跌飞刀阶段，强制控仓防御"
            }
        else:
            return {
                'is_market_ok': True, 
                'position_multiplier': 1.2, 
                'primary_scenario': 'bottom', 
                'decision_reason': f"【安全抄底】触发底背离：{bottom_signals[0]}" [1]
            }
            
    # 优先级 3：主升/右侧进攻机会（修复无 return 的漏洞，彻底释放主升武器）
    elif len(opportunity_signals) > 0:
        return {
            'is_market_ok': True, 
            'position_multiplier': 0.8 if is_above_ma60 else 0.6,
            'primary_scenario': 'opportunity', 
            'decision_reason': f"【主升进攻】触发：{opportunity_signals[0]}"
        }
        
    # 优先级 4：均线下方防御拦截（2418 笔大造血的核心基石） [1]
    elif not is_above_ma60:
        return {
            'is_market_ok': True, 
            'position_multiplier': 0.7, 
            'primary_scenario': 'caution', 
            'decision_reason': "【趋势偏弱】MA60下方，释放LGBM模型防御选股空间" [1]
        }
            
    # 优先级 5：谨慎防守场景
    elif len(caution_signals) > 0:
        return {
            'is_market_ok': True, 
            'position_multiplier': 0.7, 
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