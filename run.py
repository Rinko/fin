#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — 全系统唯一 CLI 入口

用法：
    python run.py prod      [main.py 其余参数...]     # 每日生产信号
    python run.py backtest  [--days N | main 参数...]  # 组合回测（默认截至 BASELINE_END）
    python run.py signals   [generate_signals 参数...] # 全量候选导出（轻量）
    python run.py bench     [simple_rank_benchmark 参数...]
    python run.py audit <name> [参数...]               # name ∈ trades/entry/risk/magnitude/signal_level

约定：
1. 先 config.apply(line) 盖章环境，再加载业务模块（import 顺序不可颠倒）。
2. 本文件不做任何策略决策，只做分发。
"""
import os, sys, runpy

LINES = ('prod', 'backtest', 'signals', 'bench', 'audit')

def _dispatch_audit(rest):
    mapping = {
        'trades':       'check_trades.py',
        'entry':        'audit/check_model_entry.py',
        'risk':         'audit/check_model_risk.py',
        'magnitude':    'audit/check_magnitude_model.py',
        'signal_level': 'audit_signal_level.py',
        'exits':        'exit_signal.py',
    }
    if not rest:
        raise SystemExit(f"[run] audit 需要名称: {sorted(mapping)}")
    name, args = rest[0], rest[1:]
    path = mapping.get(name)
    if path is None or not os.path.exists(path):
        raise SystemExit(f"[run] 未知审计 '{name}' 或脚本缺失")
    sys.argv = [path] + args
    runpy.run_path(path, run_name='__main__')


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        raise SystemExit(0 if len(sys.argv) > 1 else 1)
    line = sys.argv[1]
    rest = sys.argv[2:]

    import config
    os.environ['RUN_LINE']=line
    diff = config.apply(line)
    print(f"[run] 业务线={line} | 环境变更 {len(diff)} 项: {', '.join(diff[:8])}{' ...' if len(diff)>8 else ''}")

    if line == 'prod':
        sys.argv = ['main.py'] + rest
        runpy.run_path('main.py', run_name='__main__')
    elif line == 'backtest':
        sys.argv = ['main.py', '--full'] + rest
        runpy.run_path('main.py', run_name='__main__')
    elif line == 'signals':
        sys.argv = ['generate_signals.py'] + rest
        runpy.run_path('generate_signals.py', run_name='__main__')
    elif line == 'bench':
        p = 'external_data/explore_night/signal_level_backtest_20260818/simple_rank_benchmark.py'
        sys.argv = [p] + rest
        runpy.run_path(p, run_name='__main__')
    elif line == 'audit':
        _dispatch_audit(rest)


if __name__ == '__main__':
    main()
