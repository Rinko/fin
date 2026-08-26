#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — 全系统唯一 CLI 入口

用法：
    python run.py prod      [main.py 其余参数...]     # 每日生产信号
    python run.py backtest  [--days N | main 参数...]  # 组合回测（默认截至 BASELINE_END）
    python run.py signals   [generate_signals 参数...] # 全量候选导出（轻量）
    python run.py bench     [simple_rank_benchmark 参数...]
    python run.py audit <name> [参数...]               # name ∈ trades/entry/risk/magnitude/signal_level/scenario/challenger

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
        'scenario':     'audit/check_scenario.py',
        'challenger':   'audit/scenario_challenger.py',
        'exits':        'exit_signal.py',
        'daily':        'daily_signal.py',
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
    # KEY=VALUE 透传：业务线实验覆盖（apply 之后生效）
    import re as _re
    kept=[]
    for x in rest:
        m=_re.match(r'^([A-Z][A-Z0-9_]+)=(.+)$',x)
        if m: os.environ[m.group(1)]=m.group(2); print(f'[run] 覆盖 {m.group(1)}={m.group(2)}')
        else: kept.append(x)
    rest=kept
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
    elif line == 'daily':
        # 编排器：买入=权威管线(padded generate_signals, 含全部闸门)；卖出=exit_signal 规则链
        import argparse as _ap
        pa=_ap.ArgumentParser(); pa.add_argument('--holdings',default=None)
        pa.add_argument('--start',default=None); pa.add_argument('--end',
            default=pd.Timestamp.now().strftime('%Y-%m-%d') if (pd:=__import__('pandas')) else None)
        kn,pad=pa.parse_known_args(rest)
        passthrough=pad
        end=kn.end or __import__('pandas').Timestamp.now().strftime('%Y-%m-%d')
        _w=int(os.environ.get('DAILY_WARMUP_BARS','0')) or None
        start=kn.start or __import__('config').padded_start(end,warmup=_w)
        outdir=os.environ.get('DAILY_OUTDIR','external_data/daily')
        gs_argv=['generate_signals.py','--start',start,'--end',end,'--out',outdir]+passthrough
        print(f"[daily] 权威管线出候选: {start}..{end} -> {outdir}")
        sys.argv=gs_argv; runpy.run_path('generate_signals.py',run_name='__main__')
        import pandas as _pd
        sigf=os.path.join(outdir,'all_signals.csv')
        if not os.path.exists(sigf): raise SystemExit('[daily] 无候选文件')
        s=_pd.read_csv(sigf); s['date']=_pd.to_datetime(s['date'])
        last=s.date.max(); day=s[s.date==last].copy()
        qmap={'bottom':int(os.environ.get('BUY_QUOTA_BOTTOM','5')),
              'opportunity':int(os.environ.get('BUY_QUOTA_OPPORTUNITY','5')),
              'normal':int(os.environ.get('BUY_QUOTA_NORMAL','2')),
              'caution':int(os.environ.get('BUY_QUOTA_CAUTION','1')),
              'risk':int(os.environ.get('BUY_QUOTA_RISK','0'))}
        picks=[]
        for sc,g in day.groupby('primary_scenario'):
            picks.append(g.nsmallest(qmap.get(sc,5),'ml_rank'))
        buys=_pd.concat(picks).sort_values('ml_rank')[['action'] ] if False else None
        buys=_pd.concat(picks).sort_values('ml_rank')
        buys.insert(0,'action','BUY')
        tag=_pd.Timestamp(last).strftime('%Y%m%d')
        os.makedirs(outdir,exist_ok=True)
        buys[['action','date','symbol','close','ml_rank','primary_scenario']].to_csv(
            f'{outdir}/{tag}_buy_signals.csv',index=False)
        print(f"[daily] 买入 {len(buys)} 只 -> {tag}_buy_signals.csv")
        if kn.holdings:
            ex='exit_signal.py'
            sys.argv=[ex,'--holdings',kn.holdings,'--start',start,'--end',str(last.date())]
            runpy.run_path(ex,run_name='__main__')
        else:
            print('[daily] 未提供 --holdings，跳过离场判定')
        ci='check_invariants.py'; sf=f"{outdir}/{tag}_scores.csv"
        sarg=['--scores',sf]
        slf=f"{outdir}/{tag}_sell_signals.csv"
        if os.path.exists(slf): sarg+=['--sell',slf]
        sys.argv=[ci]+sarg; runpy.run_path(ci,run_name='__main__')
    elif line == 'audit':
        _dispatch_audit(rest)


if __name__ == '__main__':
    main()
