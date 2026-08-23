#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_signal.py — 每日轻量信号（买入+卖出 一条命令）

双通道打分（无 PyBroker）→ 核心过滤 → 场景化配额选买 → 持仓注入跑卖出规则链。
输出: external_data/daily/<YYYYMMDD>_buy_signals.csv / _sell_signals.csv / _scores.csv

用法:
  python daily_signal.py [--holdings pos.csv] [--scenario normal]
  # 持仓CSV列: symbol,entry_date,entry_price[,shares]
说明:
  - 大盘状态闸门与场景bias条件不在本轻量链内（以 run.py prod 重路径为最终权威），
    可用 --scenario 手工指定当日场景；--allow-market-closed 1 可绕过空仓判定。
"""
import os,sys,argparse,logging
from types import SimpleNamespace
import numpy as np,pandas as pd,joblib
_here=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,_here)
import co_compute
co_compute.FeatureConfig.MKT_FEATURES=['mkt_macro_regime']
import signal_engine
from local_data_cache import LocalDataCache
from screen import basic_screen
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

def _mkt():
    mc=pd.read_parquet('market_context_cache.parquet').set_index('date')
    mc.index=pd.to_datetime(mc.index)
    need=list(co_compute.FeatureConfig.MKT_FEATURES)
    if not all(c in mc.columns for c in need):
        r=mc.reset_index()
        mc=r.merge(co_compute.build_market_pca_table(r,min_periods=60),on='date',how='left').set_index('date')
        mc.index=pd.to_datetime(mc.index)
    return mc

def _channel(ldc,syms,mc,fetch_start,end,smooth,biz,model,pred):
    dfs=[]
    for i,s in enumerate(syms):
        try:
            df=ldc.get_stock_data(s,fetch_start,end,adjust='qfq',mode=2)
            if df.empty or len(df)<60:continue
            df['date']=pd.to_datetime(df['date'])
            df['vwap']=(df['high']+df['low']+2*df['close'])/4.0
            df=co_compute.compute_individual_indicators(df,mc,use_smooth=smooth)
            dfs.append(df)
            if i%1000==0:logging.info(f'{i}...')
        except Exception as e:logging.warning(f'{s}:{e}')
    g=pd.concat(dfs).reset_index(drop=True);del dfs
    g=g[g['date']<=pd.to_datetime(end)]
    g=co_compute.apply_standardization(g,features=biz)
    for c in co_compute.FeatureConfig.MKT_FEATURES:
        g[c]=g['date'].map(mc[c]).ffill()
    pc=g.groupby('symbol',sort=False)['close'].shift(1)
    tr=np.maximum(g['high']-g['low'],np.maximum((g['high']-pc).abs(),(g['low']-pc).abs()))
    g['_atr']=tr.groupby(g['symbol'],sort=False).transform(lambda s:s.rolling(14,min_periods=1).mean())
    feats=[f for f in model['features'] if f in g.columns]
    miss=len(model['features'])-len(feats)
    g[pred]=model['model'].predict(g[feats]) if not miss else np.nan
    logging.info(f'{pred}: 特征{len(feats)}/{len(model["features"])} 缺{miss}')
    return g[['date','symbol','close','_atr',pred]],miss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fetch-start',default=(pd.Timestamp.now()-pd.Timedelta(days=420)).strftime('%Y-%m-%d'))
    ap.add_argument('--end',default=pd.Timestamp.now().strftime('%Y-%m-%d'))
    ap.add_argument('--holdings',default=None,help='持仓CSV: symbol,entry_date,entry_price[,shares]')
    ap.add_argument('--scenario',default='normal')
    ap.add_argument('--allow-market-closed',default='1')
    ap.add_argument('--rank-threshold',default='0.03')
    a=ap.parse_args()
    ep=joblib.load('chip_accumulation_v6_g_pca1_z.pkl');rp=joblib.load('chip_risk_model_v1_g_pca1_z.pkl')
    rm=joblib.load(os.environ.get('RISKMAG_PKL','chip_risk_magnitude_for_g.pkl'))
    mc=_mkt();syms=basic_screen();ldc=LocalDataCache(cache_dir='./stock_data_cache')

    logging.info('通道1 平滑(入场)...');S,m1=_channel(ldc,syms,mc,a.fetch_start,a.end,True,co_compute.FeatureConfig.BIZ_FEATURES,ep,'raw_ml_score')
    logging.info('通道2 原始(风控+幅度)...');R,m2=_channel(ldc,syms,mc,a.fetch_start,a.end,False,co_compute.FeatureConfig.BIZ_RISK_FEATURES,rp,'risk_ml_score')
    R=R.drop_duplicates(subset=['date','symbol'],keep='last')

    last=S.date.max()
    Sd=S[S.date==last].copy();Rd=R[R.date==lastR].copy() if (lastR:=R.date.max())==last else R.sort_values('date').groupby('symbol').tail(1)
    Sd['ml_rank']=Sd.raw_ml_score.rank(pct=True,ascending=False)
    Rd['risk_ml_rank']=Rd.risk_ml_score.rank(pct=True,ascending=True)

    fin=pd.read_csv('financial_reports_all.csv',dtype={'股票代码':str},
                    usecols=['股票代码','报告日期','净利润-净利润','净利润-同比增长','每股收益'],
                    parse_dates=['报告日期'])
    fin=fin[(fin['净利润-净利润']>0)&(fin['净利润-同比增长']>0)&(fin['每股收益']>0)]
    fin=fin.sort_values('报告日期').drop_duplicates(subset=['股票代码','报告日期'],keep='last')\
           .rename(columns={'股票代码':'symbol','报告日期':'date'}).assign(is_profit_ok=True)
    Sd=Sd.sort_values('date')
    Sd=pd.merge_asof(Sd,fin[['symbol','date','is_profit_ok']],on='date',by='symbol',direction='backward')
    Sd['is_profit_ok']=Sd['is_profit_ok'].fillna(False)

    liq_q=Sd['amount_ma20'].quantile(0.30) if 'amount_ma20' in Sd else None
    Sd['buyable']=(Sd['close']>2)&Sd['is_profit_ok']
    if 'amount_ma20' in Sd.columns:
        q_a=Sd['amount_ma20'].quantile(0.30);q_v=Sd['_atr'].quantile(0.20)
        Sd['buyable']&=(Sd['amount_ma20']>=q_a)&(Sd['_atr']>=q_v)
    if str(a.allow_market_closed)!='1':
        Sd['buyable']&=False  # 占位：大盘闸门走重路径权威
    thr=float(a.rank_threshold)
    cand=Sd[Sd.buyable&(Sd.ml_rank<thr)].sort_values('ml_rank')

    qmap={'bottom':int(os.environ.get('BUY_QUOTA_BOTTOM','5')),
          'opportunity':int(os.environ.get('BUY_QUOTA_OPPORTUNITY','5')),
          'normal':int(os.environ.get('BUY_QUOTA_NORMAL','2')),
          'caution':int(os.environ.get('BUY_QUOTA_CAUTION','3')),
          'risk':int(os.environ.get('BUY_QUOTA_RISK','0'))}
    n=qmap.get(a.scenario,5)
    buys=cand.head(n)[['symbol','close','ml_rank']].rename(columns={'close':'ref_close'})
    buys.insert(0,'action','BUY')

    os.makedirs('external_data/daily',exist_ok=True)
    tag=pd.Timestamp(last).strftime('%Y%m%d')
    buys.to_csv(f'external_data/daily/{tag}_buy_signals.csv',index=False)
    Sd[['symbol','close','ml_rank','buyable']].to_csv(f'external_data/daily/{tag}_scores.csv',index=False)

    sells=[]
    if a.holdings:
        h=pd.read_csv(a.holdings,dtype={'symbol':str})
        h['symbol']=h['symbol'].str.zfill(6);h['entry_date']=pd.to_datetime(h['entry_date'])
        cal=sorted(S.date.unique());pos_of={d:i for i,d in enumerate(cal)}
        for _,row in h.iterrows():
            sym=row['symbol'];s=S[S.symbol==sym].sort_values('date')
            if s.empty or s.iloc[-1]['date']!=pd.Timestamp(last):
                sells.append({'symbol':sym,'action':'NO_DATA','reason':''});continue
            m=(s.date-row.entry_date).dt.days>=0
            first=s[m].date.min() if m.any() else s.date.min()
            bars=int(pos_of[last]-pos_of[first]+1)
            close=float(s.iloc[-1]['close']);ep_=float(row['entry_price'])
            shares=float(row.get('shares',1000));mv=shares*close
            pos=SimpleNamespace(bars=bars,market_value=mv,pnl=mv-shares*ep_,shares=shares)
            rr=R[R.symbol==sym].sort_values('date')
            rt=float(Rd[Rd.symbol==sym]['risk_ml_rank'].iloc[0]) if (Rd.symbol==sym).any() else .5
            rpct=rr.risk_ml_score.rank(pct=True);prev=float(rpct.iloc[-2]) if len(rpct)>=2 else rt
            r=R[R.symbol==sym].sort_values('date')
            rmcol=[c for c in ['risk_mag'] if c in r.columns]
            ctx=SimpleNamespace(close=list(s.close),atr=list(s._atr),
                                ml_rank=[float(Sd[Sd.symbol==sym]['ml_rank'].iloc[0])],
                                risk_ml_rank=[prev,rt],risk_mag=[float(r[rmcol[0]].iloc[-1])] if rmcol else None)
            ss,reason=signal_engine.evaluate_sell_signal(ctx,{'primary_scenario':a.scenario},pos,rm['model'])
            sells.append({'symbol':sym,'action':'SELL' if ss else 'HOLD',
                          'reason':reason or '-','pnl_pct':round((close/ep_-1)*100,2),'bars':bars})
    sdf=pd.DataFrame(sells)
    sdf.to_csv(f'external_data/daily/{tag}_sell_signals.csv',index=False)

    print('\n===== 每日信号 =====')
    print(f'评估日:{pd.Timestamp(last):%Y-%m-%d} 场景:{a.scenario} 配额:{n} | 打分股票:{len(Sd)} 可买:{int(Sd.buyable.sum())}')
    print('-- 买入 --');print(buys.to_string(index=False) if len(buys) else '(无满足条件候选)')
    print('-- 卖出 --');print(sdf[sdf.action!='NO_DATA'].to_string(index=False) if len(sdf) else '(未提供持仓)')
    print(f'\n✅ external_data/daily/{tag}_buy_signals.csv | {tag}_sell_signals.csv | {tag}_scores.csv')

if __name__=='__main__':
    main()
