#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exit_signal.py — 持仓离场信号工具

输入持仓 CSV（列: symbol, entry_date, entry_price[, shares]），
对全市场双通道打分后，注入每笔持仓跑与回测完全一致的卖出规则链
（signal_engine.evaluate_sell_signal），输出今日离场建议。

用法：
  python exit_signal.py --holdings my_positions.csv [--scenario normal]
注意：
  - 数据需已同步至评估日；评估日默认取本地数据最新交易日
  - 大盘清仓类退出依赖真实场景，工具默认 normal，可用 --scenario 覆盖
"""
import os,sys,argparse,logging
from types import SimpleNamespace
import numpy as np,pandas as pd,joblib
_here=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,_here)
import co_compute
co_compute.FeatureConfig.MKT_FEATURES=['mkt_macro_regime']
import signal_engine
from local_data_cache import LocalDataCache
from screen import basic_screen
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

def _mkt():
    mc=pd.read_parquet('market_context_cache.parquet')
    mc=mc.set_index('date');mc.index=pd.to_datetime(mc.index)
    need=list(co_compute.FeatureConfig.MKT_FEATURES)
    if not all(c in mc.columns for c in need):
        r=mc.reset_index()
        mc=r.merge(co_compute.build_market_pca_table(r,min_periods=60),on='date',how='left').set_index('date')
        mc.index=pd.to_datetime(mc.index)
    return mc

def _score_channel(ldc,syms,mc,start,end,smooth,biz_feats,model,pred_name):
    dfs=[]
    for i,s in enumerate(syms):
        try:
            df=ldc.get_stock_data(s,start,end,adjust='qfq',mode=2)
            if df.empty or len(df)<60:continue
            df['date']=pd.to_datetime(df['date'])
            df['vwap']=(df['high']+df['low']+2*df['close'])/4.0
            df=co_compute.compute_individual_indicators(df,mc,use_smooth=smooth)
            dfs.append(df)
            if i%1000==0:logging.info(f'{i}...')
        except Exception as e:
            logging.warning(f'{s}:{e}')
    g=pd.concat(dfs).reset_index(drop=True);del dfs
    g=g[g['date']<=pd.to_datetime(end)]
    g=co_compute.apply_standardization(g,features=biz_feats)
    for c in co_compute.FeatureConfig.MKT_FEATURES:
        g[c]=g['date'].map(mc[c]).ffill()
    pc = g.groupby('symbol', sort=False)['close'].shift(1)
    tr = np.maximum(g['high'] - g['low'], np.maximum((g['high'] - pc).abs(), (g['low'] - pc).abs()))
    g['_atr14'] = tr.groupby(g['symbol'], sort=False).transform(lambda s: s.rolling(14, min_periods=1).mean())
    feats = [f for f in model['features'] if f in g.columns]
    g[pred_name] = model['model'].predict(g[feats])
    cols = ['date', 'symbol', 'close', '_atr14', pred_name] + (['risk_mag'] if 'risk_mag' in g.columns else [])
    return g[cols].rename(columns={'_atr14': 'atr'})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--holdings',required=True)
    ap.add_argument('--start',required=True,help='数据起点(覆盖最早入场日即可)')
    ap.add_argument('--end',default=None,help='评估日，默认数据最新日')
    ap.add_argument('--scenario',default='normal')
    a=ap.parse_args()
    if not a.end:
        a.end=pd.Timestamp.now().strftime('%Y-%m-%d')

    h=pd.read_csv(a.holdings,dtype={'symbol':str})
    h['symbol']=h['symbol'].str.zfill(6)
    h['entry_date']=pd.to_datetime(h['entry_date'])
    if 'shares' not in h:h['shares']=1000.0

    ep=joblib.load('chip_accumulation_v6_g_pca1_z.pkl')
    rp=joblib.load('chip_risk_model_v1_g_pca1_z.pkl')
    om=joblib.load(os.environ.get('OPPORT_PKL','chip_opport_magnitude_excess_for_g.pkl'))
    rm=joblib.load(os.environ.get('RISKMAG_PKL','chip_risk_magnitude_for_g.pkl'))
    mc=_mkt();syms=basic_screen();ldc=LocalDataCache(cache_dir='./stock_data_cache')

    logging.info('通道1/2: 平滑(入场) ...')
    S=_score_channel(ldc,syms,mc,a.start,a.end,True,
                     co_compute.FeatureConfig.BIZ_FEATURES,ep,'raw_ml_score')
    logging.info('通道2/2: 原始(风控排名+风险幅度) ...')
    R=_score_channel(ldc,syms,mc,a.start,a.end,False,
                     co_compute.FeatureConfig.BIZ_RISK_FEATURES,rp,'risk_ml_score')
    # ④ 风险幅度（原始通道 25 特征）
    rm_feats=[f for f in rm['features']]
    miss=[f for f in rm_feats if f not in R.columns]
    R['risk_mag']=rm['model'].predict(R[rm_feats]) if not miss else np.nan
    if miss:logging.warning(f'risk_mag 缺特征{len(miss)}个, Risk_Mag_Exit 将跳过')

    # 当日截面排名（与生产同口径）
    last=S.date.max();lastR=R.date.max()
    Sd=S[S.date==last].copy();Rd=R[R.date==lastR].copy()
    Sd['ml_rank']=Sd.raw_ml_score.rank(pct=True,ascending=False)
    Rd['risk_ml_rank']=Rd.risk_ml_score.rank(pct=True,ascending=True)

    cal=sorted(S.date.unique());pos_of={d:i for i,d in enumerate(cal)}
    out=[]
    for _,row in h.iterrows():
        sym=str(row['symbol']).zfill(6)
        s=S[(S.symbol==sym)&(S.date<=last)].sort_values('date')
        r=R[(R.symbol==sym)&(R.date<=last)].sort_values('date')
        if s.empty or s.iloc[-1]['date']!=pd.Timestamp(last):
            out.append(dict(symbol=sym,status='NO_DATA'));continue
        entry=row['entry_date']
        m=(s.date-pd.Timestamp(entry)).dt.days>=0
        first=s[m].date.min() if m.any() else s.date.min()
        bars=int(pos_of[last]-pos_of[first]+1)
        close=float(s.iloc[-1]['close']);atr=float(s.iloc[-1]['atr'])
        shares=float(row.get('shares',1000));ep_=float(row['entry_price'])
        mv=shares*close;cost=shares*ep_
        pos=SimpleNamespace(bars=bars,market_value=mv,pnl=mv-cost,shares=shares)
        mlr=float(Sd[Sd.symbol==sym]['ml_rank'].iloc[0]) if (Sd.symbol==sym).any() else 0.5
        rr=R[R.symbol==sym].sort_values('date')
        r_hist=r.risk_ml_score.rank(pct=True)          # 该股自身历史百分位序列
        r_rank_today=float(Rd[Rd.symbol==sym]['risk_ml_rank'].iloc[0]) if (Rd.symbol==sym).any() else 0.5
        r_prev=float(r_hist.iloc[-2]) if len(r_hist)>=2 else r_rank_today
        ctx=SimpleNamespace(close=list(s['close']),atr=list(s['atr']),
                            ml_rank=[mlr],risk_ml_rank=[r_prev,r_rank_today],risk_mag=None)
        # risk_mag 注入
        rmcol=[c for c in ['risk_mag'] if c in r.columns]
        ctx.risk_mag=[float(r[rmcol[0]].iloc[-1])] if rmcol else None
        env={'primary_scenario':a.scenario}
        ss,reason=signal_engine.evaluate_sell_signal(ctx,env,pos,rm['model'])
        out.append(dict(symbol=sym,bars=bars,pnl_pct=round((close/ep_-1)*100,2),
                        ml_rank=round(mlr,4),risk_ml_rank=round(r_rank_today,4),
                        should_sell=ss,reason=reason or '-'))
    res=pd.DataFrame(out)
    res['_o']=res.should_sell.astype(int);res=res.sort_values(['_o','symbol']).drop(columns='_o')
    os.makedirs('external_data/exits',exist_ok=True)
    pth=f"external_data/exits/exit_signals_{pd.Timestamp(last):%Y%m%d}.csv"
    res.to_csv(pth,index=False);print(res.to_string(index=False));print(f'\n✅ {pth}')

if __name__=='__main__':
    main()
