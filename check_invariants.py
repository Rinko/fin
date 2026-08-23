#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_invariants.py — 每日输出不变量监控
用法: python check_invariants.py --scores <scores.csv> [--sell <sell_signals.csv>]
行为: 提取今日统计指纹 → 与近60日历史带比较 → 越界告警 → 追加台账
"""
import os,sys,argparse
import numpy as np,pandas as pd
HIST='external_data/shadow/invariants_history.csv'
def extract(sc,sl):
    d=pd.read_csv(sc,dtype={'symbol':str})
    r={'n_scored':len(d),'rank_mean':d.ml_rank.mean(),'rank_std':d.ml_rank.std(),
       'n_buyable':int(d.buyable.sum())}
    if 'raw_ml_score' in d: r['score_std']=d.raw_ml_score.std()
    if sl and os.path.exists(sl):
        s=pd.read_csv(sl,dtype={'symbol':str});s=s[s.action!='NO_DATA']
        for reason in ['Risk_Sudden_Deterioration','Time_Efficiency_Exit',
                       'Risk_Mag_Exit','Profit_Protection_Risk']:
            r[f'sell_{reason}']=int((s.reason==reason).sum())
    r['date']=pd.Timestamp.now().strftime('%Y-%m-%d')
    return r
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scores',required=True);ap.add_argument('--sell',default=None)
    a=ap.parse_args()
    row=extract(a.scores,a.sell)
    hist=pd.read_csv(HIST) if os.path.exists(HIST) else pd.DataFrame()
    alerts=[]
    if len(hist)>=15:
        for k,v in row.items():
            if k=='date' or k not in hist.columns:continue
            h=hist[k].dropna()
            lo,hi=h.quantile(.02),h.quantile(.98)
            spread=(hi-lo)/(abs(h.mean())+1e-9)
            if spread<1e-6:continue
            if v<lo-(hi-lo)*0.5 or v>hi+(hi-lo)*0.5:
                alerts.append(f'{k}={v:.4g} 出带[{lo:.4g},{hi:.4g}]')
    else:
        print(f'[invariants] 台账积累中 {len(hist)}/15，暂不做带检')
    today=pd.DataFrame([row])
    today.to_csv(HIST,mode='a',header=not os.path.exists(HIST),index=False)
    if alerts:
        print('⚠️ [invariants] 不变量越界:')
        for x in alerts:print('   ',x)
    else:
        print('✅ [invariants] 全部指标在带内' if len(hist)>=15 else '')
    print(f"[invariants] 今日指纹: "+", ".join(f'{k}={v:.4g}' if isinstance(v,(int,float,np.floating)) else f'{k}={v}' for k,v in row.items() if k!='date'))
if __name__=='__main__':main()
