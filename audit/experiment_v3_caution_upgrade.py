import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from audit.check_scenario import prepare_zzqz, load_breadth, label_days
from audit.scenario_challenger import _window_path_stats
import is_market_ok

zzqz=prepare_zzqz(); breadth=load_breadth('parquet')
common=sorted(set(zzqz.index)&set(breadth.index))
common=[d for i,d in enumerate(common) if i>=120]
ign = breadth['high10'] > breadth['high10'].shift(1).rolling(5,min_periods=2).mean()

MULT={'risk':0.0,'bottom':1.2,'opportunity':0.8,'caution':0.7,'normal':0.5}
for tag,(a,b_) in [('14-20',('2014-01-01','2020-12-31')),('21-26',('2021-01-01','2026-08-31'))]:
    ed=[d for d in common if pd.Timestamp(a)<=d<=pd.Timestamp(b_)]
    lab=label_days(ed,zzqz,breadth,is_market_ok.scenario_based_market_judgment,total_stocks=None,use_dynamic=True)
    m=(lab['primary_scenario']=='caution')&lab['decision_reason'].astype(str).str.contains('MA60下方')&ign.reindex(lab.index).fillna(False)
    ups=list(lab.index[m])
    ps=_window_path_stats(zzqz['close'],ups).dropna()
    print(f"[{tag}] 升级日 n={len(ps)} | 终值均值={ps['term'].mean():+.2%} 胜率={(ps['term']>0).mean():.0%} "
          f"| P(窗口最深<=-8%)={(ps['worst']<=-0.08).mean():.1%} | 最深回撤中位={ps['mdd'].median():.2%}")

# 生成变体标签 V3: caution_MA60下 + 点火 -> opportunity
ed=[d for d in common if pd.Timestamp('2021-01-01')<=d<=pd.Timestamp('2026-08-31')]
lab=label_days(ed,zzqz,breadth,is_market_ok.scenario_based_market_judgment,total_stocks=None,use_dynamic=True)
m=(lab['primary_scenario']=='caution')&lab['decision_reason'].astype(str).str.contains('MA60下方')&ign.reindex(lab.index).fillna(False)
v3=lab.copy()
n=0
for d in lab.index[m]:
    v3.at[d,'primary_scenario']='opportunity'
    v3.at[d,'position_multiplier']=MULT['opportunity']
    v3.at[d,'decision_reason']=str(v3.at[d,'decision_reason'])+'+V3修复点火升级'
    n+=1
v3['is_market_ok']=v3['primary_scenario']!='risk'
print(f"\n[V3] 升级天数: {n} | 变体opportunity总数: {(v3['primary_scenario']=='opportunity').sum()} (原653)")
seq=v3['primary_scenario']; print(f"[V3] 翻转率={(seq!=seq.shift()).iloc[1:].mean():.1%}")
out='external_data/scenario_audit/labels_inc_v3.csv'
v3.to_csv(out,encoding='utf-8-sig'); print("saved:",out)
