# audit/gts_reuse_check.py
# get_trend_signals(搁置方向)启用价值评估: 场景暴露 x 持仓收益的历史证据
# 结论(2026-08, 现役 live_ref 全窗口 1756 笔):
#   持仓窗口含>=1个risk日的交易 n=713(41%), 均收益-1.51%/胜率35%/MAE中位-71%,
#   显著劣于无risk日交易(+2.66%/66%/-45%)。亏损单中57%持有期含risk日。
#   原第六级规则(tend_broke x risk_ml_score_crash 联合触发)的联合触发思想
#   与现代组件(Risk_Mag_Exit / Market_Risk_Clearance)可组合, 具备试点价值。
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from audit.check_scenario import prepare_zzqz, load_breadth, label_days
import is_market_ok

def run(trades_path='results/daily_20260825_221334_live_ref/trades.xlsx'):
    zzqz=prepare_zzqz(); breadth=load_breadth('parquet')
    common=sorted(set(zzqz.index)&set(breadth.index))
    common=[d for i,d in enumerate(common) if i>=120]
    ed=[d for d in common if d<=pd.Timestamp('2026-08-21')]
    lab=label_days(ed,zzqz,breadth,is_market_ok.scenario_based_market_judgment,
                   total_stocks=None,use_dynamic=True)
    scen=lab['primary_scenario']
    t=pd.read_excel(trades_path)
    t['entry_date']=pd.to_datetime(t['entry_date']); t['exit_date']=pd.to_datetime(t['exit_date'])
    def ws(row):
        w=scen.reindex(pd.date_range(row['entry_date'],row['exit_date'],freq='D')).dropna()
        return pd.Series({'risk_d':(w=='risk').sum(),'caution_d':(w=='caution').sum()})
    ts=pd.concat([t,t.apply(ws,axis=1)],axis=1)
    for name,mask in [('无risk日',ts.risk_d==0),('含>=1risk日',ts.risk_d>=1)]:
        g=ts[mask]
        print(f"{name}: n={len(g)} 均收益={g['return_pct'].mean():+.2f}% "
              f"胜率={(g['return_pct']>0).mean():.0%} MAE中位={g['mae'].median()*100:.1f}%")

if __name__=='__main__':
    run()
