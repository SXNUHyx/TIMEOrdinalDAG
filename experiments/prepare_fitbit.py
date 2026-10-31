"""Local Fitbit export -> minute alignment, auditable counts, no remote download."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def main(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def read(name,cols):
        hits=list(root.rglob(name)) + list(root.rglob(name.replace('.csv','_merged.csv')))
        if len(hits)!=1: raise ValueError(f'Expected exactly one {name}; found {len(hits)}. Select only the April-May export.')
        d=pd.read_csv(hits[0],usecols=cols,dtype={'Id':'str'})
        return d
    steps=read('minuteStepsNarrow.csv',['Id','ActivityMinute','Steps']);hr=read('heartrate_seconds_merged.csv',['Id','Time','Value'])
    steps['time']=pd.to_datetime(steps.ActivityMinute,format='%m/%d/%Y %I:%M:%S %p').dt.floor('min');hr['time']=pd.to_datetime(hr.Time,format='%m/%d/%Y %I:%M:%S %p').dt.floor('min')
    if steps.duplicated(['Id','time']).any():raise ValueError('Duplicate step-minute keys: inspect source instead of averaging silently')
    hr['Value']=pd.to_numeric(hr.Value,errors='coerce');hr.loc[hr.Value<=0,'Value']=np.nan
    hm=hr.groupby(['Id','time'],as_index=False).agg(HR=('Value','mean'),HR_count=('Value','count'))
    panel=steps[['Id','time','Steps']].merge(hm,on=['Id','time'],how='left',validate='one_to_one').rename(columns={'Id':'subject'})
    panel['Steps']=pd.to_numeric(panel.Steps,errors='coerce');panel.loc[panel.Steps<0,'Steps']=np.nan
    panel=panel.sort_values(['subject','time']);panel.to_csv(out/'minute_panel.csv',index=False)
    audit=[]
    for sid,g in panel.groupby('subject'):
        complete=g[['Steps','HR']].notna().all(1);valid=0
        for L in [1,2,5,10]:
            good=complete.copy()
            for l in range(1,L+1):
                good &= complete.shift(l,fill_value=False) & g.time.sub(g.time.shift(l)).eq(pd.Timedelta(minutes=l))
            audit.append(dict(subject=sid,L=L,minute_rows=len(g),HR_minutes=int(g.HR.notna().sum()),complete_minutes=int(complete.sum()),lag_windows=int(good.sum()),removed_after_lag=int(len(g)-good.sum()),removed_fraction=float(1-good.mean()),zero_steps_HR_missing=int(((g.Steps==0)&g.HR.isna()).sum())))
    pd.DataFrame(audit).to_csv(out/'flow_per_subject.csv',index=False)
    summary=dict(step_subjects=steps.Id.nunique(),HR_subjects=hr.Id.nunique(),HR_raw_rows=len(hr),HR_unique_minutes=len(hm),step_minutes=len(steps),aligned_minutes=len(panel),HR_available_minutes=int(panel.HR.notna().sum()),complete_minutes=int(panel[['Steps','HR']].notna().all(1).sum()),HR_aggregation='mean of positive observations within minute',missing='no interpolation; gaps break lag windows',sleep='all clock times included; no claim of sleep identification',nonwear='zero steps alone is not nonwear; missing HR remains missing')
    (out/'flow_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Optional derived channels are audited, excluded from primary model.
    derived=[]
    for name,col in [('minuteMETsNarrow.csv','METs'),('minuteIntensitiesNarrow.csv','Intensity'),('minuteCaloriesNarrow.csv','Calories')]:
        hits=list(root.rglob(name)) + list(root.rglob(name.replace('.csv','_merged.csv')))
        if len(hits)==1:
            d=pd.read_csv(hits[0],dtype={'Id':'str'});d['time']=pd.to_datetime(d.ActivityMinute,format='%m/%d/%Y %I:%M:%S %p').dt.floor('min');d=d.rename(columns={'Id':'subject'})
            if d.duplicated(['subject','time']).any():raise ValueError(f'Duplicate keys in {name}')
            derived.append(d[['subject','time',col]])
    if derived:
        d=panel
        for v in derived:d=d.merge(v,on=['subject','time'],how='left',validate='one_to_one')
        d.to_csv(out/'derived_channels_audit_only.csv',index=False)
        if {'METs','Intensity'}<=set(d.columns):
            pd.crosstab(d.METs,d.Intensity).to_csv(out/'METs_Intensity_crosstab.csv')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='results/fitbit_prepared');a=p.parse_args();main(a.data,a.out)
