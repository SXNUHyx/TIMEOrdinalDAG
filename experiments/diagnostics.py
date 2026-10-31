"""Gap-safe raw ACF/PACF and a clearly labeled score-residual PROXY.
Proxy residuals do not constitute a latent-innovation independence test.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.special import ndtri
from run import raw_subjects
from data import subject_split,windows,encode
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--fit',required=True);p.add_argument('--columns',default='Steps,HR');p.add_argument('--maxlag',type=int,default=30);a=p.parse_args()
root=Path(a.fit);info=json.loads((root/'model.json').read_text());argsfile=root/'arguments.json'
if not argsfile.exists():argsfile=root.parent/'arguments.json'
config=json.loads(argsfile.read_text())['arguments'];subjects,rawaudit=raw_subjects(a.input,a.columns.split(','));groups,_,_=subject_split(subjects,config['seed'],config['calibrate'],config.get('mapping_fraction',.4),config.get('tuning_fraction',.2))
analysis=groups[2];maps=[np.array(c) for c in info['maps']];fit=np.load(root/'point_model.npz');B=fit['B'];L=config['lag'];dim=len(B);pdim=len(maps);rows=[]
def report(v,kind,sid,segment,variable):
 v=np.asarray(v,float);v=v-v.mean();den=v@v
 for lag in range(1,min(a.maxlag,len(v)//3)+1):
  if den<1e-10:acf=pacf=np.nan
  else:
   r=np.array([v[:-k]@v[k:]/den if k else 1 for k in range(lag+1)])
   acf=r[-1]
   try:pacf=np.linalg.solve(toeplitz(r[:-1])+np.eye(lag)*1e-8,r[1:])[-1]
   except np.linalg.LinAlgError:pacf=np.nan
  rows.append(dict(subject_index=sid,segment=segment,variable=variable,kind=kind,lag=lag,acf=acf,pacf=pacf))
for sid,sub in enumerate(analysis):
 for segment,x in enumerate(sub):
  for j in range(pdim):report(x[:,j],'raw_series',sid,segment,j)
  z,_=windows([[x]],L);c=encode(z,maps);normal=np.column_stack([ndtri((c[:,j]+.5)/(len(maps[j%pdim])+1)) for j in range(dim)])
  for j in range(pdim):report(normal[:,j]-normal@B[j],'ordinal_score_residual_proxy',sid,segment,j)
pd.DataFrame(rows).to_csv(root/'acf_pacf_diagnostics.csv',index=False)
print('Saved raw diagnostics and explicitly labeled residual proxies; inspect lag adequacy and do not infer iid from a proxy.')
