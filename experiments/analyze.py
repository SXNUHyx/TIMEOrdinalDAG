import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args();root=Path(a.results);rows=[];effects=[]
for f in root.rglob('point_summary.json'):
 r=json.loads(f.read_text());r['run']=str(f.parent.parent);r['rep']=f.parent.name;rows.append(r)
 effect=f.with_name('OCE_results.csv')
 if effect.exists():
  g=pd.read_csv(effect);g['run']=r['run'];g['rep']=r['rep'];effects.append(g)
if rows:
 d=pd.DataFrame(rows);d.to_csv(root/'all_structure.csv',index=False)
 metrics=[k for k in ['SHD','TPR','FDR','F1','contemp_SHD','contemp_F1'] if k in d]
 for k in metrics:
  g=d.groupby(['run','method','mode'])[k].agg(['count','mean','std']);g['mcse']=g['std']/np.sqrt(g['count']);g.to_csv(root/f'structure_{k}.csv')
if effects:
 d=pd.concat(effects,ignore_index=True);d.to_csv(root/'all_effects.csv',index=False);out=[]
 if 'truth' in d:
  d=d.dropna(subset=['truth'])
  for key,g in d.groupby(['run','outcome_state']):
   err=g.estimate-g.truth;r=dict(run=key[0],state=key[1],datasets=len(g),bias=err.mean(),RMSE=np.sqrt(np.mean(err**2)))
   if 'lower' in g and g.lower.notna().any():
    h=g.dropna(subset=['lower','upper']);coverage=np.mean((h.lower<=h.truth)&(h.truth<=h.upper));r.update(coverage=coverage,coverage_mcse=np.sqrt(coverage*(1-coverage)/len(h)),mean_width=(h.upper-h.lower).mean(),coverage_datasets=len(h))
   out.append(r)
 pd.DataFrame(out).to_csv(root/'effect_summary.csv',index=False)
for f in root.rglob('state_transitions.csv'):
 d=pd.read_csv(f)
 for j,g in d.groupby('variable'):
  matrix=g.pivot(index='from_state',columns='to_state',values='probability');fig,ax=plt.subplots(figsize=(4,3));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues');ax.set(xlabel='Next category',ylabel='Previous category',title=f'Observed variable {j}');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(f.with_name(f'transitions_{j}.png'),dpi=180);plt.close(fig)
for f in root.rglob('edge_frequencies.csv'):
 d=pd.read_csv(f);mask=d.source.ne(d.target);d=d[mask];adj=d.pivot(index='source',columns='target',values='frequency');fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(adj,vmin=0,vmax=1,cmap='Blues');ax.set(xlabel='Target node index',ylabel='Source node index',title='Adjacency selection frequency');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(f.with_name('edge_frequency_heatmap.png'),dpi=180);plt.close(fig)
print('Saved summaries and heatmaps. Original-R and python-reference results must be reported separately.')
