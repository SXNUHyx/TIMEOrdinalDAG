"""Show the KNOWN switching generator, never label it hidden-state recovery."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.results)/'scenario_switching';out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
counts=np.zeros((2,2),int);first=None
for f in sorted(root.glob('rep_*/truth.npz')):
    if not f.with_name('status.json').exists():continue
    t=np.load(f);states=t['states']
    for z in states:np.add.at(counts,(z[:-1],z[1:]),1)
    if first is None:first=t['A'].copy()
if first is None:raise RuntimeError('No switching results are available')
prob=counts/counts.sum(1,keepdims=True)
pd.DataFrame(prob,index=['state0','state1'],columns=['state0','state1']).to_csv(out/'known_generator_transitions.csv')
fig,axes=plt.subplots(1,3,figsize=(10,3))
for state,ax in enumerate(axes[:2]):
    matrix=first[0].copy()
    if state:matrix[1,0]=0;matrix[0,1]=.3
    ax.imshow(matrix,vmin=0,vmax=.5,cmap='Blues')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=9)
    ax.set(title=f'Known regime {state}: lag-one coefficients',xlabel='Source variable',ylabel='Target variable',xticks=range(3),yticks=range(3))
axes[2].imshow(prob,vmin=0,vmax=1,cmap='Blues')
for i in range(2):
    for j in range(2):axes[2].text(j,i,f'{prob[i,j]:.3f}',ha='center',va='center')
axes[2].set(title='Known generator-state transitions',xlabel='Next regime',ylabel='Previous regime',xticks=[0,1],yticks=[0,1])
fig.tight_layout();fig.savefig(out/'known_regimes.pdf');plt.close(fig)
