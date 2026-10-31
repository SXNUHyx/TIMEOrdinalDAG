"""Publication figures from saved fitted DAG and resampling outputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

p=argparse.ArgumentParser();p.add_argument('--fit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.fit);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
model=np.load(root/'point_model.npz');graph=model['A'];pdim=2
freq=pd.read_csv(root/'edge_frequencies.csv').set_index(['source','target']).directed_frequency
fig,ax=plt.subplots(figsize=(7,3.2))
coords={i:(2-i//pdim,1-i%pdim) for i in range(len(graph))};patches={}
for i,(x,y) in coords.items():
    patches[i]=Circle((x,y),.15,fc='#dcebf6' if i<pdim else '#f0f0f0',ec='#29475b',zorder=3)
    ax.add_patch(patches[i])
    lag=i//pdim;name=['Steps','HR'][i%pdim];time='t' if not lag else f't-{lag}'
    ax.text(x,y,f'{name}\n({time})',ha='center',va='center',fontsize=10,zorder=4)
for i,j in zip(*np.where(graph[:,:pdim])):
    x,y=coords[i];xx,yy=coords[j];frequency=float(freq.loc[(i,j)])
    radius=.35 if abs(xx-x)>1 else (.15 if x==xx else .12)
    if x!=xx and y!=yy:radius=0.
    ax.annotate('',(xx,yy),(x,y),arrowprops=dict(arrowstyle='-|>',patchA=patches[i],patchB=patches[j],shrinkA=3,shrinkB=3,
                  mutation_scale=12,lw=1+frequency*2,color='#345c7b',connectionstyle=f'arc3,rad={radius}'),zorder=2)
    labelx=(x+xx)/2+radius*(yy-y)/2
    labely=(y+yy)/2-radius*(xx-x)/2+.04
    ax.text(labelx,labely,f'{frequency:.2f}',fontsize=9,ha='center',bbox=dict(fc='white',ec='none',pad=.2),zorder=5)
ax.set(xlim=(-.4,2.4),ylim=(-.45,1.4));ax.axis('off')
fig.tight_layout();fig.savefig(out/'fitbit_current_graph.pdf');plt.close(fig)
e=pd.read_csv(root/'OCE_results.csv');fig,ax=plt.subplots(figsize=(6,3.2))
for r in e.itertuples():
    y=int(r.outcome_state)+1;ax.plot([r.lower,r.upper],[y,y],color='#345c7b',lw=2)
    ax.plot(r.estimate,y,'o',color='#b34737')
ax.axvline(0,color='gray',lw=.8,ls='--');ax.set(ylabel='Current HR category',xlabel='Latent interval OCE: highest vs lowest lag-one Steps',yticks=[1,2,3,4])
fig.tight_layout();fig.savefig(out/'fitbit_oce.pdf');plt.close(fig)
t=pd.read_csv(root/'state_transitions.csv');fig,axes=plt.subplots(1,2,figsize=(7,3))
for j,ax in enumerate(axes):
    m=t[t.variable.eq(j)].pivot(index='from_state',columns='to_state',values='probability')
    im=ax.imshow(m,vmin=0,vmax=1,cmap='Blues');ax.set(title=['Steps','HR'][j],xlabel='Next category',ylabel='Previous category')
fig.colorbar(im,ax=axes.tolist(),shrink=.8);fig.savefig(out/'fitbit_transitions.pdf',bbox_inches='tight');plt.close(fig)
