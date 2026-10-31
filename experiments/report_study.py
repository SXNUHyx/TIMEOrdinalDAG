"""Audit completed, failed and unrun experiments and export manuscript evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from simulations import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.results);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];baselines=[];effects=[];audit=[];heldout=[]
    plan=json.loads((root/'plan.json').read_text())
    for job in plan:
        spec=job['spec'];folder=root/spec['family']/f"rep_{spec['rep']:03d}";sf=folder/'status.json'
        status=json.loads(sf.read_text()) if sf.exists() else {'status':'unrun_or_running'}
        audit.append(dict(family=spec['family'],rep=spec['rep'],**status))
        if status['status']!='complete':continue
        r=json.loads((folder/'point_summary.json').read_text())
        # Recompute metrics from raw saved graphs; count a contemporaneous
        # reversal once in SHD, not as independent delete-plus-add operations.
        truth=np.load(folder/'truth.npz');pred=np.load(folder/'point_model.npz')['A']
        r.update(metrics(pred,dict(A=truth['A'],B0=truth['B0'],scenario=spec['scenario']),spec['lag'],spec['p']))
        r['metrics_version']='directed-reversal-once-v2'
        r.update(family=spec['family'],rep=spec['rep'],seconds_total=status['seconds'])
        rows.append(r)
        for file,dest in [('baselines.csv',baselines),('OCE_results.csv',effects),('heldout_discretization.csv',heldout)]:
            f=folder/file
            if f.exists() and f.stat().st_size>1:
                d=pd.read_csv(f);d['family']=spec['family'];d['rep']=spec['rep'];dest.append(d)
    pd.DataFrame(audit).to_csv(out/'run_audit.csv',index=False)
    d=pd.DataFrame(rows);d.to_csv(out/'all_point_estimates.csv',index=False)
    report=[]
    for family,g in d.groupby('family'):
        row=dict(family=family,completed=len(g),nonconverged=int(g.converged.eq(False).sum()),
                 median_seconds=g.seconds_total.median(),B_requested=g.B_requested.sum(),
                 B_success=g.B_success.sum(),B_failed=g.B_failed.sum(),
                 B_nonconverged=g.B_nonconverged.sum())
        for metric in ['SHD','TPR','FDR','F1','contemp_SHD','contemp_F1']:
            x=g[metric].dropna() if metric in g else pd.Series(dtype=float)
            row.update({metric:x.mean(),metric+'_sd':x.std(),metric+'_mcse':x.std()/np.sqrt(len(x)) if len(x) else np.nan})
        report.append(row)
    pd.DataFrame(report).to_csv(out/'structure_summary.csv',index=False)
    if baselines:
        b=pd.concat(baselines,ignore_index=True);b.to_csv(out/'all_baselines.csv',index=False)
        b[b.status.eq('ok')].groupby(['family','method'])[['SHD','TPR','FDR','F1','seconds']].agg(['count','mean','std']).to_csv(out/'baseline_summary.csv')
    if heldout:
        h=pd.concat(heldout,ignore_index=True);h.to_csv(out/'all_heldout.csv',index=False)
        h.groupby(['family','variable']).heldout_R2.agg(['count','mean','std']).to_csv(out/'heldout_summary.csv')
    if effects:
        e=pd.concat(effects,ignore_index=True);e.to_csv(out/'all_effects.csv',index=False)
        effect_summary=[]
        for (family,state),g in e.dropna(subset=['truth']).groupby(['family','outcome_state']):
            error=g.estimate-g.truth
            row=dict(family=family,state=state,datasets=len(g),bias=error.mean(),RMSE=np.sqrt(np.mean(error**2)))
            if 'lower' in g:
                h=g.dropna(subset=['lower','upper']);n=len(h)
                if n:
                    k=int(((h.lower<=h.truth)&(h.truth<=h.upper)).sum());ci=binomtest(k,n).proportion_ci()
                    row.update(coverage=k/n,coverage_datasets=n,coverage_lower=ci.low,coverage_upper=ci.high,
                               mean_width=(h.upper-h.lower).mean(),mean_ancestor_frequency=h.ancestor_frequency.mean())
            effect_summary.append(row)
        pd.DataFrame(effect_summary).to_csv(out/'effect_summary.csv',index=False)
    # Representative displays are explicitly identified by fixed seed, not selected performance.
    folder=root/'structure_cart'/'rep_000'
    if (folder/'state_transitions.csv').exists():
        transitions=pd.read_csv(folder/'state_transitions.csv')
        fig,axes=plt.subplots(1,3,figsize=(10,3))
        for j,ax in enumerate(axes):
            g=transitions[transitions.variable.eq(j)];m=g.pivot(index='from_state',columns='to_state',values='probability')
            im=ax.imshow(m,vmin=0,vmax=1,cmap='Blues');ax.set(title=f'Observed variable {j+1}',xlabel='Next category',ylabel='Previous category')
            for ii in range(len(m)):
                for jj in range(len(m.columns)):ax.text(jj,ii,f'{m.iloc[ii,jj]:.2f}',ha='center',va='center',fontsize=8,color='white' if m.iloc[ii,jj]>.55 else 'black')
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8);fig.savefig(out/'observed_transitions.pdf',bbox_inches='tight');plt.close(fig)
    plot_data=d[d.family.isin(['structure_cart','structure_quantile','structure_width'])]
    if len(plot_data):
        fig,ax=plt.subplots(figsize=(6,3));groups=[plot_data[plot_data.family.eq('structure_'+s)].SHD.to_numpy() for s in ['cart','quantile','width']]
        ax.boxplot(groups,tick_labels=['CART','Quantile','Width'],showmeans=True)
        ax.set(ylabel='Lagged SHD into current nodes',title='Python reference; independent simulation replicates')
        fig.tight_layout();fig.savefig(out/'structure_comparison.pdf');plt.close(fig)
    print(pd.DataFrame(audit).groupby(['family','status']).size().to_string())


if __name__=='__main__':main()
