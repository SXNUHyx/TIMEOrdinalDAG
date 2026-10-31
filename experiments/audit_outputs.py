"""Validate saved scientific artifacts, independently of command exit status."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from osem import acyclic, mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.results);plan=json.loads((root/'plan.json').read_text())
    complete=failed=pending=bootstrap=baseline_failures=0
    for job in plan:
        spec=job['spec'];folder=root/spec['family']/f"rep_{spec['rep']:03d}";sf=folder/'status.json'
        if not sf.exists():pending+=1;continue
        status=json.loads(sf.read_text())['status']
        if status=='FAILED':failed+=1;continue
        if status!='complete':pending+=1;continue
        complete+=1
        info=json.loads((folder/'model.json').read_text());point=json.loads((folder/'point_summary.json').read_text())
        model=np.load(folder/'point_model.npz');graph=model['A'];dim=len(graph);pdim=len(info['maps'])
        allowed=~np.eye(dim,dtype=bool) if spec['unconstrained'] else mask(pdim,spec['lag'],not spec['no_contemp'])
        assert acyclic(graph) and not np.any(graph&~allowed),folder
        assert np.isfinite(model['B']).all() and np.all(model['v']>0),folder
        assert np.linalg.eigvalsh(model['Sigma']).min()>0,folder
        effects=pd.read_csv(folder/'OCE_results.csv').estimate.to_numpy()
        assert abs(effects.sum())<1e-5 and np.max(np.abs(effects))<=1+1e-6,folder
        assert point['B_requested']==point['B_success']+point['B_failed'],folder
        assert point['B_requested']==spec['bootstrap'],folder
        if point['B_success']:
            boot=np.load(folder/'bootstrap_models.npz');assert len(boot['A'])==point['B_success'],folder
            assert not np.any(boot['A']&~allowed),folder
            assert np.all(np.abs(boot['OCE'].sum(axis=1))<1e-5),folder
            assert np.max(np.abs(boot['OCE']))<=1+1e-6,folder
            bootstrap+=len(boot['A'])
        bf=folder/'baselines.csv'
        if bf.exists() and bf.stat().st_size>1:
            b=pd.read_csv(bf);baseline_failures+=int(b.status.eq('FAILED').sum())
    real_points=real_bootstrap=0
    for dirname in ['fitbit_stride10','fitbit_lag1','fitbit_lag5']:
        folder=root/dirname
        if not (folder/'OCE_results.csv').exists():continue
        info=json.loads((folder/'model.json').read_text());model=np.load(folder/'point_model.npz')
        dim=len(model['A']);pdim=len(info['maps']);lag=dim//pdim-1
        allowed=mask(pdim,lag);assert acyclic(model['A']) and not np.any(model['A']&~allowed),folder
        effect=pd.read_csv(folder/'OCE_results.csv').estimate.to_numpy()
        assert abs(effect.sum())<1e-5 and np.max(np.abs(effect))<=1+1e-6,folder
        if (folder/'bootstrap_models.npz').exists():
            boot=np.load(folder/'bootstrap_models.npz');assert not np.any(boot['A']&~allowed),folder
            assert np.all(np.abs(boot['OCE'].sum(1))<1e-5),folder
            real_bootstrap+=len(boot['A'])
        real_points+=1
    result=dict(planned=len(plan),completed=complete,failed_runs=failed,pending=pending,
        saved_simulation_bootstrap_fits=bootstrap,baseline_failures=baseline_failures,
        fitbit_point_analyses=real_points,saved_fitbit_bootstrap_fits=real_bootstrap,
        checks='Point DAG acyclicity and temporal mask; finite parameters and positive covariance; OCE range and zero sum; bootstrap count, temporal mask and zero-sum effects')
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if pending or failed or baseline_failures or real_points!=3 or real_bootstrap!=500:raise SystemExit(1)


if __name__=='__main__':main()
