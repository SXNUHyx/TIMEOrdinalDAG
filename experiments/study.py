"""Prespecified reproducible revision study. Every failed replicate is retained.

Run with BLAS threads limited to one; four processes fit the available CPU quota.
Resume validates the full job specification before reusing a completed directory.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
from run import run_one, dump
from simulations import generate


def jobs(root):
    common = dict(backend='python-reference', mode='overlap', subjects=50,
                  length=300, p=3, lag=2, q=4, scheme='cart', penalties='6',
                  mc=20, max_em=10, bootstrap=0, block=None, stride=1,
                  rho=.5, noise=1., heterogeneity=0., scenario='gaussian',
                  calibrate=False, unconstrained=False, no_contemp=False,
                  loo=False, mapping_fraction=.4, tuning_fraction=.2,
                  baselines='')
    plans = []
    for scheme in ['cart', 'quantile', 'width']:
        plans.append(('structure_' + scheme, 50, dict(scheme=scheme,
                      penalties='2,6,10', baselines='pcmci,pcmciplus,lpcmci' if scheme=='cart' else '')))
    plans.append(('coverage_landmark', 100, dict(mode='landmark',subjects=100,
                  length=100,p=2,bootstrap=500)))
    for scenario in ['contemporaneous','nonlinear','nongaussian','confounded','switching','derived']:
        plans.append(('scenario_' + scenario,50,dict(scenario=scenario,baselines='pcmci,pcmciplus,lpcmci' if scenario not in ['switching','derived'] else '')))
    for subjects in [50,100,200]:
        plans.append(('landmark_' + str(subjects),50,dict(mode='landmark',subjects=subjects)))
    for key, values in [('lag',[1,5]),('q',[3,5]),('mc',[5,100]),
                        ('rho',[.2,.8]),('noise',[.5,2.]),('heterogeneity',[.5,1.])]:
        for value in values:
            plans.append((f'sensitivity_{key}_{value}',50,{key:value}))
    plans.append(('unconstrained',50,dict(unconstrained=True)))
    for key,values in [('subjects',[20,100]),('length',[100,600])]:
        for value in values:plans.append((f'sensitivity_{key}_{value}',50,{key:value}))
    # Exploratory follow-up to the low-N strong-penalty coverage failure.
    plans.append(('coverage_landmark_larger',100,dict(mode='landmark',subjects=500,
                  length=100,p=2,penalties='2',bootstrap=500)))
    for name, reps, overrides in plans:
        for rep in range(reps):
            spec = {**common,**overrides}
            spec.update(seed=2026+rep,rep=rep,family=name)
            yield dict(spec=spec,out=str(Path(root)/name/f'rep_{rep:03d}'))


def execute(job):
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    record=out/'job.json';status=out/'status.json'
    if record.exists() and json.loads(record.read_text())!=job['spec']:
        raise RuntimeError('Refuse to mix results with changed settings: '+str(out))
    if status.exists() and json.loads(status.read_text()).get('status')=='complete':
        return str(out),'reused'
    if status.exists() and json.loads(status.read_text()).get('status')=='FAILED':
        return str(out),'FAILED'  # Preserve failed attempts; diagnose before rerunning.
    dump(record,job['spec']);args=SimpleNamespace(**job['spec'])
    tic=time.perf_counter()
    with open(out/'execution.log','w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            sub,meta=generate(args.subjects,args.length,args.p,args.noise,args.rho,args.heterogeneity,args.scenario,args.seed)
            np.savez_compressed(out/'truth.npz',A=meta['A'],B0=meta['B0'],states=meta['states'])
            run_one(sub,args,out,meta)
            dump(status,dict(status='complete',seconds=time.perf_counter()-tic))
            return str(out),'complete'
        except Exception as ex:
            traceback.print_exc()
            dump(status,dict(status='FAILED',error=repr(ex),seconds=time.perf_counter()-tic))
            return str(out),'FAILED'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='/workspace/revision_results')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--families',default='')
    a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    selected=set(filter(None,a.families.split(',')))
    todo=[j for j in jobs(root) if not selected or j['spec']['family'] in selected]
    dump(root/'plan.json',todo)
    # Inputs are immutable once submitted. Environment variables propagate to workers.
    dump(root/'environment.json',dict(python=sys.version,workers=a.workers,
         blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    progress=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending=[pool.submit(execute,j) for j in todo]
        for future in as_completed(pending):
            path,state=future.result();progress.append(dict(path=path,status=state))
            dump(root/'progress.json',progress)
            print(f'{len(progress)}/{len(todo)} {state} {path}',flush=True)
    if any(x['status']=='FAILED' for x in progress):raise SystemExit(1)


if __name__=='__main__':main()
