"""Controlled CPU-time comparison, distinct from scientific dataset repetitions."""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from simulations import generate
from data import subject_split, windows, fit_maps, encode
from osem import TemporalOSEM
from baselines import tigramite_fit


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--reps',type=int,default=10);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    subjects,_=generate(S=50,T=300,p=3,seed=2026)
    groups,_,_=subject_split(subjects,2026,False)
    series=[segment for subject in groups[2] for segment in subject]
    def ordinal(scheme):
        mapping,_=windows(groups[0],2);validation,_=windows(groups[1],2);analysis,_=windows(groups[2],2)
        cuts,_=fit_maps(mapping,3,2,4,scheme);c=encode(analysis,cuts);v=encode(validation,cuts)
        levels=[len(cuts[j%3])+1 for j in range(9)]
        losses=[]
        for penalty in [2,6,10]:
            m=TemporalOSEM(3,2,penalty,20,10,seed=2026).fit(c,levels)
            losses.append(m.pair_logloss(v))
        return int(np.argmin(losses))
    methods={scheme:(lambda scheme=scheme:ordinal(scheme)) for scheme in ['cart','quantile','width']}
    methods.update({name:(lambda name=name:tigramite_fit(series,2,name)) for name in ['pcmci','pcmciplus','lpcmci']})
    # Exclude first-import and initial library setup from all methods equally.
    for run in methods.values():run()
    rng=np.random.default_rng(2026);rows=[]
    for rep in range(a.reps):
        for name in rng.permutation(list(methods)):
            cpu=time.process_time();wall=time.perf_counter();methods[name]()
            rows.append(dict(repetition=rep,method=name,cpu_seconds=time.process_time()-cpu,
                        wall_seconds=time.perf_counter()-wall,analysis_subjects=20,ordinal_window_rows=5960,
                        workload='mapping plus three-penalty selection' if name in ['cart','quantile','width'] else 'fixed-alpha single discovery fit'))
    d=pd.DataFrame(rows);d.to_csv(out/'runtime_repetitions.csv',index=False)
    d.groupby('method').agg(repetitions=('cpu_seconds','size'),cpu_mean=('cpu_seconds','mean'),
        cpu_sd=('cpu_seconds','std'),wall_median=('wall_seconds','median')).to_csv(out/'runtime_benchmark.csv')
    print('Saved controlled CPU-time repetitions; these are not additional independent scientific datasets.')


if __name__=='__main__':main()
