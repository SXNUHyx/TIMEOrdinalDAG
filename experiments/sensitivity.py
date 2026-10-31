import argparse,subprocess,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--out',default='results/sensitivity');p.add_argument('--reps',type=int,default=50);p.add_argument('--backend',choices=['original-r','python-reference'],default='python-reference');a=p.parse_args();jobs=[]
base=[sys.executable,'run.py','simulate','--backend',a.backend,'--subjects','50','--length','300','--p','3','--reps',str(a.reps),'--penalties','6','--mc','20','--baselines','']
for flag,values in [('lag',[1,2,5]),('q',[3,4,5]),('mc',[5,20,100]),('penalties',[2,6,10]),('rho',[.2,.5,.8]),('noise',[.5,1,2]),('heterogeneity',[0,.5,1]),('subjects',[20,50,100]),('length',[300,600,1200]),('stride',[1,3,10]),('mapping-fraction',[.3,.4,.5])]:
 for v in values:jobs.append(base+['--'+flag,str(v),'--out',str(Path(a.out)/f'{flag}_{v}')])
for scheme in ['cart','quantile','width']:jobs.append(base+['--scheme',scheme,'--out',str(Path(a.out)/f'scheme_{scheme}')])
jobs.append(base+['--unconstrained','--out',str(Path(a.out)/'unconstrained')])
# Sufficiently many independent subjects for a genuine iid-row OSEM benchmark.
for S in [50,100,200]:jobs.append(base+['--mode','landmark','--subjects',str(S),'--out',str(Path(a.out)/f'landmark_{S}')])
# Block resampling jobs are expensive; 100 repeats x500 bootstrap is an explicit full-run target.
for b in [30,60,120]:jobs.append(base+['--bootstrap','500','--block',str(b),'--out',str(Path(a.out)/f'block_{b}')])
Path(a.out).mkdir(parents=True,exist_ok=True);Path(a.out,'jobs.json').write_text(json.dumps(jobs,indent=2))
if a.execute:
 for job in jobs:subprocess.run(job,check=True)
else:print('Wrote jobs.json. Review runtime and backend, then add --execute.')
