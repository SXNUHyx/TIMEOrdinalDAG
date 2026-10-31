"""Calls the retained author's R implementation; explicit failure if R is absent."""
import json,subprocess,tempfile,shutil,os
from pathlib import Path
import numpy as np
from osem import TemporalOSEM


def fit_original_R(C,levels,p,L,lam,R,seed,temporal=True,contemporaneous=True):
    exe=os.environ.get('RSCRIPT') or shutil.which('Rscript')
    if not exe:raise RuntimeError('Rscript unavailable. Install the original R environment or explicitly choose --backend python-reference.')
    config=dict(C=np.asarray(C).tolist(),levels=list(map(int,levels)),p=p,L=L,lambda_value=lam,R=R,seed=seed,temporal=temporal,contemporaneous=contemporaneous)
    config['lambda']=config.pop('lambda_value')
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/'input.json';dst=Path(td)/'output.json';src.write_text(json.dumps(config))
        proc=subprocess.run([exe,str(Path(__file__).with_name('run_original_R.R')),str(src),str(dst)],capture_output=True,text=True)
        if proc.returncode:raise RuntimeError('Original R OSEM failed: '+proc.stderr[-3000:])
        result=json.loads(dst.read_text())
    m=TemporalOSEM(p,L,lam,R,seed=seed,temporal=temporal,contemporaneous=contemporaneous)
    for name in ['A','B','v','Sigma']:setattr(m,name,np.asarray(result[name],dtype=bool if name=='A' else float))
    m.v=m.v.reshape(-1);m.cuts=[np.asarray(x,dtype=float) for x in result['cuts']];m.levels=list(levels);m.trace=[];m.converged=None
    return m
