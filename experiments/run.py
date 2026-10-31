import argparse,json,time,sys,platform,os
from pathlib import Path
import numpy as np
import pandas as pd
from osem import TemporalOSEM,ancestors
from data import windows,subject_split,fit_maps,encode,heldout_r2
from simulations import generate,metrics,oracle
from r_backend import fit_original_R

def fit_model(C,levels,p,L,lam,args,seed):
    if args.backend=='original-r':return fit_original_R(C,levels,p,L,lam,args.mc,seed,temporal=not args.unconstrained,contemporaneous=not args.no_contemp)
    return TemporalOSEM(p,L,lam,args.mc,args.max_em,seed=seed,temporal=not args.unconstrained,contemporaneous=not args.no_contemp).fit(C,levels)

from baselines import tigramite_fit


def dump(path,obj):
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),encoding='utf-8')
    temporary.replace(path)


def raw_subjects(path,columns):
    d=pd.read_csv(path,parse_dates=['time']);sub=[];audit=[]
    for sid,g in d.groupby('subject',sort=True):
        g=g.sort_values('time')
        if g.time.duplicated().any():raise ValueError('Duplicate subject-minute keys')
        groups=g.time.diff().ne(pd.Timedelta(minutes=1)).cumsum();segs=[]
        for _,v in g.groupby(groups):
            a=v[columns].to_numpy(float);ok=np.isfinite(a).all(1)
            starts=np.flatnonzero(ok&~np.r_[False,ok[:-1]]);ends=np.flatnonzero(ok&~np.r_[ok[1:],False])+1
            segs.extend(a[x:y] for x,y in zip(starts,ends) if y-x>=100)
        if segs:sub.append(segs);audit.append(dict(subject=str(sid),rows=len(g),retained_rows=sum(map(len,segs)),segments=len(segs)))
    return sub,audit


def run_one(subjects,args,out,meta=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (mapping,validation,analysis),audit,splitids=subject_split(subjects,args.seed,calibrate=args.calibrate,map_fraction=args.mapping_fraction,tune_fraction=args.tuning_fraction)
    p=subjects[0][0].shape[1];L=args.lag
    Xmap,_=windows(mapping,L,stride=args.stride);Xval,_=windows(validation,L,stride=args.stride)
    Xana,ids=windows(analysis,L,stride=args.stride,landmark=args.mode=='landmark',seed=args.seed)
    maps,means=fit_maps(Xmap,p,L,args.q,args.scheme);C=encode(Xana,maps);V=encode(Xval,maps)
    levels=[len(maps[j%p])+1 for j in range(p*(L+1))]
    np.savez_compressed(out/'ordinal_input.npz',C=C,validation=V,levels=np.asarray(levels))
    models=[];tuning=[]
    for lam in map(float,args.penalties.split(',')):
        tic=time.perf_counter();m=fit_model(C,levels,p,L,lam,args,args.seed)
        loss=m.pair_logloss(V);models.append(m);tuning.append(dict(lam=lam,validation_pair_logloss=loss,seconds=time.perf_counter()-tic,converged=m.converged))
    best=int(np.argmin([x['validation_pair_logloss'] for x in tuning]));m=models[best]
    source=p;target=p-1
    if p<2 or L<1:raise ValueError('At least two base variables and lag>=1 are required')
    np.savez_compressed(out/'point_model.npz',A=m.A,B=m.B,v=m.v,Sigma=m.Sigma)
    dump(out/'model.json',dict(maps=maps,latent_cuts=m.cuts,trace=m.trace,tuning=tuning,selected_lambda=m.lam,levels=levels,split_subject_ids=splitids,mode=args.mode,criterion='working ordinal window score' if args.mode!='landmark' else 'independent-subject landmark rows',N_rows=len(C),S_analysis=len(analysis),calibrated=args.calibrate))
    pd.DataFrame(audit).to_csv(out/'partition_counts.csv',index=False)
    pd.DataFrame(heldout_r2(Xana,maps,means)).to_csv(out/'heldout_discretization.csv',index=False)
    # These outcomes have not been used for maps or tuning.
    summary=dict(backend=args.backend,seed=args.seed,scenario=getattr(args,'scenario',None),method='CART_OSEM_OCE_'+args.scheme,mode=args.mode,N=len(C),S=len(analysis),lambda_selected=m.lam,converged=m.converged,anc_source_to_target=ancestors(m.A,source,target))
    if meta:summary.update(metrics(m.A,meta,L,p))
    dump(out/'point_summary.json',summary)
    e0=m.effect(source,target);truth=None
    if meta and not args.calibrate:truth=oracle(meta,maps)
    if truth is not None:np.save(out/'oracle_OCE.npy',truth)
    bootstrap_A=[];bootstrap_e=[];bootstatus=[];rng=np.random.default_rng(args.seed+1000)
    for b in range(args.bootstrap):
        chosen=[analysis[i] for i in rng.integers(len(analysis),size=len(analysis))]
        if args.block:
            if args.block<=L:raise ValueError('block length must exceed L')
            processed=[]
            for sub in chosen:
                blocks=[]
                for a in sub:
                    if len(a)<=args.block:blocks.append(a);continue
                    for start in rng.integers(len(a)-args.block+1,size=int(np.ceil(len(a)/args.block))):blocks.append(a[start:start+args.block])
                processed.append(blocks)
            chosen=processed
        try:
            xx,_=windows(chosen,L,args.stride,landmark=args.mode=='landmark',seed=args.seed+b)
            mm=fit_model(encode(xx,maps),levels,p,L,m.lam,args,args.seed+b+1)
            bootstrap_A.append(mm.A);bootstrap_e.append(mm.effect(source,target));bootstatus.append(dict(b=b,status='ok',converged=mm.converged,ancestor=ancestors(mm.A,source,target)))
        except Exception as ex:bootstatus.append(dict(b=b,status='FAILED',error=repr(ex)))
        if (b+1)%25==0:dump(out/'bootstrap_progress.json',dict(completed=b+1,requested=args.bootstrap,failed=sum(x['status']=='FAILED' for x in bootstatus)))
    dump(out/'bootstrap_status.json',bootstatus)
    summary.update(B_requested=args.bootstrap,B_success=len(bootstrap_e),B_failed=args.bootstrap-len(bootstrap_e),B_nonconverged=sum(x.get('converged') is False for x in bootstatus),seconds_model_selection=sum(x['seconds'] for x in tuning))
    dump(out/'point_summary.json',summary)
    effectrows=[]
    if bootstrap_A:
        aa=np.asarray(bootstrap_A);ee=np.asarray(bootstrap_e);freq=aa.mean(0)
        np.savez_compressed(out/'bootstrap_models.npz',A=aa,OCE=ee)
        pd.DataFrame([dict(source=i,target=j,frequency=float(np.mean(aa[:,i,j]|aa[:,j,i])),directed_frequency=freq[i,j],orientation_conditional=freq[i,j]/max(freq[i,j]+freq[j,i],1e-12)) for i in range(len(freq)) for j in range(len(freq)) if i!=j]).to_csv(out/'edge_frequencies.csv',index=False)
        display=[]
        for i in range(len(freq)):
          for j in range(i+1,len(freq)):
            adjacency=float(np.mean(aa[:,i,j]|aa[:,j,i]));direction=freq[i,j]/max(freq[i,j]+freq[j,i],1e-12)
            if adjacency>=.8:
                endpoint='-->' if direction>=.8 else ('<--' if direction<=.2 else '---')
                display.append(dict(node1=i,node2=j,adjacency_frequency=adjacency,direction_fraction=direction,endpoint=endpoint))
        pd.DataFrame(display,columns=['node1','node2','adjacency_frequency','direction_fraction','endpoint']).to_csv(out/'stability_display.csv',index=False)
        for k,value in enumerate(e0):
            lo,hi=np.quantile(ee[:,k],[.025,.975]);t=None if truth is None else truth[k]
            effectrows.append(dict(outcome_state=k,estimate=value,lower=lo,upper=hi,truth=t,bias=None if t is None else value-t,covered=None if t is None else bool(lo<=t<=hi),ancestor_frequency=np.mean([x['ancestor'] for x in bootstatus if x['status']=='ok']),B_success=len(ee)))
    else:effectrows=[dict(outcome_state=k,estimate=v,truth=None if truth is None else truth[k]) for k,v in enumerate(e0)]
    pd.DataFrame(effectrows).to_csv(out/'OCE_results.csv',index=False)
    # Category transitions from exact consecutive observations inside each segment.
    transition=[]
    for j,c in enumerate(maps):
        q=len(c)+1;counts=np.zeros((q,q),int)
        for sub in analysis:
            for a in sub:
                z=np.searchsorted(c,a[:,j],side='right');np.add.at(counts,(z[:-1],z[1:]),1)
        for a in range(q):
            for b in range(q):transition.append(dict(variable=j,from_state=a,to_state=b,n=counts[a,b],probability=counts[a,b]/counts[a].sum() if counts[a].sum() else np.nan))
    pd.DataFrame(transition).to_csv(out/'state_transitions.csv',index=False)
    # Continuous baselines use EXACT analysis rows/spans at input segment level.
    baseline=[]
    for method in filter(None,args.baselines.split(',')):
        if args.mode=='landmark':
            baseline.append(dict(method=method,status='NOT_COMPARABLE',reason='Temporal methods require trajectories; report in overlap benchmark'))
            continue
        try:
            tic=time.perf_counter();adj,directed,endpoints=tigramite_fit([a for sub in analysis for a in sub],L,method)
            np.save(out/f'{method}_endpoints.npy',endpoints)
            pred=np.zeros((p*(L+1),p*(L+1)),bool)
            for l in range(1,L+1):pred[l*p:(l+1)*p,:p]=adj[l-1]
            result=dict(method=method,seconds=time.perf_counter()-tic,status='ok',metric='lagged adjacency only')
            if meta:result.update({k:v for k,v in metrics(pred,meta,L,p).items() if not k.startswith('contemp_')})
            baseline.append(result)
        except Exception as ex:baseline.append(dict(method=method,status='FAILED',error=repr(ex)))
    pd.DataFrame(baseline).to_csv(out/'baselines.csv',index=False)
    # Translation consistency diagnostics; boundary effects prevent calling this stationarity proof.
    translations=[]
    for lag in range(1,L):
      for i in range(p):
       for j in range(p):translations.append(dict(lag=lag,source=i,target=j,current_edge=bool(m.A[lag*p+i,j]),translated_edge=bool(m.A[(lag+1)*p+i,p+j])))
    pd.DataFrame(translations).to_csv(out/'translation_diagnostics.csv',index=False)
    if args.loo:
        records=[]
        for k in range(len(analysis)):
            sub=[x for i,x in enumerate(analysis) if i!=k]
            try:
                xx,_=windows(sub,L,args.stride,landmark=args.mode=='landmark',seed=args.seed)
                mm=fit_model(encode(xx,maps),levels,p,L,m.lam,args,args.seed)
                records.append(dict(omitted=k,jaccard=np.sum(mm.A[:,:p]&m.A[:,:p])/max(1,np.sum(mm.A[:,:p]|m.A[:,:p]))))
            except Exception as ex:records.append(dict(omitted=k,error=repr(ex)))
        pd.DataFrame(records).to_csv(out/'leave_one_subject_out.csv',index=False)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['simulate','real']);p.add_argument('--out',default='results/pilot');p.add_argument('--input');p.add_argument('--columns',default='Steps,HR');p.add_argument('--backend',choices=['python-reference','original-r'],default='python-reference');p.add_argument('--mode',choices=['overlap','landmark'],default='overlap');p.add_argument('--mapping-fraction',type=float,default=.4);p.add_argument('--tuning-fraction',type=float,default=.2);p.add_argument('--subjects',type=int,default=30);p.add_argument('--length',type=int,default=300);p.add_argument('--p',type=int,default=3);p.add_argument('--lag',type=int,default=2);p.add_argument('--q',type=int,default=4);p.add_argument('--scheme',choices=['cart','quantile','width'],default='cart');p.add_argument('--penalties',default='2,6,10');p.add_argument('--mc',type=int,default=20);p.add_argument('--max-em',type=int,default=10);p.add_argument('--bootstrap',type=int,default=0);p.add_argument('--block',type=int);p.add_argument('--stride',type=int,default=1);p.add_argument('--seed',type=int,default=2026);p.add_argument('--reps',type=int,default=1);p.add_argument('--rho',type=float,default=.5);p.add_argument('--noise',type=float,default=1);p.add_argument('--heterogeneity',type=float,default=0);p.add_argument('--scenario',default='gaussian');p.add_argument('--calibrate',action='store_true');p.add_argument('--unconstrained',action='store_true');p.add_argument('--no-contemp',action='store_true');p.add_argument('--loo',action='store_true');p.add_argument('--baselines',default='pcmci,pcmciplus,lpcmci');a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);dump(root/'arguments.json',dict(arguments=vars(a),python=sys.version,platform=platform.platform()))
    if a.task=='real':
        subjects,audit=raw_subjects(a.input,a.columns.split(','));pd.DataFrame(audit).to_csv(root/'raw_subject_counts.csv',index=False);run_one(subjects,a,root)
    else:
        rows=[];initial=a.seed
        for rep in range(a.reps):
            a.seed=initial+rep;sub,meta=generate(a.subjects,a.length,a.p,a.noise,a.rho,a.heterogeneity,a.scenario,a.seed)
            folder=root/f'rep_{rep:03d}';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'truth.npz',A=meta['A'],B0=meta['B0'],states=meta['states'])
            try:row=run_one(sub,a,folder,meta);row['rep']=rep;rows.append(row)
            except Exception as ex:rows.append(dict(rep=rep,status='FAILED',error=repr(ex)))
            pd.DataFrame(rows).to_csv(root/'simulation_summary.csv',index=False);print('rep',rep,'saved',flush=True)
        if any(row.get('status')=='FAILED' for row in rows):sys.exit(1)
