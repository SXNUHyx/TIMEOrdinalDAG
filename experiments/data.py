import numpy as np
from sklearn.tree import DecisionTreeRegressor


def windows(subjects,L,stride=1,landmark=False,seed=0):
    """subject -> uninterrupted segments -> arrays; node order current, lag1,..."""
    rows=[];ids=[];rng=np.random.default_rng(seed)
    for sid,segments in enumerate(subjects):
        temp=[]
        for a in segments:
            for t in range(L,len(a),stride):temp.append(np.concatenate([a[t-l] for l in range(L+1)]))
        if landmark and temp:temp=[temp[rng.integers(len(temp))]]
        rows.extend(temp);ids.extend([sid]*len(temp))
    if not rows:raise ValueError('No usable windows')
    return np.asarray(rows),np.asarray(ids)


def subject_split(subjects,seed=0,calibrate=True,map_fraction=.4,tune_fraction=.2):
    """Independent subject groups: 40% cutpoints, 20% tuning, remainder structure.
    For calibrated analyses each subject's initial 20% calibrates and is discarded.
    """
    rng=np.random.default_rng(seed);ids=rng.permutation(len(subjects))
    n=len(ids);n1=max(1,int(map_fraction*n));n2=max(1,int(tune_fraction*n))
    if n-n1-n2<2:raise ValueError('Need at least 5 subjects; substantially more for inference')
    clean=[];audit=[]
    for i,sub in enumerate(subjects):
        if calibrate:
            # Initial fifth of the SUBJECT timeline, not the first fifth of
            # every segment. Later segments never calibrate earlier outcomes.
            ncal=max(2,sum(map(len,sub))//5)
            calibration=np.concatenate(sub,axis=0)[:ncal]
            mu=calibration.mean(0);sd=calibration.std(0);sd[sd<1e-6]=1
            remaining=ncal;segs=[]
            for a in sub:
                cut=min(remaining,len(a));remaining-=cut
                if cut<len(a):segs.append((a[cut:]-mu)/sd)
            if not segs:raise ValueError('Calibration leaves no analysis observations')
        else:segs=[a.copy() for a in sub]
        clean.append(segs);audit.append(dict(subject_index=i,raw_rows=sum(map(len,sub)),analysis_rows=sum(map(len,segs)),calibrated=calibrate))
    sets=[[clean[i] for i in group] for group in [ids[:n1],ids[n1:n1+n2],ids[n1+n2:]]]
    return sets,audit,[a.tolist() for a in [ids[:n1],ids[n1:n1+n2],ids[n1+n2:]]]


def fit_maps(raw,p,L,Q=4,scheme='cart',target=None):
    """One stationary map per base variable, applied identically at all lags.
    Only the independent mapping cohort supplies supervised training pairs.
    Predict current HR from each OTHER variable's lag1. HR never predicts itself.
    """
    target=p-1 if target is None else target;maps=[];means=[];response=raw[:,target]
    for j in range(p):
        x=raw[:,p+j] if L else raw[:,j]
        unique=np.unique(x)
        if len(unique)<=Q:c=(unique[:-1]+unique[1:])/2
        elif scheme=='cart' and j!=target:
            tree=DecisionTreeRegressor(max_leaf_nodes=Q,min_samples_leaf=max(5,len(x)//50),random_state=0).fit(x[:,None],response)
            c=np.sort(tree.tree_.threshold[tree.tree_.children_left!=-1])
        elif scheme=='width':c=np.linspace(x.min(),x.max(),Q+1)[1:-1]
        else:c=np.quantile(x,np.arange(1,Q)/Q)
        c=np.unique(c);z=np.searchsorted(c,x,side='right')
        if not len(c):raise ValueError('Constant variable in map cohort')
        maps.append(c);means.append([float(response[z==k].mean()) if np.any(z==k) else float(response.mean()) for k in range(len(c)+1)])
    return maps,means


def encode(raw,maps):
    p=len(maps);return np.column_stack([np.searchsorted(maps[j%p],raw[:,j],side='right') for j in range(raw.shape[1])])


def heldout_r2(raw,maps,means,target=None):
    p=len(maps);target=p-1 if target is None else target;response=raw[:,target];out=[]
    for j,c in enumerate(maps):
        # Predict held-out HR using a predictor's lag1 category and MAP-cohort means.
        if j==target:continue
        z=np.searchsorted(c,raw[:,p+j],side='right');pred=np.asarray(means[j])[z]
        sst=np.sum((response-response.mean())**2)
        out.append(dict(variable=j,heldout_R2=float(1-np.sum((response-pred)**2)/sst) if sst>0 else np.nan))
    return out
