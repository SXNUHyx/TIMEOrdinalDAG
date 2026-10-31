import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from osem import oce


def generate(S=30,T=300,p=3,noise=1,rho=.5,heterogeneity=0,scenario='gaussian',seed=0):
    rng=np.random.default_rng(seed);A=np.zeros((2,p,p));A[0]=np.eye(p)*rho
    for j in range(1,p):A[0,j,j-1]=.3
    A[0,p-1,0]=.3  # Nonzero lag-1 effect for the prespecified OCE contrast.
    A[1,p-1,0]=.2;B0=np.zeros((p,p))
    if scenario=='contemporaneous':B0[p-1,0]=.25
    W=np.linalg.inv(np.eye(p)-B0)
    if max(abs(np.linalg.eigvals(np.block([[W@A[0],W@A[1]],[np.eye(p),np.zeros((p,p))]]))))>=.99:raise ValueError('Unstable parameter setting')
    # Vary outcome innovation noise while other channels stay at unit noise;
    # a common global scale would not change signal-to-noise after discretization.
    noise_scales=np.ones(p);noise_scales[-1]=noise
    sub=[];states=[]
    for s in range(S):
        x=np.zeros((T+300,p));h=0.;state=0;st=[];offset=rng.normal(0,heterogeneity,p)
        for t in range(2,len(x)):
            if rng.random()<.02:state=1-state
            B=A.copy()
            if scenario=='switching' and state:B[0,1,0]=0;B[0,0,1]=.3
            h=.7*h+rng.normal();past=[x[t-1],x[t-2]]
            eta=sum(B[l]@(np.tanh(past[l]) if scenario=='nonlinear' else past[l]) for l in range(2))
            eps=rng.laplace(0,noise_scales/np.sqrt(2),p) if scenario=='nongaussian' else rng.normal(0,noise_scales,p)
            x[t]=W@(eta+eps+(.6*h if scenario=='confounded' else 0));st.append(state)
        values=x[300:]+offset
        if scenario=='derived':values=np.column_stack([values,values[:,0]+values[:,-1],np.digitize(values[:,0],[-.5,.5])])
        sub.append([values]);states.append(np.asarray(st[-T:]))
    return sub,dict(A=A,B0=B0,noise=noise,noise_scales=noise_scales,heterogeneity=heterogeneity,scenario=scenario,states=np.asarray(states))


def true_edges(meta,L,p):
    if meta['scenario'] in ['derived','switching']:return None
    G=np.zeros(((L+1)*p,p),bool);G[:p]=meta['B0'].T!=0
    for l in range(1,min(2,L)+1):G[l*p:(l+1)*p]=meta['A'][l-1].T!=0
    return G


def metrics(pred,meta,L,p):
    if meta['scenario'] in ['derived','switching']:return {}
    # Compare union of requested and true lags; extra and omitted lags count.
    H=max(L,2);G=np.zeros(((H+1)*p,p),bool);P=G.copy();P[:(L+1)*p]=pred[:,:p]
    G[:p]=meta['B0'].T!=0
    for l in [1,2]:G[l*p:(l+1)*p]=meta['A'][l-1].T!=0
    def report(x,y,contemporaneous=False):
        tp=np.sum(x&y);fp=np.sum(x&~y);fn=np.sum(~x&y)
        reversals=int(np.sum(x & y.T & ~y & ~x.T)) if contemporaneous else 0
        return dict(SHD=int(fp+fn-reversals),TPR=float(tp/max(1,tp+fn)),FDR=float(fp/max(1,tp+fp)),F1=float(2*tp/max(1,2*tp+fp+fn)))
    result=report(P[p:],G[p:]);result.update({'contemp_'+k:v for k,v in report(P[:p],G[:p],contemporaneous=True).items()});return result


def oracle(meta,maps,source_var=0,target_var=None,low=0,high=None):
    """Exact finite unrolling of stationary homogeneous Gaussian VAR(2).
    Build joint initial two slices, then true structural transitions. Source is t-1.
    Initial joint law uses a Cholesky nuisance factorization, not causal claims.
    Thresholds are in observed Gaussian units; use only when calibration=False.
    """
    if meta['scenario'] not in ['gaussian','contemporaneous'] or meta.get('heterogeneity',0)!=0:return None
    A=meta['A'];B0=meta['B0'];p=len(B0);target_var=p-1 if target_var is None else target_var
    W=np.linalg.inv(np.eye(p)-B0);F=np.block([[W@A[0],W@A[1]],[np.eye(p),np.zeros((p,p))]])
    variances=np.asarray(meta.get('noise_scales',np.full(p,meta['noise'])))**2
    Q=np.zeros_like(F);Q[:p,:p]=W@np.diag(variances)@W.T
    V=solve_discrete_lyapunov(F,Q)
    # Initial order t-3,t-2 (reverse companion state order).
    ix=np.r_[np.arange(p,2*p),np.arange(p)];V0=V[np.ix_(ix,ix)]
    B=np.zeros((4*p,4*p));v=np.zeros(4*p)
    for j in range(2*p):
        if j:
            b=np.linalg.solve(V0[:j,:j],V0[:j,j]);B[j,:j]=b;v[j]=V0[j,j]-V0[j,:j]@b
        else:v[j]=V0[j,j]
    for sl in [2,3]:
        B[sl*p:(sl+1)*p,sl*p:(sl+1)*p]=B0
        for l in [1,2]:B[sl*p:(sl+1)*p,(sl-l)*p:(sl-l+1)*p]=A[l-1]
        v[sl*p:(sl+1)*p]=variances
    cuts=[np.r_[-np.inf,maps[j%p],np.inf] for j in range(4*p)]
    return oce(B,v,cuts,2*p+source_var,3*p+target_var,low,high)
