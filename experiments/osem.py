"""Temporal-constrained reference reimplementation of latent-Gaussian structural EM.
Retains OSEM's threshold model, Monte Carlo latent E-step, Gaussian DAG M-step.
Not an exact reproduction of BiDAG's MCMC search. On overlapping windows the
row-product criterion is a WORKING score, not a validated time-series likelihood.
"""
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.integrate import quad
from scipy.stats import norm,truncnorm


def regularize(S):
    v,U=np.linalg.eigh((S+S.T)/2);S=(U*np.maximum(v,1e-5))@U.T
    d=np.sqrt(np.diag(S));return S/d[:,None]/d[None,:]


def acyclic(A):
    left=np.sum(A,axis=0).copy();seen=0;todo=list(np.where(left==0)[0])
    while todo:
        i=todo.pop();seen+=1
        for j in np.where(A[i])[0]:
            left[j]-=1
            if left[j]==0:todo.append(j)
    return seen==len(A)


def fit_gaussian(S,A):
    # A[source,target]; B[target,source]
    d=len(S);B=np.zeros((d,d));v=np.diag(S).copy()
    for j in range(d):
        pa=np.flatnonzero(A[:,j])
        if len(pa):
            b=np.linalg.solve(S[np.ix_(pa,pa)],S[pa,j]);B[j,pa]=b;v[j]=max(1e-6,S[j,j]-S[j,pa]@b)
    W=np.linalg.inv(np.eye(d)-B);Cov=W@np.diag(v)@W.T
    scale=np.sqrt(np.diag(Cov));B=B*scale[None,:]/scale[:,None];v=v/scale**2
    return B,v,regularize(Cov)


def search(S,N,lam,allowed,start=None,max_moves=500):
    d=len(S);A=np.zeros((d,d),bool) if start is None else start.copy();cache={}
    def local(j,parentmask):
        key=(j,parentmask.tobytes())
        if key not in cache:
            pa=np.flatnonzero(parentmask)
            r=S[j,j]
            if len(pa):r-=S[j,pa]@np.linalg.solve(S[np.ix_(pa,pa)],S[pa,j])
            cache[key]=-N*.5*np.log(max(r,1e-8))-lam*.5*np.log(max(N,2))*len(pa)
        return cache[key]
    def total(a):return sum(local(j,a[:,j]) for j in range(d))
    score=total(A)
    for move in range(max_moves):
        best=score;winner=None
        for i in range(d):
          for j in range(d):
            if i==j:continue
            if A[i,j]:
                c=A.copy();c[i,j]=False;s=total(c)
                if s>best+1e-8:best=s;winner=c.copy()
                if allowed[j,i] and not A[j,i]:
                    c[j,i]=True
                    if acyclic(c):
                        s=total(c)
                        if s>best+1e-8:best=s;winner=c.copy()
            elif allowed[i,j] and not A[j,i]:
                c=A.copy();c[i,j]=True
                if acyclic(c):
                    s=total(c)
                    if s>best+1e-8:best=s;winner=c.copy()
        if winner is None:return A,score,True
        A=winner;score=best
    return A,score,False


def mask(p,L,contemporaneous=True):
    # Node order: current variables, lag1 variables, ..., lagL variables.
    lag=np.repeat(np.arange(L+1),p);a=lag[:,None]>lag[None,:]
    if contemporaneous:a|=(lag[:,None]==lag[None,:])&~np.eye(len(lag),dtype=bool)
    np.fill_diagonal(a,False);return a


class TemporalOSEM:
    def __init__(self,p,L=2,lam=6,R=20,max_em=10,burn=10,seed=0,temporal=True,contemporaneous=True):
        self.p=p;self.L=L;self.lam=lam;self.R=R;self.max_em=max_em;self.burn=burn;self.seed=seed
        d=p*(L+1);self.allowed=mask(p,L,contemporaneous) if temporal else ~np.eye(d,dtype=bool)
        if not contemporaneous:
            lag=np.repeat(np.arange(L+1),p);self.allowed &= lag[:,None]!=lag[None,:]
    def fit(self,C,levels=None):
        raw=np.asarray(C)
        if raw.ndim!=2 or not np.isfinite(raw).all() or np.any(raw!=np.floor(raw)):
            raise ValueError('Ordinal observations must form a finite integer matrix')
        C=raw.astype(int);n,d=C.shape
        if d!=self.p*(self.L+1) or self.R<1 or self.max_em<1:
            raise ValueError('Invalid model dimension or Monte Carlo/EM count')
        if n<d+2:raise ValueError(f'Only {n} independent/window rows for {d} nodes; increase subjects or reduce dimension')
        self.levels=list(np.max(C,axis=0)+1) if levels is None else list(levels)
        if len(self.levels)!=d or any(q<2 or q!=int(q) for q in self.levels):raise ValueError('Each node needs an integer category count >=2')
        if np.any(C<0) or np.any(C>=np.asarray(self.levels)):raise ValueError('Category code outside declared support')
        self.cuts=[];Y=np.empty_like(C,dtype=float);rng=np.random.default_rng(self.seed)
        for j,q in enumerate(self.levels):
            counts=np.bincount(C[:,j],minlength=q)+.5;c=np.r_[-np.inf,ndtri(np.cumsum(counts)[:-1]/counts.sum()),np.inf]
            self.cuts.append(c);prob=np.cumsum(counts)/counts.sum();prev=np.r_[0,prob[:-1]];Y[:,j]=ndtri((prev[C[:,j]]+prob[C[:,j]])/2)
        S=regularize(np.cov(Y,rowvar=False)+np.eye(d)*.05);A=np.zeros((d,d),bool);self.trace=[];stable=0
        for it in range(self.max_em):
            precision=np.linalg.inv(S);sumYY=np.zeros_like(S)
            for draw in range(self.burn+self.R):
                for j in range(d):
                    mean=Y[:,j]-Y@precision[:,j]/precision[j,j];sd=1/np.sqrt(precision[j,j]);c=self.cuts[j]
                    lo=ndtr((c[C[:,j]]-mean)/sd);hi=ndtr((c[C[:,j]+1]-mean)/sd)
                    u=lo+(hi-lo)*rng.random(n);draw_y=mean+sd*ndtri(np.clip(u,1e-14,1-1e-14))
                    lower=c[C[:,j]];upper=c[C[:,j]+1]
                    bad=(hi-lo<1e-12)|(draw_y<lower)|(draw_y>=upper)
                    if np.any(bad):draw_y[bad]=truncnorm.rvs((lower[bad]-mean[bad])/sd,(upper[bad]-mean[bad])/sd,loc=mean[bad],scale=sd,random_state=rng)
                    Y[:,j]=draw_y
                if draw>=self.burn:sumYY+=Y.T@Y
            S_e=regularize(sumYY/(n*self.R));new,score,search_ok=search(S_e,n,self.lam,self.allowed,A)
            B,v,S=fit_gaussian(S_e,new);changed=np.sum(new!=A);self.trace.append(dict(iteration=it+1,changed=int(changed),score=float(score),search_converged=search_ok))
            stable=stable+1 if changed==0 else 0;A=new
            if stable>=2:break
        self.A=A;self.B=B;self.v=v;self.Sigma=S;self.converged=stable>=2
        if not acyclic(A) or np.any(A&~self.allowed):raise ValueError('Graph search returned a cycle or forbidden edge')
        return self
    def pair_logloss(self,C):
        # Held-out bivariate ordinal composite score, not joint predictive likelihood.
        raw=np.asarray(C)
        if raw.ndim!=2 or not np.isfinite(raw).all() or np.any(raw!=np.floor(raw)):
            raise ValueError('Ordinal observations must form a finite integer matrix')
        C=raw.astype(int);n,d=C.shape
        if d!=self.p*(self.L+1) or self.R<1 or self.max_em<1:
            raise ValueError('Invalid model dimension or Monte Carlo/EM count')
        loss=0.;count=0
        for i in range(d):
          for j in range(i+1,d):
            rho=np.clip(self.Sigma[i,j],-.999,.999);den=np.sqrt(1-rho*rho);a=self.cuts[i];b=self.cuts[j]
            pairs,cnts=np.unique(C[:,[i,j]],axis=0,return_counts=True)
            for (u,v),freq in zip(pairs,cnts):
                value=quad(lambda x:(np.exp(-.5*x*x)/np.sqrt(2*np.pi))*(ndtr((b[v+1]-rho*x)/den)-ndtr((b[v]-rho*x)/den)),a[u],a[u+1],epsabs=1e-7)[0]
                loss-=freq*np.log(max(value,1e-12));count+=freq
        return float(loss/max(count,1))
    def effect(self,source,target,low=0,high=None):
        return oce(self.B,self.v,self.cuts,source,target,low,high)


def oce(B,v,cuts,source,target,low=0,high=None):
    """Same latent stochastic interval intervention as original DIS integral.
    Draw latent source from its fitted marginal Gaussian truncated to the category;
    remove its incoming arrows, propagate through the fitted DAG.
    """
    if source==target:raise ValueError('Choose distinct source and target')
    if high is None:high=len(cuts[source])-2
    d=len(B);W=np.linalg.inv(np.eye(d)-B);S=W@np.diag(v)@W.T;sd_i=np.sqrt(S[source,source])
    b=B.copy();b[source]=0;wd=np.linalg.inv(np.eye(d)-b);vv=np.array(v,copy=True);vv[source]=0
    variance=float((wd@np.diag(vv)@wd.T)[target,target]);coef=wd[target,source]
    if abs(coef)<1e-12:return np.zeros(len(cuts[target])-1)
    sd=np.sqrt(max(variance,1e-12));out=[]
    for level in [low,high]:
        lo,hi=cuts[source][level:level+2];mass=ndtr(hi/sd_i)-ndtr(lo/sd_i)
        if mass<1e-10:raise ValueError('Intervention category has negligible probability')
        probs=[]
        for a,bnd in zip(cuts[target][:-1],cuts[target][1:]):
            f=lambda y:(ndtr((bnd-coef*y)/sd)-ndtr((a-coef*y)/sd))*(np.exp(-.5*(y/sd_i)**2)/(sd_i*np.sqrt(2*np.pi)))/mass
            probs.append(quad(f,lo,hi,epsabs=1e-7)[0])
        out.append(probs)
    return np.asarray(out[1])-np.asarray(out[0])


def ancestors(A,source,target):
    seen={source};todo=[source]
    while todo:
        i=todo.pop()
        for j in np.flatnonzero(A[i]):
            if j==target:return True
            if j not in seen:seen.add(j);todo.append(j)
    return False
