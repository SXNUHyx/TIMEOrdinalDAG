import numpy as np
from data import windows,fit_maps,encode,subject_split
from osem import TemporalOSEM,oce,acyclic,mask
from simulations import generate,oracle
# Exact source/target orientation and temporal mask.
p,L=2,2;a=mask(p,L);assert not a[0,2] and a[2,0] and a[0,1]
x=np.arange(24).reshape(12,2);y=x+100
X,ids=windows([[x],[y]],2);assert len(X)==20
assert np.array_equal(X[0],np.r_[x[2],x[1],x[0]])
assert not np.any((X[:,:2]<100).any(1)&(X[:,2:]>=100).any(1))
subjects,meta=generate(S=15,T=50,p=2,seed=1)
(groups,audit,splitids)=subject_split(subjects,1,False)
assert not set(splitids[0])&set(splitids[1]) and not set(splitids[0])&set(splitids[2])
M,_=windows(groups[0],1);Z,_=windows(groups[2],1);maps,means=fit_maps(M,2,1,3,'cart')
C=encode(Z,maps);levels=[len(maps[j%2])+1 for j in range(4)]
f=TemporalOSEM(2,1,6,R=3,max_em=4,seed=2).fit(C,levels)
assert acyclic(f.A) and not np.any(f.A&~f.allowed)
e=f.effect(2,1);assert np.isclose(e.sum(),0,atol=1e-6)
assert np.allclose(e,-f.effect(2,1,low=levels[2]-1,high=0),atol=1e-6)
B=np.array([[0,0],[.5,0]]);v=np.ones(2);cuts=[np.array([-np.inf,0,np.inf])]*2
forward=oce(B,v,cuts,0,1);assert forward[1]>0 and np.isclose(forward.sum(),0,atol=1e-6)
assert np.allclose(oce(B,v,cuts,1,0),0)
t=oracle(meta,maps);assert t is not None and np.isclose(t.sum(),0,atol=1e-6)
print('PASS: subject separation, lag boundaries, mask orientation, DAG acyclicity, latent OCE normalization/sign/no-path, Gaussian oracle')
