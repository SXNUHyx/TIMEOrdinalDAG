"""Optional official Tigramite adapters. Exceptions are recorded, never replaced by fake results."""
import numpy as np


def tigramite_fit(series,L,method='pcmci',alpha=.05):
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    data=pp.DataFrame({i:a for i,a in enumerate(series)},analysis_mode='multiple')
    if method=='lpcmci':
        from tigramite.lpcmci import LPCMCI
        r=LPCMCI(dataframe=data,cond_ind_test=ParCorr(significance='analytic'),verbosity=0).run_lpcmci(tau_max=L,pc_alpha=alpha)
    else:
        from tigramite.pcmci import PCMCI
        engine=PCMCI(dataframe=data,cond_ind_test=ParCorr(significance='analytic'),verbosity=0)
        r=engine.run_pcmciplus(tau_min=0,tau_max=L,pc_alpha=alpha) if method=='pcmciplus' else engine.run_pcmci(tau_min=1,tau_max=L,pc_alpha=alpha,alpha_level=alpha)
    g=r['graph'];p=g.shape[0];adj=np.zeros((L,p,p),bool);directed=adj.copy()
    for l in range(1,L+1):
        adj[l-1]=g[:,:,l]!=''
        directed[l-1]=g[:,:,l]=='-->'
    # Preserve original endpoints including circles / arrowheads in the PAG.
    return adj,directed,g
