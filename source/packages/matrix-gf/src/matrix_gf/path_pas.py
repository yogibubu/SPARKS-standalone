"""A path-wide P/A/S proposal from transported anharmonic coupling changes.

Uses supplied local models, not an experimental or full-PES rate target.
The energy proxy screens derivative coverage; it is not an error bound on k.
"""
import numpy as np


def classify_path_modes(positions,frequencies_cm,coupling_matrices_cm,*,
                        temperatures_K,primary_seed=(),primary_budget_cm=10.,
                        auxiliary_budget_cm=1.,degeneracy_tolerance_cm=10.):
    """Choose one fixed hierarchy for consistently tracked path mode labels.

    Diagonal occupation anharmonicity is available in auxiliary axial scans.
    Off-diagonal changes define an edge-weighted coverage problem. A primary
    covers every incident changing pair; a greedy whole-degenerate-group
    selection stops when each uncovered row is below the declared energy
    budget. Large constant couplings remain auxiliary, not automatically P.
    Harmonic pair moments include zero-point motion and the maximum supplied
    temperature. Actual acquired mixed-auxiliary quartics remain necessary.
    """
    s=np.asarray(positions,float);w=np.asarray(frequencies_cm,float)
    x=np.asarray(coupling_matrices_cm,float);t=np.asarray(temperatures_K,float)
    if (s.ndim!=1 or w.ndim!=2 or w.shape[0]!=len(s) or x.shape!=(len(s),w.shape[1],w.shape[1])
            or np.any(w<=0) or t.ndim!=1 or not len(t) or np.any(t<=0)
            or not all(np.all(np.isfinite(a)) for a in (s,w,x,t))
            or not np.allclose(x,x.swapaxes(1,2))
            or primary_budget_cm<=0 or auxiliary_budget_cm<0 or degeneracy_tolerance_cm<0):
        raise ValueError('finite positive consistent path modes and symmetric coupling matrices required')
    n=w.shape[1];zero=np.flatnonzero(np.isclose(s,0.,rtol=0,atol=1e-12))
    if len(zero)!=1:raise ValueError('one explicit TS anchor required')
    p=set(primary_seed)
    if any(not isinstance(i,(int,np.integer)) or isinstance(i,bool) or not 0<=i<n for i in p):
        raise ValueError('valid primary seed indices required')
    groups=[{i} for i in range(n)]
    for i in range(n):
        for j in range(i):
            if np.min(abs(w[:,i]-w[:,j]))<=degeneracy_tolerance_cm:
                a=next(g for g in groups if i in g);b=next(g for g in groups if j in g)
                if a is not b:a.update(b);groups.remove(b)
    for g in groups:
        if p&g:p.update(g)
    mean=.5+1/np.expm1(np.minimum(1.438776877*np.min(w,axis=0)/max(t),700.))
    moments=mean[:,None]*mean[None,:]
    variation=np.max(abs(x-x[int(zero[0])]),axis=0)*moments
    np.fill_diagonal(variation,0.)
    original=variation.copy();history=[]
    while True:
        residual=variation.copy()
        if p:residual[list(p),:]=0;residual[:,list(p)]=0
        scores=residual.sum(axis=1)
        if max(scores,default=0)<=primary_budget_cm:break
        available=[g for g in groups if not g<=p]
        chosen=max(available,key=lambda g:(sum(scores[list(g)]),-len(g),-min(g)))
        p.update(chosen);history.append(sorted(chosen))
    magnitude=np.max(abs(x),axis=0)*moments;importance=magnitude.sum(axis=1)
    a=set(i for i in range(n) if i not in p and importance[i]>auxiliary_budget_cm)
    for group in groups:
        if not group&p and group&a:a.update(group)
    spectator=set(range(n))-p-a
    return dict(primary_modes=sorted(p),auxiliary_modes=sorted(a),spectator_modes=sorted(spectator),
        degeneracy_groups=[sorted(g) for g in groups],variation_energy_proxy_cm=original,
        residual_uncovered_row_proxy_cm=scores,thermal_pair_moments=moments,
        magnitude_row_proxy_cm=importance,promotion_groups=history,
        primary_budget_cm=primary_budget_cm,auxiliary_budget_cm=auxiliary_budget_cm,
        constant_classes_along_entire_supplied_path=True,rate_error_certified=False,
        prerequisite='mode labels and degeneracy blocks must already be consistently transported',
        mixed_auxiliary_quartics_obtained_from_gradients_alone=False)
