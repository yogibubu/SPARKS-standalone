"""Reflection-complete continuation of a measured local quadratic difference.

Both symmetry-related local jets are retained through fourth order. A bounded
C4 partition is flat through fourth order at each anchor; the chosen periodic
lift has the identity jet through fourth order at zero. No electronic slope
is discarded or projected away at its acquisition geometry.
"""


def reflection_increment(points, specification, *, backend='numpy'):
    """Evaluate the same scalar with numpy or differentiable torch tensors.

    Points are native displacements from the reflection-fixed reference.
    The declared reflection is diagonal, with entries +1 or -1. The partition
    coordinate must be odd under this reflection and periodic.
    """
    if backend=='numpy':
        import numpy as xp
        asarray=lambda value:xp.asarray(value,dtype=float)
        clip=lambda x:xp.clip(x,-1.,1.)
        stack=lambda values:xp.stack(values,axis=-1)
        einsum=xp.einsum
    elif backend=='torch':
        import torch as xp
        asarray=lambda value:points.new_tensor(value)
        clip=lambda x:xp.clamp(x,-1.,1.)
        stack=lambda values:xp.stack(values,dim=-1)
        einsum=xp.einsum
    else:raise ValueError('Explicit numpy or torch backend required')
    offset=asarray(specification['anchor_offset'])
    gradient=asarray(specification['gradient_increment_cm'])
    hessian=asarray(specification['hessian_increment_cm'])
    reflection=asarray(specification['reflection_signs'])
    periodic=set(specification['periodic_indices'])
    axis=specification['partition_axis']
    def local(q):
        delta=q-offset
        lifted=stack([(4*xp.sin(delta[...,i])/3-xp.sin(2*delta[...,i])/6)
                      if i in periodic else delta[...,i] for i in range(len(specification['anchor_offset']))])
        return einsum('...i,i->...',lifted,gradient)+einsum('...i,ij,...j->...',lifted,hessian,lifted)/2
    t=clip(xp.sin(points[...,axis])/xp.sin(offset[axis]))
    weight=.5+(315*t-420*t**3+378*t**5-180*t**7+35*t**9)/256
    return weight*local(points)+(1-weight)*local(points*reflection)


def reflection_increment_transverse(points, specification, bath_indices):
    """Exact transverse first/second derivatives of the same scalar.

    The partition coordinate is retained explicitly and must not belong to the
    transverse derivative list. All periodic coordinate lifts are differentiated
    consistently; no local fourth-order continuation is extrapolated globally.
    """
    import numpy as np
    q=np.asarray(points,float);bath=np.asarray(bath_indices,int)
    if specification['partition_axis'] in bath:
        raise ValueError('The reflection partition axis must be explicit')
    offset=np.asarray(specification['anchor_offset']);g=np.asarray(specification['gradient_increment_cm'])
    h=np.asarray(specification['hessian_increment_cm']);s=np.asarray(specification['reflection_signs'])
    periodic=specification['periodic_indices']
    def local(x):
        delta=x-offset;y=delta.copy();first=np.ones_like(delta);second=np.zeros_like(delta)
        y[:,periodic]=4*np.sin(delta[:,periodic])/3-np.sin(2*delta[:,periodic])/6
        first[:,periodic]=4*np.cos(delta[:,periodic])/3-np.cos(2*delta[:,periodic])/3
        second[:,periodic]=-4*np.sin(delta[:,periodic])/3+2*np.sin(2*delta[:,periodic])/3
        force=g+y@h.T
        gb=force[:,bath]*first[:,bath]
        hb=h[np.ix_(bath,bath)][None]*first[:,bath,None]*first[:,None,bath]
        diagonal=np.arange(len(bath));hb[:,diagonal,diagonal]+=force[:,bath]*second[:,bath]
        return gb,hb
    gp,hp=local(q);gm,hm=local(q*s)
    gm*=s[bath];hm*=s[bath,None]*s[None,bath]
    axis=specification['partition_axis'];t=np.clip(np.sin(q[:,axis])/np.sin(offset[axis]),-1.,1.)
    w=.5+(315*t-420*t**3+378*t**5-180*t**7+35*t**9)/256
    return w[:,None]*gp+(1-w[:,None])*gm,w[:,None,None]*hp+(1-w[:,None,None])*hm
