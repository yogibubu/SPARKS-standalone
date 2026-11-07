"""Positive two-coordinate Laplace--Beltrami blocks for primary scattering.

Vertex quadrature with lumped physical measure gives a block-tridiagonal
Hamiltonian and reproduces the separable finite-difference leads exactly.
The reactive coordinate is open; transverse endpoint nodes are Dirichlet.
Potential values must come from a frozen qualified analytic function. Native
covariant metrics are geometric data and have no electronic backend callback.
"""
import numpy as np


def primary_fem_blocks(reactive_grid,transverse_grid,potential_cm,covariant_metric,*,kinetic_prefactor_cm):
    s=np.asarray(reactive_grid,float);u=np.asarray(transverse_grid,float);V=np.asarray(potential_cm,float)
    if (s.ndim!=1 or u.ndim!=1 or len(s)<5 or len(u)<5 or np.any(np.diff(s)<=0) or np.any(np.diff(u)<=0)
            or not np.allclose(np.diff(s),s[1]-s[0]) or not np.allclose(np.diff(u),u[1]-u[0])
            or V.shape!=(len(s),len(u)-2) or not np.all(np.isfinite(V))
            or not np.isfinite(kinetic_prefactor_cm) or kinetic_prefactor_cm<=0):
        raise ValueError('uniform grids, interior transverse potential and positive kinetic prefactor required')
    ds=s[1]-s[0];du=u[1]-u[0];padded=np.r_[s[0]-ds,s,s[-1]+ds]
    g=np.asarray(covariant_metric(padded[:,None],u[None,:]),float)
    if g.shape!=(len(padded),len(u),2,2) or not np.all(np.isfinite(g)) or not np.allclose(g,g.swapaxes(-1,-2),atol=1e-10):
        raise ValueError('matching finite symmetric covariant metric required')
    eig=np.linalg.eigvalsh(g)
    if np.min(eig)<=0:raise ValueError('primary metric must be positive definite; no eigenvalue clipping')
    volume=np.sqrt(np.linalg.det(g));inverse=np.linalg.inv(g)
    ons=np.zeros((len(padded),len(u),len(u)));hop=np.zeros((len(padded)-1,len(u),len(u)))
    su=[0,1,0,1];uu=[0,0,1,1]
    derivatives_s=np.array([[-1,1,0,0],[-1,1,0,0],[0,0,-1,1],[0,0,-1,1]],float)/ds
    derivatives_u=np.array([[-1,0,1,0],[0,-1,0,1],[-1,0,1,0],[0,-1,0,1]],float)/du
    stiffness=np.zeros((len(padded)-1,len(u)-1,4,4))
    for k in range(4):
        local=inverse[su[k]:su[k]+len(padded)-1,uu[k]:uu[k]+len(u)-1]
        weight=volume[su[k]:su[k]+len(padded)-1,uu[k]:uu[k]+len(u)-1]*ds*du*kinetic_prefactor_cm/4
        a=derivatives_s[k];b=derivatives_u[k]
        stiffness+=weight[...,None,None]*(local[...,0,0,None,None]*np.outer(a,a)
            +local[...,0,1,None,None]*(np.outer(a,b)+np.outer(b,a))+local[...,1,1,None,None]*np.outer(b,b))
    cells_s=np.arange(len(padded)-1)[:,None];cells_u=np.arange(len(u)-1)[None,:]
    for a in range(4):
        for b in range(4):
            if su[a]==su[b]:
                np.add.at(ons,(cells_s+su[a],cells_u+uu[a],cells_u+uu[b]),stiffness[...,a,b])
            elif su[b]==su[a]+1:
                np.add.at(hop,(cells_s,cells_u+uu[a],cells_u+uu[b]),stiffness[...,a,b])
    mass=volume*ds*du
    ons=ons[1:-1,1:-1,1:-1];hop=hop[1:-1,1:-1,1:-1];mass=mass[1:-1,1:-1]
    ons/=np.sqrt(mass[:,:,None]*mass[:,None,:])
    hop/=np.sqrt(mass[:-1,:,None]*mass[1:,None,:])
    index=np.arange(len(u)-2);ons[:,index,index]+=V
    return dict(onsite_cm=ons,hopping_cm=hop,
        minimum_covariant_metric_eigenvalue=float(np.min(eig)),physical_measure='sqrt(det(g)) ds du',
        quantization='induced two-primary Laplace-Beltrami; no additional full-molecule pseudopotential inserted')
