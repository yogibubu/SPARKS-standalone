"""Local IRC potential jet in an orthonormal covariant saddle frame."""
from itertools import permutations
import numpy as np


def saddle_irc_potential_jet(hessian_cm, covariant_cubic_cm, covariant_quartic_cm,
                            *, tangent=None):
    """Electronic V2/V3/V4 along mass-weighted IRC arc, at an index-one TS.

    Inputs are covariant tensors in the SAME orthonormal frame at the saddle,
    with energy in cm^-1 and arc in bohr*sqrt(amu). Ordinary curvilinear partial
    tensors must first be transformed by their metric owner. No transverse ZPE
    or SCT effective mass is included by this electronic-potential helper.

    k_a=-C_arr/(H_aa-2H_rr), V3=C_rrr and
    V4=D_rrrr+6*C(k,t,t)+3*H(k,k)-4*H_rr*|k|^2.
    """
    H=np.asarray(hessian_cm)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or len(H)<2:
        raise ValueError('at least two orthonormal saddle coordinates required')
    n=len(H)
    tensors=[]
    for order,raw in [(2,H),(3,covariant_cubic_cm),(4,covariant_quartic_cm)]:
        a=np.asarray(raw)
        if a.shape != (n,)*order or np.iscomplexobj(a) or not np.all(np.isfinite(a)):
            raise ValueError('finite real complete covariant derivative tensors required')
        if any(not np.allclose(a,a.transpose(p),rtol=1.e-10,atol=1.e-12)
               for p in permutations(range(order))):
            raise ValueError('symmetric covariant derivative tensors required')
        tensors.append(np.array(a,float,copy=True))
    H,C,D=tensors
    eigenvalues,B=np.linalg.eigh(H)
    if np.count_nonzero(eigenvalues<0)!=1 or np.any(eigenvalues==0):
        raise ValueError('index-one saddle without zero modes required')
    t=B[:,0]
    if tangent is not None:
        supplied=np.asarray(tangent)
        if supplied.shape!=(n,) or np.iscomplexobj(supplied) or not np.all(np.isfinite(supplied)):
            raise ValueError('finite tangent in the same orthonormal frame required')
        if np.linalg.norm(supplied)==0:
            raise ValueError('nonzero saddle tangent required')
        supplied=supplied/np.linalg.norm(supplied)
        if abs(supplied@t)<1-1.e-9:
            raise ValueError('tangent must be the unstable eigenvector')
        t*=np.sign(supplied@t)
    projected=np.einsum('ijk,ia,j,k->a',C,B[:,1:],t,t)
    curvature=B[:,1:]@(-projected/(eigenvalues[1:]-2*eigenvalues[0]))
    direct=float(np.einsum('ijkl,i,j,k,l',D,t,t,t,t))
    cubic=float(6*np.einsum('ijk,i,j,k',C,curvature,t,t))
    quadratic=float(3*curvature@H@curvature-4*eigenvalues[0]*(curvature@curvature))
    return dict(potential_second_cm=float(eigenvalues[0]),
        potential_third_cm=float(np.einsum('ijk,i,j,k',C,t,t,t)),
        potential_fourth_cm=direct+cubic+quadratic,curvature=curvature,tangent=t,
        direct_quartic_cm=direct,cubic_curvature_quartic_cm=cubic,
        hessian_arc_quartic_cm=quadratic,transverse_energy_included=False,
        derivative_convention='covariant_orthonormal_at_stationary_saddle')
