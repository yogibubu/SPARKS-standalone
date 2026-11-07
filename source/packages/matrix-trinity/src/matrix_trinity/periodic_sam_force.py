"""Periodic SAM forces from amplitude, phase and shape changes of one rotor.

Potential and force derivatives use the same angular convention. Reference
SAM forces and mixed Hessians are exact anchors. The two fitted responses
modulate the potential amplitude and its positive rational denominator.
"""
from dataclasses import dataclass
import numpy as np


def periodic_force_features(angles,potential,potential_gradient,*,origin,
        origin_gradient,origin_curvature,denominator_b):
    q=np.asarray(angles,float);v=np.asarray(potential,float);g=np.asarray(potential_gradient,float)
    if q.shape!=v.shape or q.shape!=g.shape or not np.all(np.isfinite([origin,origin_gradient,origin_curvature,denominator_b])):
        raise ValueError('compatible periodic potential and angular gradients required')
    if abs(origin_curvature)<1.e-10:raise ValueError('reference phase response requires nonzero angular curvature')
    d=1+denominator_b*(1-np.cos(q))
    if np.any(d<=0) or not all(np.all(np.isfinite(x)) for x in (q,v,g)):
        raise ValueError('finite periodic field with a positive denominator required')
    amplitude=v-origin-origin_gradient*np.sin(q)
    shape=-(v-origin)*(1-np.cos(q))/d
    phase=(g-origin_gradient*np.cos(q))/origin_curvature
    return np.column_stack((amplitude,shape)),phase


@dataclass(frozen=True)
class PeriodicSAMForce:
    reference_force: np.ndarray
    reference_mixed_hessian: np.ndarray
    amplitude_shape_response: np.ndarray

    def evaluate(self,features,phase_response):
        A=np.asarray(features,float);p=np.asarray(phase_response,float)
        if A.ndim!=2 or A.shape[1]!=2 or p.shape!=(len(A),):raise ValueError('periodic force features have incompatible shapes')
        return self.reference_force+p[:,None]*self.reference_mixed_hessian+A@self.amplitude_shape_response


def fit_periodic_sam_force(features,phase_response,forces,*,reference_mixed_hessian,training_indices):
    A=np.asarray(features,float);p=np.asarray(phase_response,float);f=np.asarray(forces,float)
    mixed=np.asarray(reference_mixed_hessian,float);indices=np.asarray(training_indices,int)
    if (A.ndim!=2 or A.shape[1]!=2 or p.shape!=(len(A),) or f.ndim!=2 or f.shape[0]!=len(A)
            or mixed.shape!=(f.shape[1],) or indices.ndim!=1 or len(set(indices.tolist()))!=len(indices)
            or np.any(indices<0) or np.any(indices>=len(A)) or 0 not in indices
            or not all(np.all(np.isfinite(x)) for x in (A,p,f,mixed))):
        raise ValueError('compatible finite periodic force data and training indices required')
    residual=f-f[0]-p[:,None]*mixed
    scale=np.linalg.norm(A[indices],axis=0)
    if np.any(scale==0):raise ValueError('unidentified amplitude/shape response')
    response,_,rank,_=np.linalg.lstsq(A[indices]/scale,residual[indices],rcond=1.e-12)
    if rank!=2:raise ValueError('periodic SAM amplitude/shape response is not identified')
    return PeriodicSAMForce(f[0].copy(),mixed.copy(),response/scale[:,None])
