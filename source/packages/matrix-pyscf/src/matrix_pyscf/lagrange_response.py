"""Residual-qualified recovery of a native real MC-PDFT response system.

PySCF supplies the right-hand side, Hessian action and nuclear response.
MINRES retries a failed CG solve without changing the wavefunction, energy,
active space or response equations. An incompatible null-space component
remains a failure; no diagonal shift or force fitting changes that system.
"""
import numpy as np


def refine_lagrange_solution(gradient, initial):
    from scipy.sparse.linalg import LinearOperator, minres
    converged, vector, rhs, action, diagonal = initial
    if converged:
        gradient.response_recovery = dict(solver='native CG',converged=True,
            residual_norm=float(np.linalg.norm(rhs+action(vector))))
        return initial
    if np.iscomplexobj(rhs):
        raise ValueError('real symmetric response required for MINRES recovery')
    operator = LinearOperator((len(rhs),len(rhs)),matvec=action,dtype=float)
    vector, info = minres(operator,-rhs,rtol=min(float(gradient.conv_rtol),1e-9),
        maxiter=max(200,4*len(rhs)),check=True)
    residual = float(np.linalg.norm(rhs+action(vector)))
    threshold = max(float(gradient.conv_atol),float(gradient.conv_rtol)*float(np.linalg.norm(rhs)))
    converged = bool(info == 0 and np.all(np.isfinite(vector)) and residual <= threshold)
    gradient.response_recovery = dict(solver='MINRES on unchanged native Hessian action',
        converged=converged,info=int(info),residual_norm=residual,threshold=threshold,
        electronic_state_changed=False)
    return converged,vector,rhs,action,diagonal


def qualified_pdft_gradient(mc):
    from pyscf.grad.mcpdft import Gradients

    class QualifiedGradient(Gradients):
        def solve_lagrange(self, *args, **kwargs):
            return refine_lagrange_solution(self,super().solve_lagrange(*args,**kwargs))

    return QualifiedGradient(mc)
