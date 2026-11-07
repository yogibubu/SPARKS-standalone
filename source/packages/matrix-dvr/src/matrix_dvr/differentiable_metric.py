"""Differentiable full internal/rotation metric for a supplied geometry map."""


def internal_rotation_metric_torch(geometry, masses, point):
    """Same full mass blocks as full_metric, with a differentiable geometry.

    Returns the contravariant kinetic coefficient in cm-1 and log(sqrt(det
    covariant metric)). The caller must qualify chart regularity independently.
    Rotation follows the internal coordinates; no rotational block is omitted.
    """
    import torch
    from .full_metric import KINETIC_CM_AMU_ANGSTROM2
    m = torch.as_tensor(masses, dtype=point.dtype, device=point.device)
    x = geometry(point); d = torch.func.jacfwd(geometry)(point)
    x = x-torch.sum(m[:, None]*x, dim=0)/m.sum()
    d = d-torch.einsum('i,ias->as', m, d)[None]/m.sum()
    cross = torch.einsum('i,isa->as', m,
        torch.linalg.cross(x[:, None, :].expand(-1, point.numel(), -1), d.swapaxes(1, 2)))
    inertia = (torch.einsum('i,ia,ia->', m, x, x)*torch.eye(3, dtype=point.dtype, device=point.device)
               -torch.einsum('i,ia,ib->ab', m, x, x))
    vib = torch.einsum('i,ias,iat->st', m, d, d)
    cov = torch.cat((torch.cat((vib, cross.T), dim=1),
                     torch.cat((cross, inertia), dim=1)), dim=0)
    return KINETIC_CM_AMU_ANGSTROM2*torch.linalg.inv(cov), .5*torch.logdet(cov)


def flat_measure_scalar_cm(kinetic, kinetic_first, log_volume_first, log_volume_second):
    """Scalar W in .5*p_i G_ij p_j + W for the flat-measure operator.

    This is exactly the connection in CurvilinearSineDVR, expanded at a point;
    no additional volume or Podolsky term may be added to that same operator.
    """
    import numpy as np
    g, g1, l, l2 = map(lambda a: np.asarray(a, float),
                     (kinetic, kinetic_first, log_volume_first, log_volume_second))
    n = len(l)
    if (g.shape != (n,n) or g1.shape != (n,n,n) or l2.shape != (n,n)
            or not all(np.isfinite(a).all() for a in (g,g1,l,l2))):
        raise ValueError('compatible finite kinetic and volume jets required')
    return float(.25*(np.einsum('iji,j',g1,l)+np.einsum('ij,ij',g,l2))+.125*l@g@l)
