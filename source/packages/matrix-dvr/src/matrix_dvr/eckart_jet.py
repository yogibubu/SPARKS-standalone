"""Smooth local Eckart rotations for differentiation at nonlinear references.

The rotation is obtained from the isolated maximal quaternion eigenpair using
four Newton steps in its scalar-positive coordinate patch. No singular-vector
derivatives enter. In particular a planar reference is admissible: the three
rotation-response eigenvalues are sums of principal mass moments, so the zero
singular value of a planar Cartesian covariance is not divided by.

This owner supplies local derivatives. Global finite rotations should use the
value alignment owner. The caller must independently check finite-displacement
values/derivatives against that alignment before accepting a coordinate jet.
"""

PROTOCOL_ID='dvr-eckart-coordinate-jet.v1'


def audit_coordinate_jet(value_at, first, second, *, step=1e-4,
                         first_tolerance=1e-5, second_tolerance=1e-4):
    """Independently check a supplied jet against a value-only alignment.

    `value_at` must use an independent finite-geometry owner, not the same
    differentiated rotation. No electronic evaluation is involved.
    """
    import numpy as np
    j=np.asarray(first,float);k=np.asarray(second,float)
    if j.ndim!=2 or k.shape!=j.shape+(j.shape[1],) or not 0<step:
        raise ValueError('Compatible first/second coordinate jets and positive step required')
    n=j.shape[1];eye=np.eye(n);origin=np.asarray(value_at(np.zeros(n)))
    finite_j=np.column_stack([(value_at(step*v)-value_at(-step*v))/(2*step) for v in eye])
    finite_k=np.empty_like(k)
    for i in range(n):
        for l in range(i+1):
            if i==l:
                value=(value_at(step*eye[i])+value_at(-step*eye[i])-2*origin)/step**2
            else:
                value=(value_at(step*(eye[i]+eye[l]))-value_at(step*(eye[i]-eye[l]))
                       -value_at(step*(-eye[i]+eye[l]))+value_at(-step*(eye[i]+eye[l])))/(4*step**2)
            finite_k[:,i,l]=finite_k[:,l,i]=value
    first_error=float(abs(j-finite_j).max());second_error=float(abs(k-finite_k).max())
    return dict(protocol_id=PROTOCOL_ID,passed=first_error<first_tolerance and second_error<second_tolerance,
        independent_owner='Independent value-only proper alignment and finite Cartesian reconstruction',
        first_max_error=first_error,second_max_error=second_error,step=step,new_QM=0)


def local_eckart_rotation_torch(moving_centered, reference_centered, masses):
    """Return the proper row rotation aligning two nearby centered geometries.

    Arguments are torch tensors, with shapes (atoms, 3), (atoms, 3), (atoms,).
    At the differentiation origin moving and reference must coincide and the
    molecule must be nonlinear. Four Newton steps reproduce the implicit
    rotation jet through order fifteen in exact arithmetic; the intended
    scientific use here requires derivatives only through order four.
    """
    import torch
    s=(moving_centered*masses[:,None]).T@reference_centered
    trace=torch.trace(s)
    b=torch.stack((s[2,1]-s[1,2],s[0,2]-s[2,0],s[1,0]-s[0,1]))
    eye=torch.eye(3,dtype=s.dtype,device=s.device)
    a=s+s.T-trace*eye
    v=torch.zeros_like(b)
    for _ in range(4):
        eigenvalue=trace+torch.dot(b,v)
        residual=b+a@v-eigenvalue*v
        jacobian=a-eigenvalue*eye-torch.outer(v,b)
        v=v-torch.linalg.solve(jacobian,residual)
    zero=v.new_zeros(())
    cross=torch.stack((torch.stack((zero,-v[2],v[1])),
                       torch.stack((v[2],zero,-v[0])),
                       torch.stack((-v[1],v[0],zero))))
    return eye+2*(cross+cross@cross)/(1+torch.dot(v,v))
