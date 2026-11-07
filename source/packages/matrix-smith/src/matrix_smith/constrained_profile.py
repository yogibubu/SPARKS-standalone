"""Energy derivatives of stationary profiles with moving geometric constraints."""
import numpy as np


def constrained_profile_derivative(constraint_values,constraint_jacobian,parameter,
        coordinates,gradient,masses,*,step=1.e-4):
    """Return -lambda.dC/dparameter for C(parameter, x)=0.

    The finite differences evaluate geometry only. They include changes of
    constraint coefficients as well as changes of their target values. Both
    derivative steps and the remaining unconstrained force are reported.
    Cartesian gradients and Jacobians must use the same length unit.
    """
    x=np.asarray(coordinates,float);g=np.asarray(gradient,float).reshape(-1)
    m=np.asarray(masses,float)
    if (x.ndim!=2 or x.shape[1]!=3 or m.shape!=(len(x),) or g.shape!=(x.size,)
            or np.any(m<=0) or not all(np.all(np.isfinite(v)) for v in (x,g,m))
            or not np.isfinite(parameter) or not np.isfinite(step) or step<=0):
        raise ValueError('finite matching Cartesian geometry, force, masses and positive step required')
    c=np.asarray(constraint_jacobian(parameter,x),float)
    value=np.asarray(constraint_values(parameter,x),float)
    if c.ndim!=2 or c.shape!=(len(value),x.size) or value.ndim!=1 or len(value)==0 or not np.all(np.isfinite(c)) or not np.all(np.isfinite(value)):
        raise ValueError('finite compatible constraint values and Jacobian required')
    weights=np.repeat(m,3);gram=(c/weights)@c.T
    if np.linalg.matrix_rank(gram)!=len(c):raise ValueError('independent constraints required')
    multipliers=np.linalg.solve(gram,(c/weights)@g)
    partials=[]
    for h in [step,step/2]:
        plus=np.asarray(constraint_values(parameter+h,x),float)
        minus=np.asarray(constraint_values(parameter-h,x),float)
        if plus.shape!=value.shape or minus.shape!=value.shape or not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
            raise ValueError('finite matching neighboring constraint values required')
        partials.append((plus-minus)/(2*h))
    result=float(-multipliers@partials[-1])
    return dict(energy_derivative=result,multipliers=multipliers,
        parameter_partial_constraint=partials[-1],
        derivative_step_difference=float(abs(multipliers@(partials[-1]-partials[0]))),
        maximum_constraint_residual=float(np.max(abs(value))),
        mass_weighted_tangent_gradient_norm=float(np.linalg.norm((g-c.T@multipliers)/np.sqrt(weights))))
