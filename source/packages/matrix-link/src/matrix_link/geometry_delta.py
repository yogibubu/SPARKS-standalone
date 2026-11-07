"""Transfer a ground-state internal geometry correction to a spectator region."""
from dataclasses import dataclass
from collections.abc import Callable, Sequence
import numpy as np


@dataclass(frozen=True)
class InternalGeometryDeltaResult:
    coordinates_angstrom: np.ndarray
    internal_delta: np.ndarray
    moving_atoms: tuple[int, ...]
    iterations: int
    residual_max: float
    fixed_atoms_max_displacement_angstrom: float
    frozen_distances_max_error_angstrom: float = 0.0


def transfer_internal_geometry_delta(high_ground, low_ground, excited, *,
        moving_atoms: Sequence[int], coordinate_values: Callable,
        coordinate_jacobian: Callable, coordinate_difference: Callable | None = None,
        frozen_distances: Sequence[tuple[int, int]] = (),
        tolerance=1e-9, maximum_iterations=100) -> InternalGeometryDeltaResult:
    """Transfer projected q_H(S0)-q_L(S0), leaving all other atoms fixed.

    Coordinate values must use one frozen internal chart; Jacobian columns are
    derivatives with respect to Cartesian angstrom. A periodic-aware difference
    callback is required if that chart contains wrapped coordinates. A fixed
    left-singular basis isolates the movable-region internal coordinate space.
    Frozen distances are excluded from the transferable tangent space and
    constrained to their original target lengths during backtransformation.
    This is a geometry correction, not an optimization on the corrected PES;
    it neither corrects an energy nor creates a stationary Hessian.
    """
    high, low, initial = [np.asarray(a, float) for a in (high_ground, low_ground, excited)]
    if (initial.ndim != 2 or initial.shape[1] != 3 or high.shape != initial.shape
            or low.shape != initial.shape or not all(np.all(np.isfinite(a)) for a in (high,low,initial))):
        raise ValueError('three finite geometries with identical atom ordering required')
    moving = tuple(int(i) for i in moving_atoms)
    if not moving or len(set(moving)) != len(moving) or min(moving)<0 or max(moving)>=len(initial):
        raise ValueError('distinct in-range moving atom indices required')
    if not np.isfinite(tolerance) or tolerance<=0 or maximum_iterations<1:
        raise ValueError('positive backtransform controls required')
    columns = np.array([3*i+j for i in moving for j in range(3)])
    pairs = tuple(tuple(sorted((int(i), int(j)))) for i,j in frozen_distances)
    if len(set(pairs)) != len(pairs) or any(i<0 or j>=len(initial) or i==j for i,j in pairs):
        raise ValueError('frozen distances require distinct valid atom pairs')
    pairs = tuple((i,j) for i,j in pairs if i in moving or j in moving)
    def distances(x):
        return np.array([np.linalg.norm(x[i]-x[j]) for i,j in pairs])
    def distance_jacobian(x):
        rows = []
        for i,j in pairs:
            d=x[i]-x[j];length=np.linalg.norm(d)
            if length<1e-10:raise ValueError('frozen bond has zero length')
            row=np.zeros_like(x);row[i]=d/length;row[j]=-d/length
            rows.append(row.ravel()[columns])
        return np.asarray(rows).reshape(len(pairs),len(columns))
    tangent = np.eye(len(columns))
    if pairs:
        constraint = distance_jacobian(high)
        _,sv,right = np.linalg.svd(constraint,full_matrices=True)
        if len(sv)!=len(pairs) or sv[-1]<sv[0]*1e-10:
            raise ValueError('frozen-distance constraints must be independent')
        tangent=right[len(pairs):].T
    b = np.asarray(coordinate_jacobian(high), float)[:,columns] @ tangent
    u,s,_ = np.linalg.svd(b,full_matrices=False)
    if len(s)!=tangent.shape[1] or not len(s) or s[-1]<s[0]*1e-10:
        raise ValueError('internal chart does not independently anchor the allowed movable directions')
    difference = (lambda a,b:np.asarray(a)-np.asarray(b)) if coordinate_difference is None else coordinate_difference
    projector = u.T
    delta = projector @ difference(coordinate_values(high), coordinate_values(low))
    reference = np.asarray(coordinate_values(initial),float);x = initial.copy()
    reference_lengths=distances(initial)
    def residual(x):
        internal=delta-projector @ difference(coordinate_values(x),reference)
        return np.r_[internal,reference_lengths-distances(x)]
    for iteration in range(maximum_iterations+1):
        r = residual(x)
        distance_error=float(np.max(abs(distances(x)-reference_lengths))) if pairs else 0.
        if np.max(abs(r)) < tolerance and distance_error < min(tolerance,1e-11):
            fixed=np.ones(len(initial),bool);fixed[list(moving)]=False
            error=float(np.max(abs(x[fixed]-initial[fixed]))) if fixed.any() else 0.
            return InternalGeometryDeltaResult(x,delta,moving,iteration,float(max(abs(r))),error,distance_error)
        if iteration==maximum_iterations:break
        jacobian = projector @ np.asarray(coordinate_jacobian(x),float)[:,columns]
        if pairs:jacobian=np.vstack([jacobian,distance_jacobian(x)])
        dx = np.linalg.solve(jacobian,r).reshape(-1,3)
        scale=min(1.,.05/max(float(np.max(np.linalg.norm(dx,axis=1))),1e-15))
        old=np.linalg.norm(r)
        for _ in range(24):
            trial=x.copy();trial[list(moving)]+=scale*dx
            if np.linalg.norm(residual(trial))<old:x=trial;break
            scale*=.5
        else:raise RuntimeError('internal geometry-delta backtransform failed its line search')
    raise RuntimeError('internal geometry-delta backtransform did not converge')
