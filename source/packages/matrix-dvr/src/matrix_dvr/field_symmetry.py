"""Value-geometry qualification of a proposed clamped angular reflection."""
import hashlib
import json
import numpy as np


def audit_clamped_reflection(tree, reference_values, active_indices, points,
                            masses, plane_normal, *, atom_permutations,
                            tolerance_angstrom=1e-7):
    """Check closure of the actual frozen family, including atom labels.

    The proposed operation negates the absolute active dihedrals. Remaining
    coordinates stay frozen. Physical Cartesian reflection and allowed atom
    permutations are evaluated independently of the periodic fitting basis.
    Proper SVD alignment is used for values only, never differentiated.
    """
    ref=np.asarray(reference_values,float);active=np.asarray(active_indices,int)
    points=np.asarray(points,float);m=np.asarray(masses,float)
    normal=np.asarray(plane_normal,float);normal=normal/np.linalg.norm(normal)
    permutations=tuple(tuple(map(int,p)) for p in atom_permutations)
    if (points.ndim!=2 or not len(points) or points.shape[1]!=len(active) or np.any(m<=0)
            or not permutations or any(sorted(p)!=list(range(len(m))) for p in permutations)
            or any(not np.array_equal(m[list(p)],m) for p in permutations)
            or not np.isfinite(points).all() or not np.isfinite(tolerance_angstrom) or not tolerance_angstrom>0):
        raise ValueError('Finite source points, mass-preserving permutations and positive tolerance required')
    rows=[]
    for point in points:
        natural=ref.copy();natural[active]+=point;x=tree.geometry(natural)
        mirror=x-2*(x@normal)[:,None]*normal
        reflected=ref.copy();reflected[active]=-natural[active];target=tree.geometry(reflected)
        target-=np.average(target,axis=0,weights=m);best=None
        for permutation in permutations:
            y=mirror[list(permutation)].copy();y-=np.average(y,axis=0,weights=m)
            u,_,vh=np.linalg.svd((y*m[:,None]).T@target)
            rotation=u@np.diag([1.,1.,np.linalg.det(u@vh)])@vh
            delta=y@rotation-target;rms=float(np.sqrt(np.sum(m[:,None]*delta**2)/(3*m.sum())))
            if best is None or rms<best['mass_weighted_RMS_A']:
                best=dict(mass_weighted_RMS_A=rms,maximum_atom_error_A=float(np.linalg.norm(delta,axis=1).max()),
                          atom_permutation=list(permutation))
        rows.append(best)
    request=dict(reference_values=ref.tolist(),active_indices=active.tolist(),points=points.tolist(),
                 masses_amu=m.tolist(),plane_normal=normal.tolist(),atom_permutations=permutations,
                 tolerance_angstrom=tolerance_angstrom)
    return dict(schema='matrix.dvr.clamped_reflection_geometry_audit.v1',
                passed=all(row['maximum_atom_error_A']<=tolerance_angstrom for row in rows),
                request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
                maximum_atom_error_A=max(row['maximum_atom_error_A'] for row in rows),rows=rows,
                experimental_data_used=False,new_QM=0)
