"""Compact Cartesian tangent frames for declared rigid fragments."""
import numpy as np
from .eckart_chart import eckart_rigid_constraint_rows


def rigid_fragment_tangent_frames(geometries_angstrom,fragment_atoms_zero_based):
    xyz=np.asarray(geometries_angstrom,float)
    groups=tuple(np.asarray(g,int)for g in fragment_atoms_zero_based)
    if (xyz.ndim!=3 or xyz.shape[-1]!=3 or not len(xyz) or not groups
            or not np.all(np.isfinite(xyz))
            or any(not len(g)for g in groups)
            or sorted(np.concatenate(groups).tolist())!=list(range(xyz.shape[1]))):
        raise ValueError('finite geometries and complete disjoint fragment atom sets required')
    result=[]
    for geometry in xyz:
        rows=[]
        for group in groups:
            rigid=eckart_rigid_constraint_rows(geometry[group],np.ones(len(group)))
            norms=np.linalg.norm(rigid,axis=1);rigid=rigid[norms>1e-12]/norms[norms>1e-12,None]
            _,singular,basis=np.linalg.svd(rigid,full_matrices=False)
            basis=basis[singular>singular[0]*1e-10]
            indices=(group[:,None]*3+np.arange(3)).ravel()
            for vector in basis:
                row=np.zeros(geometry.size);row[indices]=vector;rows.append(row)
        result.append(np.array(rows))
    if len({m.shape for m in result})!=1:
        raise ValueError('fragment rigid rank changes across the acquired data')
    return np.stack(result)
