"""Condition a local SAM basis while retaining its exact geometric constraints.

The primitive functions and protected fixed/torsional combinations are kept.
Only the complementary SAM combinations are rebuilt at the current geometry.
No force constants, frequencies, energies or electronic samples are changed.
"""
from dataclasses import replace
import numpy as np
from .models import FrozenGIC
from .evaluation import build_primitive_b_matrix,build_gic_b_matrix
from .eckart_chart import eckart_rigid_constraint_rows,eckart_projected_b_matrix


def recondition_local_torsion_section(section,coordinates_angstrom,*,rank_tolerance=1.e-10):
    x=np.asarray(coordinates_angstrom,float);m=np.asarray(section.masses,float)
    if x.shape!=section.reference.shape or not np.all(np.isfinite(x)) or not 0<rank_tolerance<1:
        raise ValueError('matching finite geometry and a positive rank tolerance required')
    definition=section.definition;n=definition.rank;primitives=definition.primitives
    protected_ids=list(section.fixed_indices)+list(section.torsion_indices)
    identifiers={p.identifier:i for i,p in enumerate(primitives)}
    protected=np.zeros((len(protected_ids),len(primitives)))
    for row,i in enumerate(protected_ids):
        gic=definition.gics[i]
        for name,c in (gic.coefficients or ((gic.primitive_id,1.),)):
            protected[row,identifiers[name]]=c
    rigid=eckart_rigid_constraint_rows(x,m)
    bp=np.asarray(build_primitive_b_matrix(definition,coordinates_angstrom=x).rows)
    weighted=eckart_projected_b_matrix(bp,m,rigid)/np.sqrt(np.repeat(m,3))[None,:]
    c=protected@weighted;gram=c@c.T
    if len(c) and np.linalg.matrix_rank(gram,tol=rank_tolerance*np.linalg.norm(gram,2))!=len(c):
        raise ValueError('protected geometric constraints lose rank')
    coefficients=np.eye(len(primitives))
    if len(c):coefficients-=np.linalg.solve(gram,c@weighted.T).T@protected
    u,singular,_=np.linalg.svd(coefficients@weighted,full_matrices=False)
    count=n-len(protected_ids)
    if count<1 or len(singular)<count or singular[count-1]<=rank_tolerance*singular[0]:
        raise ValueError('primitive pool cannot span the constrained SAM space')
    sam=(u[:,:count]/singular[:count]).T@coefficients
    gics=[definition.gics[i] for i in protected_ids]
    for i,row in enumerate(sam):
        gics.append(FrozenGIC(f'RECONDITIONED_SAM{i+1:04d}',f'SAM_{i+1}','LOCAL_SAM','',
            primitives[0].identifier,'NONE',tuple((p.identifier,float(v))
                for p,v in zip(primitives,row) if v!=0.)))
    updated=replace(definition,gics=tuple(gics),
        reference_coordinates_angstrom=tuple(map(tuple,x)),symmetrize=False,
        symmetry_diagnostics=None,reduction_diagnostics=None,
        semantic_diagnostics=(*definition.semantic_diagnostics,'SAM_RECONDITIONED_WITH_EXACT_PROTECTED_PHASES'))
    old=np.asarray(build_gic_b_matrix(definition,coordinates_angstrom=x).rows)
    new=np.asarray(build_gic_b_matrix(updated,coordinates_angstrom=x).rows)
    if np.linalg.matrix_rank(np.vstack((new,rigid)))!=x.size:
        raise ValueError('reconditioned section does not span Cartesian space')
    fixed=len(section.fixed_indices);torsions=len(section.torsion_indices)
    result=replace(section,definition=updated,reference=x.copy(),
        fixed_indices=tuple(range(fixed)),torsion_indices=tuple(range(fixed,fixed+torsions)),
        sam_indices=tuple(range(fixed+torsions,n)))
    return result,dict(schema='matrix.smith.local_torsion_reconditioning.v1',
        protected_functions_unchanged=True,primitive_functions_unchanged=True,
        original_augmented_condition=float(np.linalg.cond(np.vstack((old,rigid)))),
        reconditioned_augmented_condition=float(np.linalg.cond(np.vstack((new,rigid)))),
        complementary_rank=count)
