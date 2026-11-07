"""Nonredundant curvilinear analysis charts from frozen ORACLE primitives.

This explicit analysis policy selects geometric internal primitives before
symmetry adaptation. It does not change coordinate defaults for other tools.
"""
from dataclasses import replace
import numpy as np
from .models import FrozenGIC
from .symmetrization import _select_ranked_primitives_with_diagnostics
from .definition import symmetrize_gic_definition
from .evaluation import build_gic_b_matrix


INTERNAL_ANALYSIS_FUNCTIONS = frozenset({'R', 'A', 'D', 'L'})


def build_internal_analysis_definition(definition, *, atom_symbols, symmetry_operations=()):
    """Select a full-rank ESALC-I chart without Cartesian or legacy U rows.

    Bond distances, bends, torsions and referenced linear bends are taken
    only from the supplied frozen ORACLE pool. An incomplete pool fails;
    no Cartesian completion, topology inference or finite differences occur.
    Existing SMITH rank selection and symmetry adaptation own the reduction.
    """
    xyz=np.asarray(definition.reference_coordinates_angstrom,dtype=float)
    if xyz.shape!=(len(atom_symbols),3) or not np.all(np.isfinite(xyz)):
        raise ValueError('finite reference geometry and matching atom symbols required')
    candidates=tuple(p for p in definition.primitives if p.function in INTERNAL_ANALYSIS_FUNCTIONS)
    if not candidates:raise ValueError('frozen pool has no supported geometric internal primitives')
    selected,rank,diagnostics=_select_ranked_primitives_with_diagnostics(
        candidates,xyz,target_rank=definition.target_rank,rank_tolerance=1.e-9,
        condition_ordinary=True)
    if rank!=definition.target_rank:
        raise ValueError(f'geometric internal pool rank {rank}/{definition.target_rank}; no Cartesian completion permitted')
    gics=tuple(FrozenGIC(f'GIC{i+1:03d}',p.name,p.family,'A',p.identifier,
        p.gaussian_expression(),((p.identifier,1.),)) for i,p in enumerate(selected))
    result=replace(definition,primitives=candidates,gics=gics,rank=rank,
        candidate_count=len(candidates),symmetrize=False,symmetry_diagnostics=None,
        reduction_diagnostics=diagnostics,
        semantic_diagnostics=(*definition.semantic_diagnostics,
            'ANALYSIS_COORDINATES=ESALC-I_GEOMETRIC_INTERNALS',
            'INTERNAL_ANALYSIS_PRIMITIVES=R,A,D,L',
            'INTERNAL_ANALYSIS_CARTESIAN_COMPLETION=FORBIDDEN'))
    result=symmetrize_gic_definition(result,atom_symbols=tuple(atom_symbols),
        symmetry_operations=tuple(symmetry_operations))
    B=np.asarray(build_gic_b_matrix(result,coordinates_angstrom=xyz).rows,dtype=float)
    if B.shape!=(definition.target_rank,xyz.size) or np.linalg.matrix_rank(B,tol=1.e-9)!=definition.target_rank:
        raise ValueError('symmetry-adapted internal chart lost its prescribed rank')
    return result


def build_internal_analysis_from_oracle(path, *, symmetry_group):
    """Build the explicit geometric-internal analysis policy from frozen data.

    The full measured pool is consumed, including regular torsions across
    linear axes. The local optimizer chart is not an input prerequisite.
    A common channel symmetry cap must be supplied by the caller.
    """
    from pathlib import Path
    from matrix_core import read_sectioned_lines
    from matrix_chem import expected_vibrational_mode_count
    from .models import GICDefinition
    from .oracle_contract_consumer import load_validated_oracle_candidate_pool
    from .definition import _consume_oracle_primitive_contract
    from .definition_io import _point_group, _symmetry_operations, _apply_symmetry_group_limit
    from .primitive_handoff import bind_oracle_measurements
    if not str(symmetry_group).strip():raise ValueError('an a priori channel symmetry cap is required')
    path=Path(path);pool=load_validated_oracle_candidate_pool(path)
    xyz=np.asarray(pool.coordinates_angstrom,dtype=float)
    lines=read_sectioned_lines(path)
    group,operations=_apply_symmetry_group_limit(_point_group(lines),_symmetry_operations(lines),str(symmetry_group))
    source,schema,digest=_consume_oracle_primitive_contract(path,xyz,bonds=pool.contract.primary_topology.bonds)
    base=GICDefinition(backend='matrix-smith-internal-analysis.v1',point_group=group,
        symmetrize=False,target_rank=expected_vibrational_mode_count(xyz),rank=0,
        candidate_count=len(pool.candidates),reference_coordinates_angstrom=pool.coordinates_angstrom,
        primitives=pool.candidates,gics=(),primitive_source=source,
        primitive_source_schema=schema,primitive_b_matrix_sha256=digest)
    result=build_internal_analysis_definition(base,atom_symbols=pool.atom_symbols,
        symmetry_operations=operations)
    return bind_oracle_measurements(result,pool)


__all__ = ['INTERNAL_ANALYSIS_FUNCTIONS', 'build_internal_analysis_definition',
           'build_internal_analysis_from_oracle']
