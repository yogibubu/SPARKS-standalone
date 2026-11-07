"""Declarative construction of a reaction-channel coordinate contract."""
from pathlib import Path
from matrix_core import read_sectioned_lines,section_content
from matrix_chem import preprocess_to_enriched_xyz,write_validation_section
from matrix_fragments import write_fragment_build_section
from .definition import write_sonic_build_sections_from_cartesian


def bond_change_analysis_definition(definition, *, atom_symbols, symmetry_operations=()):
    """Build an opt-in analysis chart without closed reactive support chords.

    ORACLE's canonical TS contract is kept as the input record. A distance
    explicitly marked as intercomponent *support* is omitted from analysis
    selection only when two other reactive distances already connect its
    endpoints. The bending direction is then completed by the existing
    rank-aware SMITH selector and symmetry projector. This avoids using a
    near-linear triangle's third distance as an artificial stretch.

    No atom numbers, geometry threshold or Hessian eigenvalues select the
    policy. Defaults and canonical definitions of other consumers are not
    changed. Rank loss is an error, never a Cartesian completion or sign fix.
    """
    from dataclasses import replace
    import numpy as np
    from .models import FrozenGIC
    from .symmetrization import _select_ranked_primitives_with_diagnostics
    from .definition import symmetrize_gic_definition

    support = {p.identifier for p in definition.primitives
               if p.function == 'R' and len(p.atoms) == 2
               and 'ORACLE_TS_SUPPORT_INTERCOMPONENT_CLOSEST' in p.refs}
    edges = [set(p.atoms) for p in definition.primitives
             if p.function == 'R' and p.family == 'TS_REACTION_DISTANCE'
             and len(p.atoms) == 2 and p.identifier not in support]
    excluded = set()
    for primitive in definition.primitives:
        if primitive.identifier not in support:
            continue
        endpoints = set(primitive.atoms)
        if any(len(a & b) == 1 and a ^ b == endpoints
               for a in edges for b in edges if a != b):
            excluded.add(primitive.identifier)
    if not excluded:
        return definition
    candidates = tuple(p for p in definition.primitives if p.identifier not in excluded)
    selected, rank, diagnostics = _select_ranked_primitives_with_diagnostics(
        candidates, np.array(definition.reference_coordinates_angstrom),
        target_rank=definition.target_rank, rank_tolerance=1e-9, condition_ordinary=True)
    if rank != definition.target_rank:
        # A reactive atlas may contain only the chord, with no independent
        # bend of the two changing bonds. Complete that graph-derived angle
        # using the resident referenced linear-bend pair. Its fourth atom is
        # chosen geometrically, never by molecular identity or frequencies.
        from matrix_chem import linear_bend_reference_atom
        from .models import GICPrimitive
        coordinates = np.array(definition.reference_coordinates_angstrom)
        completions = []
        for first in edges:
            for second in edges:
                common = first & second
                if len(common) != 1 or first == second:
                    continue
                ends = first ^ second
                if not any(set(p.atoms) == ends and p.identifier in excluded
                           for p in definition.primitives):
                    continue
                left, right = sorted(ends)
                chain = (left, next(iter(common)), right)
                if any(p.atoms == chain for p in completions):
                    continue
                ref = linear_bend_reference_atom(tuple(i-1 for i in chain), coordinates)
                if ref is None:
                    continue
                for mode in (-1, -2):
                    identifier = 'REACTION_LB_'+'_'.join(map(str, chain))+'_'+str(-mode)
                    completions.append(GICPrimitive(identifier, identifier, 'LINEAR_BEND',
                        'L', chain, mode=mode, ref_atoms=(ref+1,),
                        refs=('REACTION_PATH_ANALYSIS_GRAPH_COMPLETION',)))
        completed_atom_sets = {frozenset((*p.atoms, *p.ref_atoms)) for p in completions}
        # The bend pair owns both angular directions. A torsion on exactly
        # the same four atoms would consume one slot before the atomic pair
        # and prevent the native selector from admitting it as a unit.
        candidates = tuple(p for p in candidates if not(
            p.function == 'D' and frozenset(p.atoms) in completed_atom_sets))
        candidates = (*candidates, *completions)
        selected, rank, diagnostics = _select_ranked_primitives_with_diagnostics(
            candidates, coordinates, target_rank=definition.target_rank,
            rank_tolerance=1e-9, condition_ordinary=True)
        if rank != definition.target_rank:
            raise ValueError('bond-change analysis chart lost the molecular tangent rank')
    gics = tuple(FrozenGIC(f'GIC{i+1:03d}', p.name, p.family, 'A', p.identifier,
                          p.gaussian_expression(), ((p.identifier, 1.),))
                 for i, p in enumerate(selected))
    analysis = replace(definition, primitives=candidates, gics=gics, rank=rank,
        symmetrize=False, symmetry_diagnostics=None, reduction_diagnostics=diagnostics,
        semantic_diagnostics=(*definition.semantic_diagnostics,
            'REACTION_PATH_ANALYSIS BOND_CHANGES_FIRST',
            'REACTION_PATH_ANALYSIS EXCLUDED_SUPPORT_CHORDS='+','.join(sorted(excluded))))
    return symmetrize_gic_definition(analysis, atom_symbols=tuple(atom_symbols),
                                    symmetry_operations=tuple(symmetry_operations))


def write_reaction_path_build_sections_from_cartesian(
    source: Path, target: Path, *, symmetry_group: str,
    source_kind: str = 'auto', sycart: bool = False,
):
    """Build at a chosen channel reference using its common symmetry cap.

    ``symmetry_group`` is an a priori maximum: use the lowest common symmetry
    retained along the channel, not an endpoint's possibly higher group.
    ORACLE creates missing chemical/TS contracts and SMITH builds the basis.
    Callers retain this definition for transport; they must not rebuild with
    higher endpoint symmetry. Cs selects a pointwise molecular mirror when
    available rather than an atom-exchanging endpoint-only mirror.
    """
    group=str(symmetry_group).strip()
    if not group:raise ValueError('an a priori reaction-channel symmetry_group is required')
    source=Path(source);target=Path(target)
    lines=read_sectioned_lines(source)
    required=('VALIDATION','TOPOLOGY','SYNTHONS','SYMMETRY')
    if not all(section_content(lines,key) for key in required):
        preprocess_to_enriched_xyz(source,target,source_kind=source_kind)
    elif source.resolve()!=target.resolve():
        target.write_text(source.read_text(encoding='utf-8'),encoding='utf-8')
    lines=read_sectioned_lines(target)
    if not section_content(lines,'FRAGMENTS'):write_fragment_build_section(target)
    write_validation_section(target)
    if not section_content(read_sectioned_lines(target),'ORACLE_SONIC_CONTRACT'):
        from matrix_oracle import write_oracle_sonic_contract_from_xyzin
        write_oracle_sonic_contract_from_xyzin(target)
    # The high-level workflow asks the owner to construct a TS atlas. The
    # lower-level SMITH builder retains its strict no-inference TS gate.
    if not section_content(read_sectioned_lines(target),'ORACLE_TRANSITION_STATE_GEOMETRY'):
        from matrix_oracle.transition_state_geometry import write_oracle_transition_state_geometry_contract_from_xyzin
        write_oracle_transition_state_geometry_contract_from_xyzin(target)
    return write_sonic_build_sections_from_cartesian(target,symmetrize=True,
        symmetry_group=group,sycart=sycart,fragment_context='transition_state')
