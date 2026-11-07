"""Qualify an SCF reference within its declared spin and spatial symmetry.

An SCF stationary point need not be a minimum with respect to orbital
rotations. Follow only the resident backend's internal, symmetry-preserving
instability direction; never disable symmetry or change the active space.
"""
import numpy as np


def _internal_rotation_count(mf):
    from pyscf.scf.hf_symm import get_orbsym
    occupation = np.asarray(mf.mo_occ)
    symmetry = (np.zeros(len(occupation), int) if mf.mol.groupname == 'C1'
                else get_orbsym(mf.mol, mf.mo_coeff))
    return int(np.count_nonzero((occupation[:, None] > occupation[None, :])
                               & (symmetry[:, None] == symmetry[None, :])))


def qualify_internal_reference(mf, *, maximum_refinements=4):
    """Return the stable reference and a record of any SCF refinements.

    Calling this owner can restart SCF and therefore belongs inside the
    explicitly authorized electronic acquisition, before endpoint gradients.
    Exhausting the finite refinement budget blocks the electronic point.
    """
    from pyscf.scf import stability

    if (not isinstance(maximum_refinements, int) or maximum_refinements < 0
            or not mf.converged):
        raise ValueError('converged SCF and a finite nonnegative refinement budget required')
    if np.asarray(mf.mo_coeff).ndim != 2:
        raise ValueError('restricted closed- or open-shell reference required')
    check = stability.rohf_internal if mf.mol.spin else stability.rhf_internal
    history = []
    for attempt in range(maximum_refinements + 1):
        dimension = _internal_rotation_count(mf)
        if dimension:
            orbitals, stable = check(mf, with_symmetry=True, return_status=True,
                                     nroots=min(3, dimension), tol=1.e-7)
        else:
            orbitals, stable = mf.mo_coeff, True
        history.append(dict(refinement=attempt, internally_stable=bool(stable),
            allowed_orbital_rotations=dimension,
            energy_hartree=float(mf.e_tot),
            orbital_direction_norm=float(np.linalg.norm(orbitals-mf.mo_coeff))))
        if stable:
            return mf, dict(internally_stable=True, spatial_symmetry=mf.mol.groupname,
                spin=int(mf.mol.spin), symmetry_preserved=True,
                scf_refinements=attempt, history=history)
        if attempt == maximum_refinements:
            break
        previous_energy = float(mf.e_tot)
        density = mf.make_rdm1(orbitals, mf.mo_occ)
        mf.kernel(dm0=density)
        if not mf.converged:
            raise RuntimeError('SCF refinement of an internal instability did not converge')
        if not np.isfinite(mf.e_tot) or mf.e_tot > previous_energy + 1.e-7:
            raise RuntimeError('internal stability refinement increased the reference energy')
    raise RuntimeError('SCF reference remains internally unstable; endpoint data are blocked')
