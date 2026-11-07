"""Frame-covariant overlaps of AO transition densities; no electronic solve."""
from functools import lru_cache
from pathlib import Path

import numpy as np


def _source_key(source):
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Transition-density overlap needs its orbital source: {path}')
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=32)
def _molecule(key):
    from matrix_pyscf import load_molden_wavefunction
    return load_molden_wavefunction(key[0])[0]


def ao_rotation(mol, rotation):
    """Rotate AO coefficients with a proper row-coordinate rotation.

    All contracted shells are retained, including Cartesian or spherical
    higher-angular-momentum shells in PySCF's normalized AO convention.
    """
    from pyscf.symm.basis import _momentum_rotation_matrices
    rotation = np.asarray(rotation, dtype=float)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10) or np.linalg.det(rotation) < 0:
        raise ValueError('A proper orthogonal rotation is required')
    angular = _momentum_rotation_matrices(mol, rotation)
    transform = np.zeros((mol.nao_nr(), mol.nao_nr()))
    locations = mol.ao_loc_nr()
    for shell in range(mol.nbas):
        block = angular[mol.bas_angular(shell)]
        width = block.shape[0]
        for contraction in range(mol.bas_nctr(shell)):
            start = locations[shell] + contraction * width
            transform[start:start+width, start:start+width] = block
    return transform


@lru_cache(maxsize=32)
def _pair_metrics(reference_key, candidate_key):
    from pyscf import gto
    reference = _molecule(reference_key)
    candidate = _molecule(candidate_key)
    if reference.natm != candidate.natm or any(
        reference.atom_symbol(i) != candidate.atom_symbol(i) for i in range(reference.natm)
    ):
        raise ValueError('Transition-density comparison requires matched atom identities and order')
    left, right = reference.atom_coords(), candidate.atom_coords()
    lc, rc = left.mean(axis=0), right.mean(axis=0)
    a, b = left-lc, right-rc
    u, singular, vt = np.linalg.svd(b.T @ a)
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    # A linear/atomic geometry cannot determine rotation about its axis.
    # Fail explicitly instead of assigning a state from an arbitrary frame.
    if np.count_nonzero(singular > max(float(singular[0]), 1.) * 1e-12) < 2:
        raise ValueError('A linear or atomic reference needs an explicit electronic frame for AO-state comparison')
    aligned = candidate.set_geom_(b @ rotation + lc, unit='Bohr', inplace=False)
    return (
        reference.intor('int1e_ovlp'), aligned.intor('int1e_ovlp'),
        gto.intor_cross('int1e_ovlp', reference, aligned),
        ao_rotation(candidate, rotation),
    )


def transition_density_overlap(reference, candidate):
    """Normalized two-coordinate integral of aligned transition densities.

    Cross-basis overlap integrals supply the AO metric. Aligning both centres
    and angular functions makes the value independent of a rigid laboratory
    rotation; contractions with transition densities remove MO-phase choices.
    """
    saa, sbb, sab, rotation = _pair_metrics(_source_key(reference.source), _source_key(candidate.source))
    na, nb = saa.shape[0], sbb.shape[0]
    if reference.vector.size != na*na or candidate.vector.size != nb*nb:
        raise ValueError('Transition-density dimensions do not match their orbital sources')
    ta = reference.vector.reshape(na, na)
    tb = rotation @ candidate.vector.reshape(nb, nb) @ rotation.T
    norm_a = float(np.sum(ta * (saa @ ta @ saa)))
    norm_b = float(np.sum(tb * (sbb @ tb @ sbb)))
    if norm_a <= 0 or norm_b <= 0:
        raise ValueError('Transition density has a nonpositive AO-metric norm')
    value = float(np.sum(ta * (sab @ tb @ sab.T)) / np.sqrt(norm_a * norm_b))
    if abs(value) > 1 + 1e-7:
        raise ValueError('Transition-density overlap exceeds its normalized bound')
    return min(1., abs(value)), (1. if value >= 0 else -1.)
