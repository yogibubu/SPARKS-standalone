"""Electrostatic probes of an immutable saved SCF density; no SCF is run."""
from pathlib import Path
import hashlib
import numpy as np


def checkpoint_electrostatic_potential(checkpoint, coordinates_bohr, *, expected_sha256=None, batch_size=256):
    """Evaluate nuclear minus electronic potential using native PySCF integrals.

    Coordinates remain in the checkpoint frame. This is the saved SCF
    reference density, not a correlated CC or MC-PDFT density. Spin channels
    of a UHF checkpoint are summed for electrostatics.
    """
    from pyscf import scf, gto, df
    path = Path(checkpoint)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and expected_sha256 != fingerprint:
        raise ValueError('saved density fingerprint differs from the declared reference')
    points = np.asarray(coordinates_bohr, float)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points))
            or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError('finite probe coordinates and positive integer batch size required')
    mol, record = scf.chkfile.load_scf(str(path))
    mo, occ = np.asarray(record['mo_coeff']), np.asarray(record['mo_occ'])
    density = np.einsum('...pi,...i,...qi->...pq', mo, occ, mo.conj())
    if density.ndim == 3:
        density = density.sum(axis=0)
    density = np.real_if_close(density)
    if np.iscomplexobj(density) or not np.all(np.isfinite(density)):
        raise ValueError('finite real saved charge density required')
    distance = np.linalg.norm(points[:, None]-mol.atom_coords()[None], axis=-1)
    if np.any(distance <= 1.e-10):
        raise ValueError('electrostatic probes cannot coincide with nuclei')
    potential = (1/distance) @ mol.atom_charges()
    for start in range(0, len(points), batch_size):
        stop = min(start+batch_size, len(points))
        probe = gto.fakemol_for_charges(points[start:stop])
        integrals = df.incore.aux_e2(mol, probe)
        potential[start:stop] -= np.einsum('ijp,ij->p', integrals, density)
    return potential, dict(checkpoint=str(path.resolve()), sha256=fingerprint,
        density='saved SCF reference, spin-summed', new_scf_calculations=0,
        probe_count=len(points), units='hartree per unit positive charge; bohr coordinates')
