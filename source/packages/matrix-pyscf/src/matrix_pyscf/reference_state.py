"""Reuse a saved electronic reference without changing its symmetry sector.

This is backend projection and state bookkeeping, not chemical perception.
The calling acquisition provides the checkpoint and its content fingerprint.
No electronic optimization is performed by this module.
"""
from pathlib import Path
import hashlib
import numpy as np


def project_checkpoint_reference(mol, checkpoint, *, expected_sha256):
    from pyscf import scf
    from pyscf.scf import hf_symm, uhf_symm

    path = Path(checkpoint)
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_sha256:
        raise ValueError('electronic reference checkpoint fingerprint differs from the declared input')
    source, record = scf.chkfile.load_scf(str(path))
    if (source.natm != mol.natm or source.charge != mol.charge or source.spin != mol.spin
            or not np.array_equal(source.atom_charges(), mol.atom_charges())
            or not np.allclose(source.atom_coords(), mol.atom_coords(), atol=1.e-9, rtol=0)):
        raise ValueError('common-state reference requires identical atoms, geometry, charge and spin')
    if source.groupname != mol.groupname:
        raise ValueError('reference and target symmetry groups require an explicit state mapping')
    coefficients = np.asarray(record['mo_coeff'], float)
    occupations = np.asarray(record['mo_occ'], float)
    if coefficients.ndim == 3:
        density = np.einsum('spi,si,sqi->spq', coefficients, occupations, coefficients)
        counts = (uhf_symm.get_irrep_nelec(source, coefficients, occupations)
                  if source.groupname != 'C1' else {})
    elif coefficients.ndim == 2:
        if source.spin:
            density = np.array([(coefficients*np.minimum(occupations, 1.))@coefficients.T,
                (coefficients*np.maximum(occupations-1., 0.))@coefficients.T])
            counts = {}
            if source.groupname != 'C1':
                labels = hf_symm.get_orbsym(source, coefficients)
                counts = {name: (int(np.sum(np.minimum(occupations[labels == irrep], 1.))),
                                 int(np.sum(np.maximum(occupations[labels == irrep]-1., 0.))))
                          for name, irrep in zip(source.irrep_name, source.irrep_id)}
        else:
            density = (coefficients*occupations)@coefficients.T
            counts = (hf_symm.get_irrep_nelec(source, coefficients, occupations)
                      if source.groupname != 'C1' else {})
    else:
        raise ValueError('unsupported saved reference orbital representation')
    if mol.spin and any(alpha < beta for alpha, beta in counts.values()):
        raise ValueError('source irrep spin populations cannot be represented by this restricted reference')
    # Use the resident backend cross-basis projector, including the UHF spin
    # densities when the target is a restricted open-shell calculation.
    projected = scf.addons.project_dm_nr2nr(source, density, mol)
    return projected, counts, dict(checkpoint=str(path.resolve()), sha256=actual_hash,
        source_basis=str(source.basis), target_basis=str(mol.basis),
        symmetry_group=mol.groupname,
        irrep_electrons={key: list(map(int, value)) if isinstance(value, tuple) else int(value)
                        for key, value in counts.items()},
        convention='same geometry and spin; projected density and retained irrep electron counts')
