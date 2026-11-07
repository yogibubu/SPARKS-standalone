"""Electronic-sector checks on existing checkpoints, without electronic jobs."""
from pathlib import Path
import hashlib
import numpy as np


def checkpoint_electronic_sector(checkpoint):
    from pyscf import scf, symm
    from pyscf.scf import hf_symm, uhf_symm
    path = Path(checkpoint)
    mol, data = scf.chkfile.load_scf(str(path))
    orbitals = np.asarray(data['mo_coeff'], float)
    occupations = np.asarray(data['mo_occ'], float)
    if orbitals.ndim == 3:
        reference = scf.UHF(mol)
        owner = uhf_symm
    elif orbitals.ndim == 2:
        reference = scf.ROHF(mol) if mol.spin else scf.RHF(mol)
        owner = hf_symm
    else:
        raise ValueError('unsupported reference orbital representation')
    reference.mo_coeff, reference.mo_occ = orbitals, occupations
    name = (symm.irrep_id2name(mol.groupname, owner.get_wfnsym(reference))
            if mol.groupname != 'C1' else 'A')
    return dict(point_group=mol.groupname, determinant_irrep=name, charge=int(mol.charge),
                alpha_minus_beta_electrons=int(mol.spin),
                symmetry_resolves_sector=mol.groupname != 'C1',
                checkpoint=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require_common_electronic_sector(left, right):
    """Reject different determinant sectors before electronic calibration.

    A C1 label alone does not identify an electronic root. The returned scope
    makes that explicit; state-subspace tracking belongs to APOC.
    """
    keys = ('point_group', 'determinant_irrep', 'charge', 'alpha_minus_beta_electrons')
    if any(left[k] != right[k] for k in keys):
        raise ValueError('electronic calibration would mix different symmetry sectors: '
                         f"{left['point_group']} {left['determinant_irrep']} versus "
                         f"{right['point_group']} {right['determinant_irrep']}")
    return dict(compatible=True,
        resolved_by_symmetry=bool(left['symmetry_resolves_sector'] and right['symmetry_resolves_sector']),
        left=left, right=right)
