"""Explicit-state MC-PDFT endpoint data from the resident PySCF owner.

No active space, spin, functional or hybrid convention is inferred here.
Endpoint gradients are partial derivatives at a fixed mixing parameter.
The caller supplies its geometric parameter response before fitting forces.
"""
from __future__ import annotations

import numpy as np
import sys


def mcpdft_endpoint_point(atoms, coordinates_angstrom, *, basis, charge,
                         multiplicity, active_orbitals, active_electrons,
                         ontop_functional, processors=1, memory_mb=8000,
                         grids_level=4, checkpoint=None, symmetry_subgroup=None,
                         electronic_reference_checkpoint=None,
                         electronic_reference_sha256=None):
    from pyscf import gto, lib, mcpdft, scf
    from pyscf.grad.casscf import Gradients as CASSCFGradients

    xyz = np.asarray(coordinates_angstrom, float)
    if (xyz.shape != (len(atoms), 3) or not np.all(np.isfinite(xyz))
            or int(active_orbitals) < 1 or int(multiplicity) < 1
            or not str(ontop_functional).startswith('t')):
        raise ValueError('finite geometry and explicit translated functional/spin/active space required')
    lib.num_threads(int(processors))
    mol = gto.Mole()
    mol.stdout = sys.stdout
    mol.build(atom=list(zip(atoms, xyz.tolist())), unit='Angstrom',
              basis=basis, charge=int(charge), spin=int(multiplicity)-1,
              symmetry=True, symmetry_subgroup=symmetry_subgroup,
              verbose=4, max_memory=int(memory_mb))
    mf = scf.ROHF(mol) if mol.spin else scf.RHF(mol)
    if checkpoint is not None:
        from pathlib import Path
        if Path(checkpoint).exists():
            mf.init_guess = 'chkfile'
        mf.chkfile = str(checkpoint)
    state_reference = None
    if electronic_reference_checkpoint is not None:
        from .reference_state import project_checkpoint_reference
        density, counts, state_reference = project_checkpoint_reference(
            mol, electronic_reference_checkpoint, expected_sha256=electronic_reference_sha256)
        if counts:
            mf.irrep_nelec = counts
        mf.kernel(dm0=density)
    else:
        if electronic_reference_sha256 is not None:
            raise ValueError('an electronic reference fingerprint requires its checkpoint')
        mf.kernel()
    if not mf.converged:
        raise RuntimeError('MC-PDFT reference SCF did not converge')
    from .reference_stability import qualify_internal_reference
    mf, stability_record = qualify_internal_reference(mf)
    mc = mcpdft.CASSCF(mf, ontop_functional, int(active_orbitals),
                       active_electrons, grids_level=int(grids_level))
    if state_reference is not None and mol.groupname != 'C1':
        from pyscf.scf.hf_symm import get_wfnsym
        mc.fcisolver.wfnsym = get_wfnsym(mf)
    mc.kernel()
    if not mc.converged:
        raise RuntimeError('MC-PDFT reference CASSCF did not converge')
    from .lagrange_response import qualified_pdft_gradient
    response = qualified_pdft_gradient(mc)
    g0 = np.asarray(response.kernel(), float)
    if not response.converged:
        raise RuntimeError('MC-PDFT Lagrange response did not converge; gradient is not qualified')
    g1 = np.asarray(CASSCFGradients(mc).kernel(), float)
    e0, e1 = float(mc.e_tot), float(mc.e_mcscf)
    if not np.all(np.isfinite([e0, e1])) or not np.all(np.isfinite([g0, g1])):
        raise FloatingPointError('nonfinite electronic endpoint data')
    return dict(energy_pdft_hartree=e0, energy_casscf_hartree=e1,
                gradient_pdft_hartree_per_bohr=g0.tolist(),
                gradient_casscf_hartree_per_bohr=g1.tolist(),
                active_orbitals=int(active_orbitals), active_electrons=active_electrons,
                multiplicity=int(multiplicity), ontop_functional=ontop_functional,
                detected_symmetry=mol.topgroup, calculation_symmetry=mol.groupname,
                hybrid_convention='average: E(a)=(1-a)*E_PDFT+a*E_CASSCF',
                gradient_convention='fixed-parameter endpoint partial derivatives',
                reference_orbital_occupations=np.asarray(mf.mo_occ).tolist(),
                active_natural_occupations=np.linalg.eigvalsh(
                    mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas))[::-1].tolist(),
                scf_converged=bool(mf.converged), casscf_converged=bool(mc.converged),
                gradient_response_converged=bool(response.converged),
                gradient_response=response.response_recovery,
                electronic_reference=state_reference,
                reference_internal_stability=stability_record,
                pyscf_version=__import__('pyscf').__version__)
