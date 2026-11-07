"""Streamed libcint contraction versus independent full AO integrals."""
import numpy as np
import pytest
from matrix_sparks.continuum.coulomb_native import DirectCoulombContraction


@pytest.mark.parametrize('cart', [False, True])
@pytest.mark.parametrize('count', [0, 1, 311, 312, 313, 625])
def test_direct_coulomb_matches_all_pairs(cart, count):
    pyscf=pytest.importorskip('pyscf')
    mol=pyscf.gto.M(atom='O 0 0 0; H 0 0 1.8; H 0 1.7 -.4',unit='Bohr',basis='cc-pvdz',cart=cart,verbose=0)
    rng=np.random.default_rng(125)
    matrix=rng.normal(size=(mol.nao_nr(),mol.nao_nr()))
    density=matrix@matrix.T
    points=rng.normal(size=(count,3))
    direct=DirectCoulombContraction(mol)
    actual=direct(points,density)
    expected=np.einsum('pij,ji->p',mol.intor('int1e_grids',grids=points,hermi=0),density)
    np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=3e-11)


def test_direct_coulomb_small_workspace_and_exact_zero_density():
    pyscf=pytest.importorskip('pyscf')
    mol=pyscf.gto.M(atom='O 0 0 0',spin=2,basis='cc-pvdz',verbose=0)
    shell=int(np.max(np.diff(mol.ao_loc_nr())))
    direct=DirectCoulombContraction(mol,workspace_bytes=8*shell**2)
    assert direct.tile==1
    points=np.random.default_rng(10).normal(size=(7,3))
    np.testing.assert_array_equal(direct(points,np.zeros((mol.nao,mol.nao))),np.zeros(7))
    density=np.eye(mol.nao)
    expected=np.einsum('pij,ji->p',mol.intor('int1e_grids',grids=points,hermi=0),density)
    np.testing.assert_allclose(direct(points,density),expected,rtol=3e-13,atol=3e-12)
    for p,d in [(points[:,:2],density),(points,density[:-1]),(points*float('nan'),density),
                (points,density.astype(complex))]:
        with pytest.raises(ValueError):direct(p,d)


def test_auto_contraction_falls_back_only_when_compiler_is_unavailable(monkeypatch):
    from types import SimpleNamespace
    from matrix_sparks.continuum import coulomb_native
    from matrix_sparks.continuum.molecular import AOHartreePotential
    from matrix_apoc import population
    pyscf=pytest.importorskip('pyscf')
    mol=pyscf.gto.M(atom='He 0 0 0',basis='sto-3g',verbose=0)
    state=SimpleNamespace(mo_coefficients=(None,),density_ao=lambda channel:np.eye(mol.nao),
                          overlap_ao=np.eye(mol.nao))
    monkeypatch.setattr(population,'pyscf_molecule_from_electronic_state',lambda state:mol)
    def unavailable(*args,**kwargs):raise FileNotFoundError('compiler unavailable')
    monkeypatch.setattr(coulomb_native,'DirectCoulombContraction',unavailable)
    assert AOHartreePotential(state).contraction_backend=='matrix'
    assert AOHartreePotential(state,contraction='matrix').contraction_backend=='matrix'
    with pytest.raises(FileNotFoundError):AOHartreePotential(state,contraction='direct')
