"""Exact angular projections and Hermitian Coulomb grid integrals."""
import numpy as np
import pytest
from matrix_sparks.continuum import RealSphericalChannels
from matrix_sparks.continuum.molecular import ProjectedLocalPotential, AOHartreePotential


@pytest.mark.parametrize('batch', [1, 7, 16])
def test_projected_potential_matches_independent_weighted_gram(batch):
    channels = RealSphericalChannels([(l,m) for l in range(4) for m in range(-l,l+1)])
    radii = np.linspace(.1, 3., 19)
    origin = np.array([.2,-.1,.3])
    def potential(points):
        return np.sin(points[:,0])+points[:,1]*points[:,2]+np.exp(-np.sum(points**2,axis=1))
    projector = ProjectedLocalPotential(channels,potential,origin_bohr=origin,radial_batch=batch)
    expected=[]
    for radius in radii:
        values=potential(origin+radius*channels.directions)
        block=np.einsum('pa,p,pb,p->ab',channels.values,values,channels.values,channels.weights,optimize=False)
        expected.append((block+block.T)*.5)
    np.testing.assert_allclose(projector(radii),expected,rtol=2e-13,atol=2e-14)
    assert projector(np.array([])).shape==(0,16,16)


@pytest.mark.parametrize('cart', [False,True])
def test_hermitian_coulomb_grids_match_all_ao_pairs(cart):
    pyscf=pytest.importorskip('pyscf')
    mol=pyscf.gto.M(atom='O 0 0 0; H 0 0 1.8; H 0 1.7 -.4',unit='Bohr',basis='cc-pvdz',cart=cart,verbose=0)
    rng=np.random.default_rng(45)
    matrix=rng.normal(size=(mol.nao_nr(),mol.nao_nr()))
    density=matrix@matrix.T
    points=rng.uniform(-2.,3.,size=(13,3))
    provider=AOHartreePotential.__new__(AOHartreePotential)
    provider.molecule=mol;provider.density=density;provider.batch_size=5
    full=mol.intor('int1e_grids',grids=points,hermi=0)
    expected=np.einsum('pij,ji->p',full,density)
    np.testing.assert_allclose(provider(points),expected,rtol=2e-13,atol=2e-12)


@pytest.mark.parametrize('batch', [1, 5, 16])
def test_joint_electronic_projection_matches_separate_fields(batch):
    channels = RealSphericalChannels([(l,m) for l in range(4) for m in range(-l,l+1)])
    radii = np.linspace(.1, 4., 21)
    def coulomb(points):
        return 3. / np.sqrt(1. + np.sum(points**2, axis=1))
    def xc(points):
        return -.2 * np.exp(-np.sum(points**2, axis=1)) * (1. + points[:,0])
    def projected(field):
        return ProjectedLocalPotential(channels, field, radial_batch=batch)(radii)
    np.testing.assert_allclose(projected(lambda p: coulomb(p)+xc(p)),
                               projected(coulomb)+projected(xc), atol=4e-15, rtol=3e-14)
