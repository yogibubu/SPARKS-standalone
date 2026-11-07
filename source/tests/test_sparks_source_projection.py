"""Alternative orbital sources reuse the same radial/atomic projection owners."""
from types import SimpleNamespace
import numpy as np
import pytest
from matrix_sparks.continuum import (RadialBasis,RealSphericalChannels,ProjectedDipoleSource,
    LocalPhotoionizationCalculation,project_radial_sources)


def test_standalone_source_projection_matches_explicit_quadrature():
    radial=RadialBasis(np.linspace(0,4,7),degree=3,quadrature=6)
    channels=RealSphericalChannels([(0,0),(1,-1),(1,0),(1,1)])
    orbital=lambda p:np.exp(-np.sum(p*p,axis=1))*(1+.1*p[:,0])
    source=ProjectedDipoleSource(channels,orbital)
    values=source(radial.nodes)
    expected=np.einsum('ri,rcs,r->cis',radial.values,values,radial.weights,optimize=False).reshape(4*radial.size,3)
    calculation=LocalPhotoionizationCalculation.__new__(LocalPhotoionizationCalculation)
    calculation.radial=radial;calculation.channels=channels
    calculation.operators=SimpleNamespace(basis=radial)
    np.testing.assert_allclose(calculation.dipole_sources(orbital),expected,rtol=1e-13,atol=1e-14)
    assert project_radial_sources(radial,4,None).shape==(4*radial.size,0)
    assert project_radial_sources(radial,4,lambda r:np.zeros((len(r),4,0))).shape==(4*radial.size,0)
    with pytest.raises(ValueError):project_radial_sources(radial,4,lambda r:np.ones((len(r),3)))


def test_alternative_orbital_source_includes_compact_atomic_blocks():
    from matrix_sparks.continuum.multicentre import (
        SphericalSplineFunctions,AtomicSplineBlock,MulticentreBasis,MulticentreOperators)
    radial=RadialBasis(np.linspace(0,4,7),degree=3,quadrature=6)
    channels=RealSphericalChannels([(0,0),(1,-1),(1,0),(1,1)])
    central=SphericalSplineFunctions(radial,channels)
    local_channels=RealSphericalChannels([(0,0)])
    atom=AtomicSplineBlock([0,0,2.],np.linspace(0,.5,5),local_channels)
    basis=MulticentreBasis(central,(atom,),np.eye(atom.size))
    n=basis.size
    calculation=LocalPhotoionizationCalculation.__new__(LocalPhotoionizationCalculation)
    calculation.radial=radial;calculation.channels=channels
    calculation.operators=MulticentreOperators(np.eye(n),np.eye(n),np.zeros((n,4)),
                                             np.empty((n,0)),basis,4.)
    sources=calculation.dipole_sources(lambda p:np.ones(len(p)))
    expected=2*np.sqrt(4*np.pi)*(atom.radial.weights*atom.radial.nodes)@atom.radial.values
    np.testing.assert_allclose(sources[central.size:,2],expected[atom.indices],atol=1e-14,rtol=1e-13)
    np.testing.assert_allclose(sources[central.size:,:2],0,atol=1e-14)
    batch=calculation.dipole_sources(lambda p:np.column_stack((np.ones(len(p)),-2*np.ones(len(p)))))
    np.testing.assert_allclose(batch,np.column_stack((sources,-2*sources)),rtol=1e-13,atol=1e-14)


def test_batched_source_projection_matches_individual_orbitals():
    radial=RadialBasis(np.linspace(0,4,7),degree=3,quadrature=6)
    channels=RealSphericalChannels([(0,0),(1,-1),(1,0),(1,1)])
    orbitals=[lambda p:np.exp(-np.sum(p*p,axis=1)),
              lambda p:p[:,0]*np.exp(-np.sum(p*p,axis=1)),
              lambda p:-2*np.exp(-np.sum(p*p,axis=1))]
    batch=lambda p:np.column_stack([o(p) for o in orbitals])
    actual=ProjectedDipoleSource(channels,batch)(radial.nodes)
    expected=np.concatenate([ProjectedDipoleSource(channels,o)(radial.nodes) for o in orbitals],axis=2)
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-14)
    assert ProjectedDipoleSource(channels,batch)([]).shape==(0,4,9)


def test_batched_ao_evaluation_reuses_each_grid_batch(monkeypatch):
    from matrix_sparks.continuum import AODysonOrbital
    from matrix_apoc import population
    calls=[]
    class Molecule:
        cart=False
        def nao_nr(self):return 2
        def eval_gto(self,name,points):
            calls.append(len(points));return np.column_stack((np.ones(len(points)),points[:,0]))
    monkeypatch.setattr(population,'pyscf_molecule_from_electronic_state',lambda s:Molecule())
    state=SimpleNamespace(overlap_ao=np.eye(2))
    coefficients=np.array([[1.,2.,0.],[0.,1.,-1.]])
    batch=AODysonOrbital(state,coefficients,batch_size=2)
    points=np.arange(15.).reshape(5,3)
    np.testing.assert_allclose(batch(points),np.column_stack((np.ones(5),points[:,0]))@coefficients)
    np.testing.assert_allclose(batch.norm_squared,[1,5,1])
    assert calls==[2,2,1]


@pytest.mark.parametrize('coupling,positive',[(.2,True),(1.2,False)])
@pytest.mark.parametrize('sparse',[False,True])
def test_compact_schur_check_detects_underintegrated_overlap(coupling,positive,sparse):
    from scipy.sparse import csc_matrix
    from matrix_sparks.continuum.multicentre import compact_overlap_diagnostics
    overlap=np.block([[np.eye(2),coupling*np.eye(2)],[coupling*np.eye(2),np.eye(2)]])
    operators=SimpleNamespace(basis=SimpleNamespace(central=SimpleNamespace(size=2)),
                              overlap=csc_matrix(overlap) if sparse else overlap)
    result=compact_overlap_diagnostics(operators)
    assert result['positive']==positive
    np.testing.assert_allclose(result['minimum_normalized_schur_eigenvalue'],1-coupling**2)
    assert (np.linalg.eigvalsh(overlap)[0]>0)==positive
    if not positive:
        with pytest.raises(ValueError,match='not positive'):
            compact_overlap_diagnostics(operators,require_positive=True)


def test_atomic_potential_samples_are_reused_without_changing_operators():
    from matrix_sparks.continuum import assemble_sparse_radial
    from matrix_sparks.continuum.multicentre import AtomicSplineBlock,assemble_multicentre
    ch=RealSphericalChannels([(0,0),(1,-1),(1,0),(1,1)])
    edges=np.linspace(0,4,9)
    central=assemble_sparse_radial(edges,ch.angular_momenta,lambda r:np.zeros((len(r),4,4)),degree=3,quadrature=6)
    atom=AtomicSplineBlock([0,0,2.],np.linspace(0,.5,5),RealSphericalChannels([(0,0)]))
    counts={}
    outputs=[]
    for reuse in [False,True]:
        counts[reuse]=0
        def potential(p):
            counts[reuse]+=len(p)
            return .17+.01*np.sum(p*p,axis=1)
        outputs.append(assemble_multicentre(central,ch,[atom],[0.],[[0,0,2.]],electronic_potential=potential,reuse_electronic_samples=reuse))
    np.testing.assert_allclose(outputs[0].hamiltonian.toarray(),outputs[1].hamiltonian.toarray(),rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(outputs[0].overlap.toarray(),outputs[1].overlap.toarray(),rtol=0,atol=0)
    assert counts[False]==2*counts[True]


def test_overlap_only_assembly_skips_potentials_and_cartesian_gradients(monkeypatch):
    from matrix_sparks.continuum import assemble_sparse_radial
    from matrix_sparks.continuum.multicentre import AtomicSplineBlock, assemble_multicentre, SphericalSplineFunctions
    ch = RealSphericalChannels([(0, 0), (1, -1), (1, 0), (1, 1)])
    edges = np.linspace(0, 4, 9)
    central = assemble_sparse_radial(edges, ch.angular_momenta,
        lambda r: np.zeros((len(r), 4, 4)), degree=3, quadrature=6)
    atom = AtomicSplineBlock([0, 0, 2.], np.linspace(0, .5, 5), RealSphericalChannels([(0, 0)]))
    full = assemble_multicentre(central, ch, [atom], [1.], [[0, 0, 2.]],
                                electronic_potential=lambda p: .1*np.ones(len(p)))
    original = SphericalSplineFunctions.evaluate
    def no_gradients(self, points, *, gradients=False):
        assert not gradients
        return original(self, points, gradients=gradients)
    monkeypatch.setattr(SphericalSplineFunctions, 'evaluate', no_gradients)
    def forbidden(p):
        raise AssertionError('Electronic field evaluated during overlap check')
    check = assemble_multicentre(central, ch, [atom], [1.], [[0, 0, 2.]],
                                 electronic_potential=forbidden, overlap_only=True)
    np.testing.assert_allclose(check.overlap.toarray(), full.overlap.toarray(), rtol=2e-14, atol=1e-15)


def test_invalid_overlap_is_rejected_before_electronic_preparation(monkeypatch):
    from matrix_sparks.continuum import photoionization, multicentre
    monkeypatch.setattr(photoionization.AODysonOrbital, 'from_apoc', lambda d: None)
    def forbidden(*args, **kw):
        raise AssertionError('Electronic preparation preceded the overlap check')
    monkeypatch.setattr(photoionization, 'AOHartreePotential', forbidden)
    def indefinite(*args, **kw):
        assert kw['overlap_only']
        return SimpleNamespace(overlap=np.array([[1., 2.], [2., 1.]]),
                               basis=SimpleNamespace(central=SimpleNamespace(size=1)))
    monkeypatch.setattr(multicentre, 'assemble_multicentre', indefinite)
    density = SimpleNamespace(atomic_numbers=np.array([1]),
                               coordinates_bohr=np.array([[0., 0., 2.]]), charge=0)
    ch = RealSphericalChannels([(0, 0)])
    with pytest.raises(ValueError, match='not positive'):
        LocalPhotoionizationCalculation(density, SimpleNamespace(state=density), ch,
            np.linspace(0, 4, 9), forbidden, asymptotic_charge=1, outer_radius=4,
            degree=3, quadrature=6, atomic_blocks=(object(),))


def test_positive_overlap_preflight_preserves_complete_electronic_assembly(monkeypatch):
    from matrix_sparks.continuum import photoionization, multicentre
    monkeypatch.setattr(photoionization.AODysonOrbital, 'from_apoc',
                        lambda d: lambda p: np.exp(-np.sum(p*p, axis=1)))
    original = multicentre.compact_overlap_diagnostics
    checked = []
    def check(*args, **kw):
        result = original(*args, **kw)
        checked.append(result)
        return result
    monkeypatch.setattr(multicentre, 'compact_overlap_diagnostics', check)
    def hartree(state):
        assert len(checked) == 1 and checked[0]['positive']
        return lambda p: np.zeros(len(p))
    monkeypatch.setattr(photoionization, 'AOHartreePotential', hartree)
    density = SimpleNamespace(atomic_numbers=np.array([1]),
                               coordinates_bohr=np.array([[0., 0., 2.]]), charge=0)
    ch = RealSphericalChannels([(0, 0)])
    atom = multicentre.AtomicSplineBlock([0, 0, 2.], np.linspace(0, .5, 5), ch)
    calc = LocalPhotoionizationCalculation(density, SimpleNamespace(state=density), ch,
        np.linspace(0, 4, 9), lambda p: np.zeros(len(p)), asymptotic_charge=1,
        outer_radius=4, degree=3, quadrature=6, atomic_blocks=(atom,))
    assert len(checked) == 2
    assert calc.overlap_diagnostics['positive']
    assert calc.overlap_preflight_seconds > 0
    np.testing.assert_allclose(checked[0]['minimum_normalized_schur_eigenvalue'],
                               checked[1]['minimum_normalized_schur_eigenvalue'], atol=1e-14)
