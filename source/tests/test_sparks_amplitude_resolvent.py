"""Exact source-transpose amplitudes share the full outgoing problem."""
import numpy as np
import pytest
from matrix_sparks.continuum import BoundaryWaves, SparseResolventBackend, SpectralRMatrixBackend
from matrix_sparks.continuum.photoionization import LocalPhotoionizationCalculation


@pytest.mark.parametrize('permutation', ['MMD_AT_PLUS_A', 'COLAMD'])
@pytest.mark.parametrize('coupled', [False, True])
@pytest.mark.parametrize('nsources', [0, 1, 3, 6, 8])
def test_source_transpose_matches_full_and_spectral(coupled, nsources, permutation):
    rng = np.random.default_rng(744)
    n, channels = 42, 6
    v = rng.normal(size=(n, n))
    h = v.T @ v / n + np.eye(n)
    s = np.diag(rng.uniform(.8, 1.2, n))
    b = rng.normal(size=(n, channels))
    d = rng.normal(size=(n, nsources))
    arrays = [rng.normal(size=channels) for _ in range(4)]
    if coupled:
        arrays = [np.diag(a) + .15*rng.normal(size=(channels, channels)) for a in arrays]
    waves = BoundaryWaves(*arrays)
    sparse = SparseResolventBackend(h, s, b, d, permutation=permutation)
    reference = SpectralRMatrixBackend(h, s, b, d)
    for energy in [.3, reference.eigenvalues[0]]:
        expected = sparse.evaluate(energy, waves)
        actual = sparse.evaluate_amplitudes(energy, waves)
        np.testing.assert_allclose(actual, expected.amplitudes, rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(actual, reference.evaluate(energy, waves).amplitudes,
                                   rtol=2e-10, atol=2e-11)
        assert sparse.last_diagnostics['solve_right_hand_sides'] == (
            nsources if 0 < nsources < channels else channels)
        assert sparse.last_diagnostics['backward_error'] < 1e-14


def test_local_amplitude_scan_uses_sources_without_scattering():
    class Solver:
        def evaluate_amplitudes(self, energy, waves):
            assert energy == waves
            return np.full((3, 5), energy + 1j)
        def evaluate(self, *args):
            raise AssertionError('Scattering calculation was requested')
    calculation = LocalPhotoionizationCalculation.__new__(LocalPhotoionizationCalculation)
    calculation.solver = Solver()
    calculation.boundary_waves = lambda energy: energy
    actual = calculation.amplitudes([.2, .4])
    assert actual.shape == (2, 3, 5)
    np.testing.assert_allclose(actual[:, 0, 0], [.2+1j, .4+1j])


def test_local_scan_accepts_backend_with_only_full_evaluate():
    from types import SimpleNamespace
    class Solver:
        def evaluate(self, energy, waves):
            assert energy == waves
            return SimpleNamespace(amplitudes=np.full((2, 4), energy+1j))
    calculation = LocalPhotoionizationCalculation.__new__(LocalPhotoionizationCalculation)
    calculation.solver = Solver()
    calculation.boundary_waves = lambda energy: energy
    actual = calculation.amplitudes([.2, .4])
    assert actual.shape == (2, 2, 4)
    np.testing.assert_allclose(actual[:, 0, 0], [.2+1j, .4+1j])


def test_sparse_permutation_rejects_unknown_ordering():
    with pytest.raises(ValueError,match="permutation"):
        SparseResolventBackend(np.eye(2),np.eye(2),np.ones((2,1)),np.ones((2,1)),permutation="unknown")


@pytest.mark.parametrize('coupled',[False,True])
def test_explicit_source_expansion_preserves_all_complex_amplitudes(coupled):
    rng=np.random.default_rng(808);n=21;channels=5
    v=rng.normal(size=(n,n));h=v.T@v+np.eye(n)
    b=rng.normal(size=(n,channels));d=rng.normal(size=(n,3))
    transform=np.array([[1.,0.,0.,2.,0.,0.],[0.,1.,0.,0.,-.3,0.],[0.,0.,1.,0.,0.,1.7]])
    arrays=[rng.normal(size=channels) for _ in range(4)]
    if coupled:arrays=[np.diag(a)+.1*rng.normal(size=(channels,channels)) for a in arrays]
    waves=BoundaryWaves(*arrays)
    compressed=SparseResolventBackend(h,np.eye(n),b,d,source_transform=transform)
    explicit=SparseResolventBackend(h,np.eye(n),b,d@transform)
    for energy in [.2,.7]:
        expected=explicit.evaluate(energy,waves).amplitudes
        np.testing.assert_allclose(compressed.evaluate_amplitudes(energy,waves),expected,rtol=1e-11,atol=1e-12)
        assert compressed.last_diagnostics['solve_right_hand_sides']==3
        assert compressed.last_diagnostics['output_source_columns']==6
        np.testing.assert_allclose(compressed.evaluate(energy,waves).amplitudes,expected,rtol=1e-11,atol=1e-12)
    with pytest.raises(ValueError,match='rows'):
        SparseResolventBackend(h,np.eye(n),b,d,source_transform=np.ones((2,4)))
