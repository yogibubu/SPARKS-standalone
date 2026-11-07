"""Energy conservation, coherent interpolation and absolute normalization."""

import numpy as np
import pytest
from scipy.constants import alpha
from matrix_sparks import ContinuumDipoleTable, fixed_photon_photoelectron_lines
from matrix_sparks.photoelectron import BOHR_SQUARED_TO_MB


@pytest.mark.parametrize('budget', [1, 7, 400, 100000])
def test_blocked_gaussian_profiles_retain_all_signed_lines(budget, monkeypatch):
    from matrix_sparks.photoelectron import gaussian_line_profiles
    rng = np.random.default_rng(61)
    x = np.linspace(-4, 6, 83)
    lines = rng.normal(size=37)
    strengths = rng.normal(size=(2, 4, len(lines)))
    kernel = np.exp(-.5*((x[:, None]-lines[None, :])/.3)**2)/(np.sqrt(2*np.pi)*.3)
    expected = strengths @ kernel.T
    sizes = []
    original = np.exp
    def measured(values, **kw):
        sizes.append(values.size)
        return original(values, **kw)
    monkeypatch.setattr(np, 'exp', measured)
    actual = gaussian_line_profiles(x, lines, strengths, sigma=.3,
                                    maximum_kernel_entries=budget)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    assert max(sizes) <= budget


def test_gaussian_profile_empty_lines_and_validation():
    from matrix_sparks.photoelectron import gaussian_line_profiles
    assert gaussian_line_profiles([1, 2], [], np.empty((4, 0)), sigma=1).shape == (4, 2)
    assert gaussian_line_profiles([], [1], [2], sigma=1).shape == (0,)
    for budget in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            gaussian_line_profiles([0], [1], [2], sigma=1, maximum_kernel_entries=budget)
    with pytest.raises(ValueError):
        gaussian_line_profiles([0], [1], [2j], sigma=1)


def test_complex_table_interpolates_before_squaring():
    c = np.zeros((2, 3, 1, 1), complex)
    c[:, 0, 0, 0] = [1+1j, -1-1j]
    table = ContinuumDipoleTable([.1, .3], c)
    np.testing.assert_allclose(table([.1, .3]), c)
    np.testing.assert_allclose(table([.2]), 0, atol=1e-15)
    for e in ([0], [.09], [.31], [np.nan]):
        with pytest.raises(ValueError):
            table(e)


def test_each_transition_uses_its_own_electron_energy():
    queried = []
    def provider(e):
        queried.extend(e)
        c = np.zeros((len(e), 3, 1, 2), complex)
        c[:, 0, 0, 0] = 1+1j*e
        c[:, 1, 0, 1] = e
        return c
    result = fixed_photon_photoelectron_lines([.6, .8], [.4, .7],
        [[.9, .2], [.3, -.4]], provider, populations=[1, .5])
    assert result['open_channels'].tolist() == [[True, False], [True, True]]
    np.testing.assert_allclose(queried, [.1, .2, .4])
    np.testing.assert_allclose(result['strengths'][0, 3], [.9**2*1.04+.2**2*.04, 0])
    np.testing.assert_allclose(result['strengths'][:, :3].sum(axis=1), result['strengths'][:, 3])
    np.testing.assert_allclose(result['cross_sections_Mb'][:, 3],
        result['strengths'][:, 3]*(4*np.pi**2*alpha/3*np.array([.6, .8])*BOHR_SQUARED_TO_MB)[:, None])


def test_closed_and_exact_threshold_channels_do_not_query_provider():
    def reject(e):
        raise AssertionError('Provider called for a closed channel')
    result = fixed_photon_photoelectron_lines([.2, .4], [.4, .5], [[1], [1]], reject)
    assert not result['cross_sections_Mb'].any()


def test_isotropic_rotation_and_channel_unitary_invariance():
    rng = np.random.default_rng(71)
    c = rng.normal(size=(3, 4, 3))+1j*rng.normal(size=(3, 4, 3))
    moments = rng.normal(size=(5, 3))+1j*rng.normal(size=(5, 3))
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
    transformed = np.einsum('ij,jca,cd->ida', rot, c, unitary)
    def calculate(coeff):
        return fixed_photon_photoelectron_lines([.8], np.linspace(.4, .5, 5), moments,
            lambda e: np.broadcast_to(coeff, (len(e), *coeff.shape)))['cross_sections_Mb']
    np.testing.assert_allclose(calculate(c), calculate(transformed), rtol=2e-14, atol=1e-13)
    np.testing.assert_allclose(calculate(.3*c), .09*calculate(c), rtol=2e-14, atol=1e-13)


def test_constant_operator_is_fc_only_and_no_extra_spin_factor():
    c = np.ones((3, 1, 1), complex)
    result = fixed_photon_photoelectron_lines([1.], [.5], [[1]],
        lambda e: np.broadcast_to(c, (len(e), *c.shape)))
    np.testing.assert_allclose(result['strengths'][0, :, 0], [3, 0, 0, 3])
    np.testing.assert_allclose(result['cross_sections_Mb'][0, 3, 0], 4*np.pi**2*alpha*BOHR_SQUARED_TO_MB)


@pytest.mark.parametrize('energies', [[.2,.1], [0,.1], [.1,.1], [np.nan,.2]])
def test_bad_energy_tables_rejected(energies):
    with pytest.raises(ValueError):
        ContinuumDipoleTable(energies, np.ones((2,3,1,1)))


def test_provider_shape_and_nonfinite_values_rejected():
    for bad in (np.ones((1,3,1,2)), np.full((1,3,1,1), np.nan)):
        with pytest.raises(ValueError):
            fixed_photon_photoelectron_lines([.8], [.4], [[1]], lambda e: bad)


def test_portable_archive_requires_operator_identity():
    from matrix_sparks.photoelectron_cli import _basis_identity
    assert _basis_identity({'operator_basis_id': np.array('neutral-normal-frame-v1')}) == 'neutral-normal-frame-v1'
    for bad in ({}, {'operator_basis_id':np.array('')}, {'operator_basis_id':np.array(['ambiguous'])}):
        with pytest.raises(ValueError): _basis_identity(bad)
