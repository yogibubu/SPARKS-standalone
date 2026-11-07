import numpy as np
import pytest

from matrix_sparks import molecular_frame_photoelectron_distribution as distribution
from matrix_sparks.continuum import RealSphericalChannels


def test_atomic_linear_and_circular_patterns():
    channels = RealSphericalChannels([(1, 1), (1, -1), (1, 0)])
    xyz = np.eye(3)
    common = dict(electron_energy_hartree=.2, residual_charge=1, directions=xyz)
    linear = distribution(np.eye(3), channels,
                          polarization_density=np.diag([0, 0, 1]), **common)
    np.testing.assert_allclose(linear['angular_strength_per_steradian'],
                               [0, 0, 3 / (4 * np.pi)], atol=2e-15)
    eps = np.array([1, 1j, 0]) / np.sqrt(2)
    circular = distribution(np.eye(3), channels,
                            polarization_density=np.outer(eps, eps.conj()), **common)
    np.testing.assert_allclose(circular['angular_strength_per_steradian'],
                               [3 / (8 * np.pi), 3 / (8 * np.pi), 0], atol=2e-15)


def test_complex_interference_and_quadrature_sum_rule():
    channels = RealSphericalChannels([(l, m) for l in range(4) for m in range(-l, l + 1)])
    rng = np.random.default_rng(982)
    d = rng.normal(size=(2, 3, 16)) + 1j * rng.normal(size=(2, 3, 16))
    eps = np.array([1, .3j, -.2])
    eps /= np.linalg.norm(eps)
    rho = np.outer(eps, eps.conj())
    result = distribution(d, channels, electron_energy_hartree=.4,
                          residual_charge=1, polarization_density=rho)
    integrated = result['angular_strength_per_steradian'] @ channels.weights
    expected = np.sum(abs(np.einsum('a,...ac->...c', eps, d)) ** 2, axis=-1)
    np.testing.assert_allclose(integrated, expected, rtol=3e-14)
    np.testing.assert_allclose(result['integrated_dipole_strength'], expected, rtol=3e-14)
    mixed = distribution(d, channels, electron_energy_hartree=.4,
                         residual_charge=1, polarization_density=np.eye(3) / 3)
    np.testing.assert_allclose(mixed['integrated_dipole_strength'],
                               np.sum(abs(d) ** 2, axis=(-1, -2)) / 3, rtol=3e-14)


@pytest.mark.parametrize('rho', [np.eye(3), np.diag([1.1, -.1, 0]),
                               [[1, 1j, 0], [0, 0, 0], [0, 0, 0]],
                               np.full((3, 3), np.nan)])
def test_reject_nonphysical_polarization(rho):
    with pytest.raises(ValueError):
        distribution(np.eye(3), RealSphericalChannels([(1, 1), (1, -1), (1, 0)]),
                     electron_energy_hartree=.2, residual_charge=1,
                     polarization_density=rho)
