"""Compact atomic source integrals retain independent local contributions."""
from types import SimpleNamespace
import numpy as np
import pytest
from numpy.testing import assert_allclose
from matrix_sparks.continuum.angular import RealSphericalChannels
from matrix_sparks.continuum.multicentre import (
    AtomicSplineBlock, MulticentreBasis, MulticentreOperators,
    with_multicentre_spatial_sources,
)


def test_atomic_constant_source_matches_analytic_angular_integral():
    channels = RealSphericalChannels([(0,0)], quadrature='lebedev', lebedev_order=17)
    atom = AtomicSplineBlock([0,0,2.], np.linspace(0,.5,5), channels)
    transform = 2*np.eye(atom.size)
    basis = MulticentreBasis(SimpleNamespace(size=2), (atom,), transform)
    n = basis.size
    operators = MulticentreOperators(np.eye(n), np.eye(n), np.zeros((n,1)),
                                    np.empty((n,0)), basis, 4.)
    center = np.array([[.3],[.7]])
    result = with_multicentre_spatial_sources(
        operators, lambda points: np.ones((len(points),1)), central_sources=center)
    radial = atom.radial.evaluate(atom.radial.nodes)
    exact_raw = np.sqrt(4*np.pi)*(atom.radial.weights*atom.radial.nodes)@radial
    expected = 2*exact_raw[atom.indices]
    assert_allclose(result.sources[:2], center, atol=0)
    assert_allclose(result.sources[2:,0], expected, rtol=2e-14, atol=1e-15)
    with pytest.raises(ValueError, match='Spatial source'):
        with_multicentre_spatial_sources(operators, lambda points: np.ones(len(points)),
                                        central_sources=center)
