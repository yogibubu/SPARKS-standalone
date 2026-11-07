"""Integrate moving nuclear kinks without changing the variational space."""

import numpy as np
import pytest
from matrix_sparks.continuum import RadialBasis, assemble_radial, assemble_sparse_radial


def test_fixed_trial_space_with_extra_integration_cuts():
    points = np.array([0.0, 1.0, 2.0])
    old = RadialBasis(points, degree=3, quadrature=8)
    split = RadialBasis(points, degree=3, quadrature=8, quadrature_breakpoints=[0.37, 1.3])
    assert old.size == split.size
    np.testing.assert_array_equal(
        old.evaluate(np.linspace(0, 2, 31)), split.evaluate(np.linspace(0, 2, 31))
    )

    def potential(r):
        return (r**2)[:, None, None]

    a = assemble_radial(points, [0], potential, degree=3, quadrature=8)
    b = assemble_sparse_radial(points, [0], potential, degree=3, quadrature=8, radial_basis=split)
    np.testing.assert_allclose(a.hamiltonian, b.hamiltonian.toarray(), atol=2e-14)
    np.testing.assert_allclose(a.overlap, b.overlap.toarray(), atol=2e-14)
    with pytest.raises(ValueError):
        RadialBasis(points, quadrature_breakpoints=[2.1])
    with pytest.raises(ValueError):
        assemble_sparse_radial(points, [0], potential, degree=4, quadrature=8, radial_basis=split)
