"""The common real Taylor fitter must preserve complex continuum derivatives."""

import numpy as np
import pytest
from matrix_sparks.continuum import symmetric_amplitude_derivative


def test_complex_quartic_stencil():
    q = np.array([-0.02, -0.01, 0, 0.01, 0.02])
    c = np.array([1 + 0.2j, 0.3 - 0.7j])
    a = np.array([0.1 + 0.4j, -0.2 + 0.1j])
    b = np.array([0.3 + 0.2j, 0.1 - 0.1j])
    values = (
        c
        + q[:, None] * a
        + q[:, None] ** 2 * b
        + q[:, None] ** 3 * (0.4 + 0.1j)
        + q[:, None] ** 4 * (0.2 - 0.1j)
    )
    result = symmetric_amplitude_derivative(q, values)
    np.testing.assert_allclose(result.first, a, atol=2e-13)
    np.testing.assert_allclose(result.second, 2 * b, atol=2e-10)
    np.testing.assert_allclose(
        result.first, (4 * result.fine_first - result.coarse_first) / 3, atol=2e-13
    )
    with pytest.raises(ValueError):
        symmetric_amplitude_derivative(q[:-1], values[:-1])


def test_complex_three_point_basis_check():
    q = np.array([-0.02, 0, 0.02])
    a = np.array([0.3 + 0.7j, -0.2j])
    b = np.array([0.2j, 0.4])
    result = symmetric_amplitude_derivative(q, 1 + q[:, None]*a + q[:, None]**2*b)
    np.testing.assert_allclose(result.first, a, atol=1e-13)
    np.testing.assert_allclose(result.second, 2*b, atol=1e-10)
    assert result.fine_first is None
    assert result.relative_step_difference is None
