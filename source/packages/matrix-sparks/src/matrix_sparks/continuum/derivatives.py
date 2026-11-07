"""Coordinate derivatives of phase-consistent complex continuum amplitudes."""

from dataclasses import dataclass

import numpy as np
from matrix_numerics.finite_difference import fit_taylor_derivatives


@dataclass(frozen=True)
class AmplitudeDerivative:
    first: np.ndarray
    second: np.ndarray
    coarse_first: np.ndarray
    fine_first: np.ndarray | None
    step_difference_norm: float | None
    relative_step_difference: float | None


def symmetric_amplitude_derivative(displacements, amplitudes):
    """Fit an explicit central three- or five-point amplitude stencil.

    All amplitudes must share a channel convention, coordinate frame and Dyson
    sign. Reuses MATRIX's array Taylor fitter on separate real/imaginary slots.
    Five points give a fourth-order first derivative; coarse/fine derivatives
    provide a step-sensitivity diagnostic, not a statistical error estimate.
    """
    q = np.asarray(displacements, float)
    values = np.asarray(amplitudes)
    if (
        q.shape not in ((3,), (5,))
        or values.ndim < 2
        or values.shape[0] != len(q)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Require three or five finite amplitude arrays and their displacements")
    order = np.argsort(q)
    q = q[order]
    values = values[order]
    step = q[-1]
    if (
        not np.isfinite(step)
        or step <= 0
        or not np.allclose(
            q,
            step * (np.array([-1, 0, 1]) if len(q) == 3 else np.array([-1, -0.5, 0, 0.5, 1])),
            rtol=1e-12,
            atol=0,
        )
    ):
        raise ValueError("Expected the explicit symmetric h and h/2 stencil")
    real_slots = np.stack((values.real, values.imag), axis=-1)

    def fit(indices, degree):
        result = fit_taylor_derivatives(q[indices], real_slots[indices], max_order=degree)
        return [a[..., 0] + 1j * a[..., 1] for a in result.derivatives]

    if len(q) == 3:
        first, second = fit(np.arange(3), 2)
        return AmplitudeDerivative(first, second, first, None, None, None)
    first, second, *_ = fit(np.arange(5), 4)
    (coarse,) = fit(np.array([0, 4]), 1)
    (fine,) = fit(np.array([1, 3]), 1)
    difference = float(np.linalg.norm(fine - coarse))
    norm = float(np.linalg.norm(first))
    return AmplitudeDerivative(
        first, second, coarse, fine, difference, None if norm == 0 else difference / norm
    )
