"""Adiabatic EVB evaluators with analytic first and second derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surface_representation import SurfaceRepresentationRequest, validate_basin_transition


@dataclass(frozen=True)
class EVBEvaluation:
    energy: float
    gradient: np.ndarray
    state_weights: np.ndarray
    gap: float
    hessian: np.ndarray | None = None


@dataclass(frozen=True)
class EVBCouplingJet:
    """Value, gradient and Hessian of one real diabatic coupling."""

    value: float
    gradient: np.ndarray
    hessian: np.ndarray


@dataclass(frozen=True)
class CompactJetEvaluation:
    """Value and first two derivatives of a compactly supported local jet."""

    value: float
    gradient: np.ndarray
    hessian: np.ndarray


@dataclass(frozen=True)
class PathConditionedScalarEvaluation:
    """Scalar value and derivatives in an invariant reaction-path chart."""

    value: float
    gradient: np.ndarray
    hessian: np.ndarray
    path_coordinate: float
    transverse_distance: float
    segment_index: int


def _pchip_slopes(abscissae: np.ndarray, ordinates: np.ndarray) -> np.ndarray:
    """Return Fritsch--Butland slopes for a one-dimensional PCHIP."""

    x = np.asarray(abscissae, dtype=float).reshape(-1)
    y = np.asarray(ordinates, dtype=float).reshape(-1)
    if x.size < 2 or y.shape != x.shape or np.any(np.diff(x) <= 0.0):
        raise ValueError("PCHIP knots require equal arrays and strictly increasing abscissae")
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)
    if x.size == 2:
        slopes[:] = delta[0]
        return slopes
    for index in range(1, x.size - 1):
        left = delta[index - 1]
        right = delta[index]
        if left == 0.0 or right == 0.0 or np.sign(left) != np.sign(right):
            slopes[index] = 0.0
        else:
            w1 = 2.0 * h[index] + h[index - 1]
            w2 = h[index] + 2.0 * h[index - 1]
            slopes[index] = (w1 + w2) / (w1 / left + w2 / right)

    def endpoint(h0: float, h1: float, d0: float, d1: float) -> float:
        value = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(value) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(value) > 3.0 * abs(d0):
            return 3.0 * d0
        return float(value)

    slopes[0] = endpoint(h[0], h[1], delta[0], delta[1])
    slopes[-1] = endpoint(h[-1], h[-2], delta[-1], delta[-2])
    return slopes


def _pchip_value_derivatives(
    abscissae: np.ndarray,
    ordinates: np.ndarray,
    coordinate: float,
) -> tuple[float, float, float]:
    """Evaluate a shape-preserving cubic and its first two derivatives."""

    x = np.asarray(abscissae, dtype=float).reshape(-1)
    y = np.asarray(ordinates, dtype=float).reshape(-1)
    slopes = _pchip_slopes(x, y)
    s = float(np.clip(coordinate, x[0], x[-1]))
    index = min(int(np.searchsorted(x, s, side="right") - 1), x.size - 2)
    index = max(index, 0)
    width = x[index + 1] - x[index]
    t = (s - x[index]) / width
    y0, y1 = y[index], y[index + 1]
    m0, m1 = slopes[index], slopes[index + 1]
    value = (
        (2.0 * t**3 - 3.0 * t**2 + 1.0) * y0
        + (t**3 - 2.0 * t**2 + t) * width * m0
        + (-2.0 * t**3 + 3.0 * t**2) * y1
        + (t**3 - t**2) * width * m1
    )
    first = (
        (6.0 * t**2 - 6.0 * t) * y0
        + (3.0 * t**2 - 4.0 * t + 1.0) * width * m0
        + (-6.0 * t**2 + 6.0 * t) * y1
        + (3.0 * t**2 - 2.0 * t) * width * m1
    ) / width
    second = (
        (12.0 * t - 6.0) * y0
        + (6.0 * t - 4.0) * width * m0
        + (-12.0 * t + 6.0) * y1
        + (6.0 * t - 2.0) * width * m1
    ) / (width * width)
    return float(value), float(first), float(second)


def evaluate_path_conditioned_scalar(
    coordinates: np.ndarray,
    path_knots: np.ndarray,
    knot_values: np.ndarray,
    *,
    inner_radius: float,
    outer_radius: float,
) -> PathConditionedScalarEvaluation:
    """Evaluate a PCHIP path field with a C2 compact transverse tube.

    All vectors are expressed in the same invariant chart, normally ESALC-I.
    The nearest line segment fixes the longitudinal coordinate.  A quintic
    taper leaves the field unchanged inside ``inner_radius`` and makes its
    value, gradient and Hessian exactly zero at ``outer_radius``.
    """

    q = np.asarray(coordinates, dtype=float).reshape(-1)
    knots = np.asarray(path_knots, dtype=float)
    values = np.asarray(knot_values, dtype=float).reshape(-1)
    if knots.ndim != 2 or knots.shape[0] < 2 or knots.shape[1] != q.size:
        raise ValueError("reaction-path knots have incompatible shape")
    if values.shape != (knots.shape[0],):
        raise ValueError("one scalar value is required for every path knot")
    if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(values)):
        raise ValueError("reaction-path data must be finite")
    inner = float(inner_radius)
    outer = float(outer_radius)
    if inner < 0.0 or outer <= inner:
        raise ValueError("path tube radii must satisfy 0 <= inner < outer")
    segment_vectors = np.diff(knots, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    if np.any(segment_lengths <= np.finfo(float).eps):
        raise ValueError("reaction-path knots must be distinct")
    arc = np.r_[0.0, np.cumsum(segment_lengths)]
    best: tuple[float, int, float, np.ndarray] | None = None
    for index, (origin, vector, length) in enumerate(
        zip(knots[:-1], segment_vectors, segment_lengths)
    ):
        fraction = float(np.clip((q - origin) @ vector / (length * length), 0.0, 1.0))
        transverse = q - (origin + fraction * vector)
        candidate = (float(np.linalg.norm(transverse)), index, fraction, transverse)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    radius, index, fraction, transverse = best
    tangent = segment_vectors[index] / segment_lengths[index]
    path_coordinate = float(arc[index] + fraction * segment_lengths[index])
    value, first, second = _pchip_value_derivatives(arc, values, path_coordinate)
    projector = np.eye(q.size) - np.outer(tangent, tangent)
    if radius <= inner:
        taper = 1.0
        taper_gradient = np.zeros_like(q)
        taper_hessian = np.zeros((q.size, q.size), dtype=float)
    elif radius >= outer:
        return PathConditionedScalarEvaluation(
            0.0, np.zeros_like(q), np.zeros((q.size, q.size)), path_coordinate, radius, index
        )
    else:
        width = outer - inner
        scaled = (radius - inner) / width
        taper = 1.0 - 10.0 * scaled**3 + 15.0 * scaled**4 - 6.0 * scaled**5
        radial_first = (-30.0 * scaled**2 + 60.0 * scaled**3 - 30.0 * scaled**4) / width
        radial_second = (-60.0 * scaled + 180.0 * scaled**2 - 120.0 * scaled**3) / (
            width * width
        )
        direction = transverse / radius
        taper_gradient = radial_first * direction
        taper_hessian = radial_second * np.outer(direction, direction) + (
            radial_first / radius
        ) * (projector - np.outer(direction, direction))
    longitudinal_gradient = first * tangent
    gradient = taper * longitudinal_gradient + value * taper_gradient
    hessian = (
        taper * second * np.outer(tangent, tangent)
        + np.outer(longitudinal_gradient, taper_gradient)
        + np.outer(taper_gradient, longitudinal_gradient)
        + value * taper_hessian
    )
    return PathConditionedScalarEvaluation(
        float(value * taper), gradient, 0.5 * (hessian + hessian.T), path_coordinate, radius, index
    )


def evaluate_compact_support_jet(
    displacement: np.ndarray,
    jet: EVBCouplingJet,
    *,
    inner_radius: float,
    outer_radius: float,
) -> CompactJetEvaluation:
    """Realize a local two-jet with a C2 compact radial support.

    The displacement is expressed in the caller's invariant coordinate chart
    (ESALC-I for reactive ZAFF/EVB surfaces). The taper is one in the inner
    ball and zero outside the outer ball, so the local jet is exact at the
    chart origin and cannot alter a remote basin.
    """

    q = np.asarray(displacement, dtype=float).reshape(-1)
    gradient = np.asarray(jet.gradient, dtype=float).reshape(-1)
    hessian = np.asarray(jet.hessian, dtype=float)
    if gradient.shape != q.shape or hessian.shape != (q.size, q.size):
        raise ValueError("compact-support jet and displacement have incompatible shapes")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(gradient)) or not np.all(
        np.isfinite(hessian)
    ):
        raise ValueError("compact-support jet requires finite inputs")
    inner = float(inner_radius)
    outer = float(outer_radius)
    if not np.isfinite(inner) or not np.isfinite(outer) or inner < 0.0 or outer <= inner:
        raise ValueError("compact-support radii must satisfy 0 <= inner < outer")

    polynomial_gradient = gradient + hessian @ q
    polynomial = float(jet.value + gradient @ q + 0.5 * q @ hessian @ q)
    radius = float(np.linalg.norm(q))
    if radius <= inner:
        taper = 1.0
        taper_gradient = np.zeros_like(q)
        taper_hessian = np.zeros_like(hessian)
    elif radius >= outer:
        return CompactJetEvaluation(0.0, np.zeros_like(q), np.zeros_like(hessian))
    else:
        width = outer - inner
        t = (radius - inner) / width
        taper = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
        radial_first = (-30.0 * t**2 + 60.0 * t**3 - 30.0 * t**4) / width
        radial_second = (-60.0 * t + 180.0 * t**2 - 120.0 * t**3) / (width * width)
        direction = q / radius
        projector = np.eye(q.size) - np.outer(direction, direction)
        taper_gradient = radial_first * direction
        taper_hessian = radial_second * np.outer(direction, direction) + (
            radial_first / radius
        ) * projector

    value = polynomial * taper
    result_gradient = taper * polynomial_gradient + polynomial * taper_gradient
    result_hessian = (
        taper * hessian
        + np.outer(polynomial_gradient, taper_gradient)
        + np.outer(taper_gradient, polynomial_gradient)
        + polynomial * taper_hessian
    )
    result_hessian = 0.5 * (result_hessian + result_hessian.T)
    return CompactJetEvaluation(float(value), result_gradient, result_hessian)


def calibrate_two_state_coupling_jet(
    diabatic_energies: np.ndarray,
    diabatic_gradients: np.ndarray,
    diabatic_hessians: np.ndarray,
    *,
    target_energy: float,
    target_gradient: np.ndarray,
    target_hessian: np.ndarray,
    sign: float = 1.0,
) -> EVBCouplingJet:
    """Recover the coupling two-jet that reproduces a target adiabatic jet."""

    energies = np.asarray(diabatic_energies, dtype=float).reshape(-1)
    gradients = np.asarray(diabatic_gradients, dtype=float)
    hessians = np.asarray(diabatic_hessians, dtype=float)
    target_gradient = np.asarray(target_gradient, dtype=float).reshape(-1)
    target_hessian = np.asarray(target_hessian, dtype=float)
    dimension = target_gradient.size
    if energies.shape != (2,) or gradients.shape != (2, dimension):
        raise ValueError("two-state coupling calibration requires two compatible diabatic jets")
    if hessians.shape != (2, dimension, dimension):
        raise ValueError("two-state coupling calibration has incompatible Hessians")
    if target_hessian.shape != (dimension, dimension):
        raise ValueError("target Hessian has incompatible shape")
    arrays = (energies, gradients, hessians, target_gradient, target_hessian)
    if any(not np.all(np.isfinite(value)) for value in arrays) or not np.isfinite(target_energy):
        raise ValueError("coupling calibration requires finite jets")
    gaps = energies - float(target_energy)
    if np.any(gaps <= 0.0):
        raise ValueError("target root must lie below both diabatic energies")
    orientation = 1.0 if float(sign) >= 0.0 else -1.0
    coupling = orientation * float(np.sqrt(gaps[0] * gaps[1]))
    relative_gradients = gradients - target_gradient[None, :]
    coupling_gradient = (
        relative_gradients[0] * gaps[1] + gaps[0] * relative_gradients[1]
    ) / (2.0 * coupling)
    relative_hessians = hessians - target_hessian[None, :, :]
    numerator = (
        relative_hessians[0] * gaps[1]
        + gaps[0] * relative_hessians[1]
        + np.outer(relative_gradients[0], relative_gradients[1])
        + np.outer(relative_gradients[1], relative_gradients[0])
    )
    coupling_hessian = numerator / (2.0 * coupling) - np.outer(
        coupling_gradient, coupling_gradient
    ) / coupling
    coupling_hessian = 0.5 * (coupling_hessian + coupling_hessian.T)
    return EVBCouplingJet(coupling, coupling_gradient, coupling_hessian)


def evaluate_multistate_evb(
    energies: np.ndarray,
    gradients: np.ndarray,
    couplings: np.ndarray,
    *,
    hessians: np.ndarray | None = None,
    coupling_gradients: np.ndarray | None = None,
    coupling_hessians: np.ndarray | None = None,
    root: int = 0,
) -> EVBEvaluation:
    """Evaluate one root of a real symmetric multistate EVB Hamiltonian.

    ``couplings`` contains the off-diagonal diabatic couplings and must have a
    zero diagonal. Its derivative arrays use shapes ``(n, n, d)`` and
    ``(n, n, d, d)``. When second derivatives are supplied, the returned
    Hessian includes the exact eigenvector-response term.
    """

    diagonal = np.asarray(energies, dtype=float).reshape(-1)
    gradient_rows = np.asarray(gradients, dtype=float)
    coupling_matrix = np.asarray(couplings, dtype=float)
    nstate = diagonal.size
    if nstate < 2 or not np.all(np.isfinite(diagonal)):
        raise ValueError("EVB requires at least two finite diabatic energies")
    if (
        gradient_rows.ndim != 2
        or gradient_rows.shape[0] != nstate
        or not np.all(np.isfinite(gradient_rows))
    ):
        raise ValueError("EVB gradients must have shape (nstate, ncoordinate)")
    if (
        coupling_matrix.shape != (nstate, nstate)
        or not np.all(np.isfinite(coupling_matrix))
        or not np.allclose(coupling_matrix, coupling_matrix.T, atol=1.0e-14, rtol=0.0)
        or not np.allclose(np.diag(coupling_matrix), 0.0, atol=1.0e-14, rtol=0.0)
    ):
        raise ValueError("EVB couplings must be finite, symmetric and have zero diagonal")
    if root < 0 or root >= nstate:
        raise ValueError("EVB root index is outside the diabatic state space")

    dimension = gradient_rows.shape[1]
    first = np.zeros((dimension, nstate, nstate), dtype=float)
    for coordinate in range(dimension):
        np.fill_diagonal(first[coordinate], gradient_rows[:, coordinate])
    if coupling_gradients is not None:
        coupling_first = np.asarray(coupling_gradients, dtype=float)
        if (
            coupling_first.shape != (nstate, nstate, dimension)
            or not np.all(np.isfinite(coupling_first))
            or not np.allclose(
                coupling_first, coupling_first.transpose(1, 0, 2), atol=1.0e-14, rtol=0.0
            )
        ):
            raise ValueError("EVB coupling gradients have invalid shape, values or symmetry")
        for coordinate in range(dimension):
            first[coordinate] += coupling_first[:, :, coordinate]

    matrix = np.diag(diagonal) + coupling_matrix
    values, vectors = np.linalg.eigh(matrix)
    coefficients = vectors[:, root]
    weights = coefficients * coefficients
    transformed_first = np.einsum("is,aij,jt->ast", vectors, first, vectors, optimize=True)
    gradient = transformed_first[:, root, root].copy()
    other_gaps = np.abs(values - values[root])
    other_gaps[root] = np.inf
    gap = float(np.min(other_gaps))

    result_hessian = None
    if hessians is not None or coupling_hessians is not None:
        second = np.zeros((dimension, dimension, nstate, nstate), dtype=float)
        if hessians is not None:
            diagonal_second = np.asarray(hessians, dtype=float)
            if (
                diagonal_second.shape != (nstate, dimension, dimension)
                or not np.all(np.isfinite(diagonal_second))
                or not np.allclose(
                    diagonal_second,
                    diagonal_second.transpose(0, 2, 1),
                    atol=1.0e-14,
                    rtol=0.0,
                )
            ):
                raise ValueError("EVB diabatic Hessians have invalid shape, values or symmetry")
            for a in range(dimension):
                for b in range(dimension):
                    np.fill_diagonal(second[a, b], diagonal_second[:, a, b])
        if coupling_hessians is not None:
            coupling_second = np.asarray(coupling_hessians, dtype=float)
            if (
                coupling_second.shape != (nstate, nstate, dimension, dimension)
                or not np.all(np.isfinite(coupling_second))
                or not np.allclose(
                    coupling_second,
                    coupling_second.transpose(1, 0, 2, 3),
                    atol=1.0e-14,
                    rtol=0.0,
                )
                or not np.allclose(
                    coupling_second,
                    coupling_second.transpose(0, 1, 3, 2),
                    atol=1.0e-14,
                    rtol=0.0,
                )
            ):
                raise ValueError("EVB coupling Hessians have invalid shape, values or symmetry")
            for a in range(dimension):
                for b in range(dimension):
                    second[a, b] += coupling_second[:, :, a, b]

        direct_second = np.einsum(
            "i,abij,j->ab", coefficients, second, coefficients, optimize=True
        )
        response = np.zeros((dimension, dimension), dtype=float)
        for state in range(nstate):
            if state == root:
                continue
            denominator = values[root] - values[state]
            left = transformed_first[:, root, state]
            right = transformed_first[:, state, root]
            response += (np.outer(left, right) + np.outer(right, left)) / denominator
        result_hessian = direct_second + response
        result_hessian = 0.5 * (result_hessian + result_hessian.T)

    return EVBEvaluation(
        float(values[root]), gradient, weights, gap, hessian=result_hessian
    )


def evaluate_two_state_evb(
    energies: tuple[float, float] | np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray] | np.ndarray,
    coupling: float,
    coupling_gradient: np.ndarray | None = None,
) -> EVBEvaluation:
    """Evaluate the lower eigenvalue of a two-state diabatic Hamiltonian."""

    diagonal = np.asarray(energies, dtype=float).reshape(-1)
    if diagonal.shape != (2,) or not np.all(np.isfinite(diagonal)):
        raise ValueError("EVB requires two finite diabatic energies")
    grad = np.asarray(gradients, dtype=float)
    if grad.ndim != 2 or grad.shape[0] != 2 or not np.all(np.isfinite(grad)):
        raise ValueError("EVB gradients must have shape (2, n) and be finite")
    coupling = float(coupling)
    if not np.isfinite(coupling):
        raise ValueError("EVB coupling must be finite")
    coupling_matrix = np.asarray(((0.0, coupling), (coupling, 0.0)), dtype=float)
    coupling_first = None
    if coupling_gradient is not None:
        derivative = np.asarray(coupling_gradient, dtype=float).reshape(-1)
        if derivative.shape != (grad.shape[1],) or not np.all(np.isfinite(derivative)):
            raise ValueError("EVB coupling gradient has invalid shape or values")
        coupling_first = np.zeros((2, 2, grad.shape[1]), dtype=float)
        coupling_first[0, 1] = coupling_first[1, 0] = derivative
    return evaluate_multistate_evb(
        diagonal,
        grad,
        coupling_matrix,
        coupling_gradients=coupling_first,
    )


def evaluate_two_state_evb_request(
    request: SurfaceRepresentationRequest,
    energies: tuple[float, float] | np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray] | np.ndarray,
    coupling: float,
    coupling_gradient: np.ndarray | None = None,
) -> EVBEvaluation:
    """Evaluate EVB only when the explicit representation/gate contract is present."""

    if request.provider != "EVB":
        raise ValueError("EVB evaluator requires provider='EVB'")
    validate_basin_transition(request)
    return evaluate_two_state_evb(energies, gradients, coupling, coupling_gradient)
