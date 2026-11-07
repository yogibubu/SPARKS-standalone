"""Coulomb matching without a TIRESIA or Fortran dependency.

Outgoing asymptotic expansion: DLMF 33.11.1 and phase 33.2.9. Standard SciPy
DOP853 propagates all requested angular orders together to the matching radius.
This initial general provider prioritizes controlled accuracy; timing and
large-angular-momentum/near-threshold limits must be checked for each use.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import loggamma

from .spectral import BoundaryWaves


def _outgoing_series(eta, orders, rho, atol):
    """Outgoing Coulomb amplitude series and its dimensionless derivative."""
    term = np.ones(len(orders), dtype=complex)
    series = term.copy()
    derivative = np.zeros(len(orders), dtype=complex)
    for n in range(64):
        term *= ((n + 1j * eta) * (n + 1 + 1j * eta) - orders * (orders + 1)) / (
            2j * (n + 1) * rho
        )
        series += term
        derivative -= (n + 1) * term / rho
        if np.max(np.abs(term)) < min(1e-15, atol * 1e-3):
            return series, derivative
    raise ArithmeticError("Coulomb outgoing expansion did not reach its requested accuracy")


def coulomb_boundary(energy, radius, angular_momenta, charge, *, rtol=2e-11, atol=2e-12):
    """Return energy-normalized F,G and r derivatives for open Coulomb channels.

    V=-charge/r, eta=-charge/k. No artificial broadening or phase adjustment.
    The outgoing expansion is summed at a sufficiently distant rho and checked
    before propagation; unsupported scales raise rather than silently truncate.
    """
    raw = np.asarray(angular_momenta)
    if (
        raw.ndim != 1
        or not len(raw)
        or not np.all(np.isfinite(raw))
        or np.any(raw < 0)
        or np.any(raw != raw.astype(int))
    ):
        raise ValueError("Angular momenta must be nonnegative integers")
    if (
        not np.all(np.isfinite([energy, radius, charge, rtol, atol]))
        or energy <= 0
        or radius <= 0
        or min(rtol, atol) <= 0
    ):
        raise ValueError("Require finite charge, positive energy/radius and tolerances")
    requested, inverse = np.unique(raw.astype(int), return_inverse=True)
    if charge == 0:
        from .radial import free_boundary

        return free_boundary(energy, radius, raw)
    k = np.sqrt(2 * energy)
    eta, target = -charge / k, k * radius
    # In the oscillatory region an upward angular recurrence is stable. Solve
    # only l=0 there, rather than integrating every l from a distant lmax^2
    # asymptotic radius. Near/inside the turning point retain direct propagation.
    turning = eta + np.sqrt(eta**2 + requested.max() * (requested.max() + 1))
    recur = target > turning + 2
    l = np.array([0]) if recur else requested
    start = max(target, 64.0, 4 * (eta**2 + l.max() * (l.max() + 1) + 1))
    series, derivative = _outgoing_series(eta, l, start, atol)
    sigma = np.imag(loggamma(l + 1 + 1j * eta))
    phase = np.exp(1j * (start - eta * np.log(2 * start) - np.pi * l / 2 + sigma))
    h = phase * series
    hp = phase * (1j * (1 - eta / start) * series + derivative)
    if start > target:

        def rhs(rho, state):
            u, up = state.reshape(2, len(l))
            return np.r_[up, -(1 - 2 * eta / rho - l * (l + 1) / rho**2) * u]

        result = solve_ivp(
            rhs,
            (start, target),
            np.r_[h, hp],
            method="DOP853",
            rtol=rtol,
            atol=atol,
            t_eval=[target],
        )
        if not result.success:
            raise ArithmeticError(f"Coulomb propagation failed: {result.message}")
        h, hp = result.y[:, -1].reshape(2, len(l))
    if recur:
        sequence = np.empty(int(requested.max()) + 2, dtype=complex)
        sequence[0] = h[0]
        sequence[1] = ((1 / target + eta) * h[0] - hp[0]) / np.sqrt(1 + eta**2)
        for order in range(1, len(sequence) - 1):
            sequence[order + 1] = (
                (2 * order + 1) * (eta + order * (order + 1) / target) * sequence[order]
                - (order + 1) * np.sqrt(order**2 + eta**2) * sequence[order - 1]
            ) / (order * np.sqrt((order + 1) ** 2 + eta**2))
        h = sequence[requested]
        hp = ((requested + 1) / target + eta / (requested + 1)) * h - np.sqrt(
            1 + (eta / (requested + 1)) ** 2
        ) * sequence[requested + 1]
    norm = np.sqrt(2 / (np.pi * k))
    f, g = norm * h.imag[inverse], norm * h.real[inverse]
    fp, gp = norm * k * hp.imag[inverse], norm * k * hp.real[inverse]
    wronskian = g * fp - f * gp
    if not np.allclose(wronskian, 2 / np.pi, rtol=max(2e-8, 100 * rtol), atol=100 * atol):
        raise ArithmeticError("Coulomb normalization lost during propagation")
    return BoundaryWaves(f, g, fp, gp)


def propagate_boundary(
    energy, inner_radius, outer_radius, angular_momenta, potential, waves, *, rtol=2e-10, atol=2e-11
):
    """Propagate asymptotic channel matrices through a molecular potential tail.

    The supplied waves are defined at outer_radius. The full local potential,
    excluding centrifugal terms, is evaluated from inner_radius to outer_radius.
    Outside outer_radius the supplied asymptotic model is still an approximation
    for a general molecule; convergence in outer_radius must be demonstrated.
    """
    l = np.asarray(angular_momenta)
    if (
        l.ndim != 1
        or not len(l)
        or not np.all(np.isfinite(l))
        or np.any(l < 0)
        or np.any(l != l.astype(int))
    ):
        raise ValueError("Angular momenta must be nonnegative integers")
    if (
        not np.all(np.isfinite([energy, inner_radius, outer_radius, rtol, atol]))
        or energy <= 0
        or inner_radius <= 0
        or outer_radius < inner_radius
        or min(rtol, atol) <= 0
    ):
        raise ValueError("Require open energy, positive tolerances and 0<inner<=outer radius")
    nc = len(l)
    matrices = []
    for values in (
        waves.regular,
        waves.irregular,
        waves.regular_derivative,
        waves.irregular_derivative,
    ):
        array = np.asarray(values)
        if (
            np.iscomplexobj(array)
            or array.shape not in ((nc,), (nc, nc))
            or not np.all(np.isfinite(array))
        ):
            raise ValueError("Boundary arrays must be real finite channel vectors or matrices")
        matrices.append(np.diag(array) if array.ndim == 1 else array)
    if inner_radius == outer_radius:
        return BoundaryWaves(*matrices)

    def rhs(r, state):
        f, g, fp, gp = state.reshape(4, nc, nc)
        v = np.asarray(potential(np.array([r])))
        if v.shape != (1, nc, nc) or np.iscomplexobj(v) or not np.all(np.isfinite(v)):
            raise ValueError(
                "Tail potential must return real finite (points,channels,channels) matrices"
            )
        operator = 2 * (v[0] - energy * np.eye(nc)) + np.diag(l * (l + 1) / r**2)
        return np.stack((fp, gp, operator @ f, operator @ g)).ravel()

    solution = solve_ivp(
        rhs,
        (outer_radius, inner_radius),
        np.stack(matrices).ravel(),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=[inner_radius],
    )
    if not solution.success:
        raise ArithmeticError(f"Molecular tail propagation failed: {solution.message}")
    return BoundaryWaves(*solution.y[:, -1].reshape(4, nc, nc))
