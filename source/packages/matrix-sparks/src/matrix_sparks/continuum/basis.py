"""Reusable reduced radial B-spline basis and finite-box bound states."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from scipy.special import roots_legendre


class RadialFunctionBasis(Protocol):
    """Common evaluation contract for radial B-spline and finite-element bases."""

    breakpoints: np.ndarray
    degree: int
    quadrature: int
    radius: float
    size: int
    nodes: np.ndarray
    weights: np.ndarray
    boundary: np.ndarray

    def evaluate(self, radii) -> np.ndarray: ...

    def evaluate_derivative(self, radii) -> np.ndarray: ...


class RadialBasis:
    """Open clamped B-splines with u(0)=0 and optional integration-only cuts.

    Extra quadrature breakpoints resolve moving potential kinks without adding
    trial functions or changing the spline knots across a derivative stencil.
    """

    def __init__(self, breakpoints, *, degree=7, quadrature=12, quadrature_breakpoints=None):
        points = np.asarray(breakpoints, dtype=float)
        if (
            points.ndim != 1
            or len(points) < 2
            or points[0] != 0
            or not np.all(np.isfinite(points))
            or np.any(np.diff(points) <= 0)
        ):
            raise ValueError("Breakpoints must increase from zero to a finite radius")
        if (
            degree < 1
            or int(degree) != degree
            or int(quadrature) != quadrature
            or quadrature < degree + 1
        ):
            raise ValueError("Use integer degree >=1 and integer quadrature >=degree+1")
        self.radius = float(points[-1])
        self.breakpoints = points.copy()
        self.degree = int(degree)
        self.quadrature = int(quadrature)
        degree, quadrature = int(degree), int(quadrature)
        knots = np.r_[np.repeat(points[0], degree), points, np.repeat(points[-1], degree)]
        integration = points
        if quadrature_breakpoints is not None:
            raw_cuts = np.asarray(quadrature_breakpoints)
            if np.iscomplexobj(raw_cuts):
                raise ValueError("Quadrature breakpoints must be real")
            cuts = np.asarray(raw_cuts, dtype=float)
            if (
                cuts.ndim != 1
                or not np.all(np.isfinite(cuts))
                or np.any(cuts < 0)
                or np.any(cuts > self.radius)
            ):
                raise ValueError("Quadrature breakpoints must lie inside the fixed spline box")
            integration = np.unique(np.r_[points, cuts])
        self.integration_breakpoints = integration.copy()
        nodes, weights = roots_legendre(quadrature)
        half = np.diff(integration) / 2
        self.nodes = (integration[:-1, None] + half[:, None] * (nodes + 1)).ravel()
        self.weights = (half[:, None] * weights).ravel()
        count = len(knots) - degree - 1
        self._spline = BSpline(knots, np.eye(count), degree, extrapolate=False)
        self.size = count - 1
        self.support_bounds = np.column_stack((knots[1:count], knots[degree + 2:count + degree + 1]))
        self.values = self.evaluate(self.nodes)
        self._derivative = self._spline.derivative()
        self.derivatives = self._derivative(self.nodes)[:, 1:]
        self.boundary = self._spline(self.radius)[1:]

    def evaluate(self, radii):
        """Basis values on an array of radii within [0,a]."""
        r = np.asarray(radii, dtype=float)
        if r.ndim != 1 or np.any(r < 0) or np.any(r > self.radius) or not np.all(np.isfinite(r)):
            raise ValueError("Evaluation radii must lie within the basis interval")
        return self._spline(r)[:, 1:]

    def evaluate_derivative(self, radii):
        """Analytic radial derivatives, with the same shape/order as evaluate."""
        r = np.asarray(radii, dtype=float)
        if r.ndim != 1 or np.any(r < 0) or np.any(r > self.radius) or not np.all(np.isfinite(r)):
            raise ValueError("Evaluation radii must lie within the basis interval")
        return self._derivative(r)[:, 1:]


@dataclass(frozen=True)
class BoundRadialState:
    """S-normalized finite-box bound solution, including zero endpoint columns."""

    energy: float
    coefficients: np.ndarray
    basis: RadialFunctionBasis

    def evaluate(self, radii):
        return self.basis.evaluate(radii) @ self.coefficients.T


def _bound_eigenpair(h, s, state_index):
    """Retain the dense reference path; use partial Lanczos for sparse input."""
    index = int(state_index)
    if issparse(h) and index + 1 < h.shape[0]:
        # A fixed generic starting vector makes qualification reproducible.
        v0 = np.random.default_rng(261026).normal(size=h.shape[0])
        energies, vectors = eigsh(h, M=s, k=index + 1, which="SA", tol=1e-11, v0=v0)
        chosen = np.argsort(energies)[index]
        return float(energies[chosen]), vectors[:, chosen]
    if issparse(h):
        # Requesting the complete spectrum is explicitly a dense operation.
        h, s = h.toarray(), s.toarray()
    energies, vectors = eigh(h, s, subset_by_index=(index, index), check_finite=False)
    return float(energies[0]), vectors[:, 0]


def solve_bound_state(operators, *, state_index=0):
    """Solve a real local finite-box bound state with u(a)=0.

    This is a one-particle bound problem, not a correlated electronic method.
    Box/angular/radial convergence is the caller's responsibility.
    """
    if state_index < 0 or int(state_index) != state_index:
        raise ValueError("Bound state index must be a nonnegative integer")
    basis = operators.basis
    if basis is None:
        raise ValueError("Bound-state reconstruction requires a radial basis")
    nr = basis.size
    nc = operators.hamiltonian.shape[0] // nr
    keep = np.arange(nc * nr).reshape(nc, nr)[:, :-1].ravel()
    if state_index >= len(keep):
        raise ValueError("Bound state index exceeds the trial space")
    ix = np.ix_(keep, keep)
    energy, vector = _bound_eigenpair(operators.hamiltonian[ix], operators.overlap[ix], state_index)
    if energy >= 0:
        raise ValueError("Requested finite-box eigenstate is not bound")
    coefficients = np.zeros(nc * nr)
    coefficients[keep] = vector
    return BoundRadialState(energy, coefficients.reshape(nc, nr), basis)


def bound_dipole_source(initial_channels, bound_state, final_channels):
    """Return a callable reduced dipole source r <Yf|n_xyz|Yi> u_i(r)."""
    if len(initial_channels.labels) != len(bound_state.coefficients):
        raise ValueError("Bound coefficients and initial angular channels differ")
    couplings = final_channels.dipole_couplings(initial_channels)

    def source(radii):
        r = np.asarray(radii, dtype=float)
        return r[:, None, None] * np.einsum(
            "fmi,ri->rfm", couplings, bound_state.evaluate(r), optimize=True
        )

    return source
