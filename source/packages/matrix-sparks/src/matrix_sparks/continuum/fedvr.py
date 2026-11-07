"""Finite-element Gauss-Lobatto DVR for coupled radial continuum equations.

Adjacent elements share one continuous bridge function. The origin degree of
freedom is removed and the outer endpoint is retained for R-matrix matching.
The DVR mass and local potential use Lobatto quadrature; kinetic integrals of
the polynomial derivatives are exact. All quantities are atomic units.
"""

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.special import eval_legendre, roots_jacobi

from .radial import RadialOperators


class FiniteElementDVR:
    """Normalized piecewise Lagrange functions, with a diagonal DVR mass."""

    def __init__(self, breakpoints, *, order=10):
        points = np.asarray(breakpoints, dtype=float)
        if (
            points.ndim != 1
            or len(points) < 2
            or points[0] != 0
            or not np.all(np.isfinite(points))
            or np.any(np.diff(points) <= 0)
        ):
            raise ValueError("Element boundaries must increase from zero")
        if int(order) != order or order < 3:
            raise ValueError("Use integer Lobatto order >=3")
        order = int(order)
        self.breakpoints = points.copy()
        self.radius = float(points[-1])
        self.degree = order - 1
        self.quadrature = order
        self.order = order
        x = np.r_[-1.0, roots_jacobi(order - 2, 1, 1)[0], 1.0]
        w = 2 / (order * (order - 1) * eval_legendre(order - 1, x) ** 2)
        self._cardinal = BarycentricInterpolator(x, np.eye(order))
        self._reference_nodes = x
        self._reference_weights = w
        count = (len(points) - 1) * (order - 1) + 1
        all_nodes = np.empty(count)
        all_weights = np.zeros(count)
        self._elements = []
        for e, (left, right) in enumerate(zip(points[:-1], points[1:])):
            half = (right - left) / 2
            indices = e * (order - 1) + np.arange(order)
            all_nodes[indices] = left + half * (x + 1)
            all_weights[indices] += half * w
            self._elements.append((indices, half))
        self._normalization = 1 / np.sqrt(all_weights)
        self.nodes = all_nodes[1:]
        self.weights = all_weights[1:]
        self.size = count - 1
        self.boundary = np.zeros(self.size)
        self.boundary[-1] = self._normalization[-1]

    def _evaluate(self, radii, derivative):
        r = np.asarray(radii, dtype=float)
        if r.ndim != 1 or not np.all(np.isfinite(r)) or np.any(r < 0) or np.any(r > self.radius):
            raise ValueError("Evaluation radii must be inside the finite box")
        out = np.zeros((len(r), self.size + 1))
        element = np.minimum(
            np.searchsorted(self.breakpoints, r, side="right") - 1, len(self._elements) - 1
        )
        for e in np.unique(element):
            selected = np.flatnonzero(element == e)
            indices, half = self._elements[e]
            x = (r[selected] - self.breakpoints[e]) / half - 1
            value = self._cardinal.derivative(x) / half if derivative else self._cardinal(x)
            out[np.ix_(selected, indices)] = value * self._normalization[indices]
        return out[:, 1:]

    def evaluate(self, radii):
        return self._evaluate(radii, False)

    def evaluate_derivative(self, radii):
        """Right derivative at internal bridges; left derivative at the box end."""
        return self._evaluate(radii, True)

    def kinetic(self):
        """Exact weak kinetic integral; sparsity follows element connectivity."""
        derivative = self._cardinal.derivative(self._reference_nodes)
        local = derivative.T @ (self._reference_weights[:, None] * derivative)
        matrix = np.zeros((self.size + 1, self.size + 1))
        for indices, half in self._elements:
            normalization = self._normalization[indices]
            matrix[np.ix_(indices, indices)] += (
                0.5 / half * local * normalization[:, None] * normalization
            )
        return matrix[1:, 1:]


def assemble_fedvr(breakpoints, angular_momenta, potential, *, order=10, source=None):
    """Assemble a radial DVR with full angular-channel coupling at every node.

    The potential and source contracts are identical to assemble_radial. A
    Lobatto grid is a quadrature approximation for non-polynomial potentials;
    element/order convergence must be established for each physical model.
    """
    basis = FiniteElementDVR(breakpoints, order=order)
    l = np.asarray(angular_momenta)
    if (
        l.ndim != 1
        or not len(l)
        or not np.all(np.isfinite(l))
        or np.any(l < 0)
        or np.any(l != l.astype(int))
    ):
        raise ValueError("Angular momenta must be nonnegative integers")
    r = basis.nodes
    nc, nr = len(l), basis.size
    v = np.asarray(potential(r))
    if (
        np.iscomplexobj(v)
        or v.shape != (nr, nc, nc)
        or not np.all(np.isfinite(v))
        or not np.allclose(v, v.swapaxes(1, 2), rtol=1e-12, atol=1e-13)
    ):
        raise ValueError("Potential must return finite real symmetric channel matrices")
    kinetic = basis.kinetic()
    h = np.zeros((nc, nr, nc, nr))
    index = np.arange(nr)
    for i in range(nc):
        for j in range(nc):
            h[i, index, j, index] = v[:, i, j]
        h[i, :, i, :] += kinetic + np.diag(0.5 * l[i] * (l[i] + 1) / r**2)
    d = np.empty((nc * nr, 0))
    if source is not None:
        values = np.asarray(source(r))
        if (
            np.iscomplexobj(values)
            or values.ndim != 3
            or values.shape[:2] != (nr, nc)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Source must return finite real (nodes,channels,sources)")
        d = (np.sqrt(basis.weights)[:, None, None] * values).transpose(1, 0, 2).reshape(nc * nr, -1)
    return RadialOperators(
        h.reshape(nc * nr, nc * nr),
        np.eye(nc * nr),
        np.kron(np.eye(nc), basis.boundary[:, None]),
        d,
        basis.radius,
        basis,
    )
