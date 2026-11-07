"""Energy-independent inner-region spectral representation, in atomic units.

The Bloch Hamiltonian includes +1/2 delta(r-a) d/dr. Boundary columns
contain reduced radial basis values at a. Our R has units of length:
u(a) = R(E) u'(a); it differs by a factor a from dimensionless R conventions.
No eigenpairs are discarded and no energy broadening is introduced.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.linalg import eigh, solve


def _real_matrix(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real for this backend")
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix")
    return array


@dataclass(frozen=True)
class BoundaryWaves:
    """Open-channel F,G and radial derivatives, energy normalized.

    Vectors denote diagonal channels; square matrices include a propagated
    coupled outer region. Columns refer to asymptotic channels.

    F is the regular standing wave; G+iF is outgoing. Channel thresholds,
    long-range potentials and radius must match the supplied inner operators.
    """

    regular: np.ndarray
    irregular: np.ndarray
    regular_derivative: np.ndarray
    irregular_derivative: np.ndarray


@dataclass(frozen=True)
class ContinuumResult:
    """Incoming-channel-normalized transition amplitudes and scattering matrix.

    Amplitudes have shape (number of source columns, number of channels).
    Angular Coulomb phases are not applied; the boundary convention is G+iF.
    """

    energy: float
    amplitudes: np.ndarray
    scattering: np.ndarray
    boundary_values: np.ndarray
    used_pole_fallback: bool


class ContinuumBackend(Protocol):
    """Replaceable solver contract, independent of electronic file formats."""

    def evaluate(self, energy: float, waves: BoundaryWaves) -> ContinuumResult: ...


class SpectralRMatrixBackend:
    """Diagonalize once; solve only channel-sized systems at each energy.

    H and S describe a finite real Galerkin space. B has one boundary column
    per channel. D contains integrals of each basis function with the physical
    source (for example each Cartesian dipole times a Dyson orbital).
    NumPy/SciPy delegate all dense contractions to compiled BLAS/LAPACK.
    """

    def __init__(self, hamiltonian, overlap, boundary, sources):
        h = _real_matrix(hamiltonian, "Hamiltonian")
        s = _real_matrix(overlap, "overlap")
        b = _real_matrix(boundary, "boundary")
        d = _real_matrix(sources, "sources")
        n = len(h)
        if n == 0 or h.shape != (n, n) or s.shape != (n, n):
            raise ValueError("Hamiltonian and overlap must be equal nonempty square matrices")
        if b.shape[0] != n or d.shape[0] != n or b.shape[1] == 0:
            raise ValueError("Boundary/source rows must match the inner-region basis")
        for name, matrix in (("Hamiltonian", h), ("overlap", s)):
            if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-13):
                raise ValueError(f"{name} must be symmetric")
        # eigh also checks positive definiteness of S; do not regularize it.
        self.eigenvalues, eigenvectors = eigh(h, s, check_finite=False)
        self._boundary = b.T @ eigenvectors
        self._sources = eigenvectors.T @ d
        self.nchannels = b.shape[1]

    def evaluate(self, energy: float, waves: BoundaryWaves) -> ContinuumResult:
        """Match an outgoing solution without K-matrix singularities.

        At an inner-region eigenvalue the R sum has a pole although the
        physical solution is finite. There we solve the modal outgoing system
        directly, rather than shift the energy or drop the pole.
        """
        energy = float(energy)
        if not np.isfinite(energy):
            raise ValueError("Energy must be finite")
        vectors = tuple(
            np.asarray(x)
            for x in (
                waves.regular,
                waves.irregular,
                waves.regular_derivative,
                waves.irregular_derivative,
            )
        )
        if any(
            np.iscomplexobj(x)
            or x.shape not in ((self.nchannels,), (self.nchannels, self.nchannels))
            or not np.all(np.isfinite(x))
            for x in vectors
        ):
            raise ValueError("Boundary waves must be finite real channel vectors or matrices")
        if any(x.ndim == 2 for x in vectors):
            return self._evaluate_coupled(
                energy, *(np.diag(x) if x.ndim == 1 else x for x in vectors)
            )
        f, g, fp, gp = vectors
        outgoing = g + 1j * f
        if np.any(outgoing == 0):
            raise ValueError("Outgoing boundary wave vanishes")
        q = (gp + 1j * fp) / outgoing
        driving = fp - q * f
        gap = self.eigenvalues - energy
        scale = max(1.0, abs(energy), float(np.max(np.abs(self.eigenvalues))))
        pole = bool(np.min(np.abs(gap)) <= 128 * np.finfo(float).eps * scale)
        w = self._boundary
        if pole:
            operator = np.diag(gap).astype(complex) - 0.5 * w.T @ (q[:, None] * w)
            coefficients = solve(operator, 0.5 * w.T * driving, check_finite=False)
            values = w @ coefficients
        else:
            resolvent_boundary = 0.5 * w.T / gap[:, None]
            r_matrix = w @ resolvent_boundary
            values = solve(
                np.eye(self.nchannels) - r_matrix * q,
                r_matrix * driving,
                check_finite=False,
            )
            coefficients = resolvent_boundary @ (np.diag(driving) + q[:, None] * values)
        transition = (values - np.diag(f)) / outgoing[:, None]
        scattering = np.eye(self.nchannels) + 2j * transition
        return ContinuumResult(energy, self._sources.T @ coefficients, scattering, values, pole)

    def _evaluate_coupled(self, energy, f, g, fp, gp):
        """Match full boundary matrices, preserving asymptotic channel labels."""
        h = g + 1j * f
        q = solve(h.T, (gp + 1j * fp).T, check_finite=False).T
        driving = fp - q @ f
        gap = self.eigenvalues - energy
        scale = max(1.0, abs(energy), float(np.max(np.abs(self.eigenvalues))))
        pole = bool(np.min(np.abs(gap)) <= 128 * np.finfo(float).eps * scale)
        w = self._boundary
        if pole:
            operator = np.diag(gap).astype(complex) - 0.5 * w.T @ q @ w
            coefficients = solve(operator, 0.5 * w.T @ driving, check_finite=False)
            values = w @ coefficients
        else:
            resolvent_boundary = 0.5 * w.T / gap[:, None]
            r_matrix = w @ resolvent_boundary
            values = solve(
                np.eye(self.nchannels) - r_matrix @ q, r_matrix @ driving, check_finite=False
            )
            coefficients = resolvent_boundary @ (driving + q @ values)
        scattering = np.eye(self.nchannels) + 2j * solve(h, values - f, check_finite=False)
        return ContinuumResult(energy, self._sources.T @ coefficients, scattering, values, pole)
