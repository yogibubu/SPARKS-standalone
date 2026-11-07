"""Read the diagnostic reference exports without modifying their operators."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class ReferenceOperators:
    strong_hamiltonian: np.ndarray
    overlap: np.ndarray
    # (basis, degenerate component, Cartesian dipole, bound-state component).
    sources: np.ndarray


@dataclass(frozen=True)
class ReferenceVector:
    energy: float
    charge: float
    angular_momenta: np.ndarray
    radial_counts: np.ndarray
    knots: np.ndarray
    degree: int
    vectors: np.ndarray
    regular_inverse: np.ndarray
    asymptotic_shift: float = 0.0
    matching_mode: int = 1
    matching_type: int = 0

    def boundary_operators(self):
        """Return the reduced radial boundary value and derivative columns."""
        nc = len(self.angular_momenta)
        count = len(self.knots) - self.degree - 1
        radial = BSpline(self.knots, np.eye(count), self.degree)
        value = radial(self.knots[-1])[1:]
        derivative = radial.derivative()(self.knots[-1])[1:]
        if np.any(self.radial_counts != len(value)):
            raise ValueError("Reference radial truncation needs an explicit basis map")
        boundary = np.zeros((len(self.vectors), nc))
        slope = boundary.copy()
        for i in range(nc):
            rows = slice(i * len(value), (i + 1) * len(value))
            boundary[rows, i] = value
            slope[rows, i] = derivative
        return boundary, slope


def read_vector(path):
    with Path(path).open("rb") as stream:
        n, nc, na, nk, order, nintervals = np.fromfile(stream, dtype="<i4", count=6)
        energy, charge = np.fromfile(stream, dtype="<f8", count=2)
        angular = np.fromfile(stream, dtype="<i4", count=na)
        counts = np.fromfile(stream, dtype="<i4", count=na)
        knots = np.fromfile(stream, dtype="<f8", count=nk)
        # Early diagnostic exports included the allocated, unused knot tail.
        needed = int(np.max(counts)) + int(order) + 1
        if nk < needed:
            raise ValueError("Incomplete reference knot vector")
        knots = knots[:needed]
        vec = np.fromfile(stream, dtype="<f8", count=n * nc).reshape(n, nc, order="F")
        inverse = np.fromfile(stream, dtype="<f8", count=nc * nc).reshape(nc, nc, order="F")
        # The export also retains K eigenvectors and scratch K storage for audit.
        extra = np.fromfile(stream, dtype="<f8")
        if len(extra) != 2 * nc * nc or nc != na or nintervals < 1:
            raise ValueError("Unexpected reference vector layout")
    sidecar = Path(path).with_name(Path(path).name.replace("eq-vector-", "eq-matching-"))
    shift, mode, kind = 0.0, 1, 0
    if sidecar.exists():
        with sidecar.open("rb") as stream:
            stored_energy, stored_charge, shift = np.fromfile(stream, dtype="<f8", count=3)
            mode, kind = np.fromfile(stream, dtype="<i4", count=2)
        if stored_energy != energy or stored_charge != charge:
            raise ValueError("Matching metadata belongs to a different continuum frame")
    return ReferenceVector(
        float(energy),
        float(charge),
        angular,
        counts,
        knots,
        int(order) - 1,
        vec,
        inverse,
        float(shift),
        int(mode),
        int(kind),
    )


def read_operators(directory, active_index):
    directory = Path(directory)
    h = s = None
    coverage = None
    for path in sorted(directory.glob(f"eq-hs-{active_index:04d}-r*.bin")):
        with path.open("rb") as stream:
            n, ldi, ldj, pr, pc, ir, ic, br, bc = np.fromfile(stream, dtype="<i4", count=9)
            matrix = np.fromfile(stream, dtype="<f8")
        if len(matrix) != 2 * ldi * ldj:
            raise ValueError("Malformed distributed operator block")
        if h is None:
            h = np.zeros((n, n))
            s = h.copy()
            coverage = np.zeros((n, n), dtype=np.int8)
        rows = np.flatnonzero((np.arange(n) // br) % pr == ir)
        cols = np.flatnonzero((np.arange(n) // bc) % pc == ic)
        if len(rows) > ldi or len(cols) > ldj:
            raise ValueError("ScaLAPACK descriptor does not cover local data")
        ix = np.ix_(rows, cols)
        s[ix] = matrix[: ldi * ldj].reshape(ldi, ldj, order="F")[: len(rows), : len(cols)]
        h[ix] = matrix[ldi * ldj :].reshape(ldi, ldj, order="F")[: len(rows), : len(cols)]
        coverage[ix] += 1
    if h is None or not np.all(coverage == 1):
        raise ValueError("Distributed H/S export is missing or overlapping")
    source = None
    for kind in ("central", "local"):
        paths = sorted(directory.glob(f"eq-source-{kind}-{active_index:04d}-r*.bin"))
        if not paths:
            raise ValueError("Dipole source export is missing")
        for path in paths:
            with path.open("rb") as stream:
                n, deg, nb = np.fromfile(stream, dtype="<i4", count=3)
                data = np.fromfile(stream, dtype="<f8")
            if len(data) != n * deg * 3 * nb:
                raise ValueError("Malformed dipole source export")
            data = data.reshape(n, deg, 3, nb, order="F")
            if source is None:
                source = np.zeros_like(data)
            source += data
    source = np.sqrt(4 * np.pi / 3) * source[:, :, [2, 0, 1], :]
    return ReferenceOperators(h, s, source)
