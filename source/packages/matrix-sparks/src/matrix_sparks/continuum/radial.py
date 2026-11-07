"""General coupled radial B-spline Galerkin assembly; all quantities atomic units."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse as sp
from scipy.special import spherical_jn, spherical_yn

from .basis import RadialBasis, RadialFunctionBasis
from .spectral import BoundaryWaves


@dataclass(frozen=True)
class RadialOperators:
    """Channel-major operators with u(0)=0 and an open boundary at radius a."""

    hamiltonian: np.ndarray
    overlap: np.ndarray
    boundary: np.ndarray
    sources: np.ndarray
    radius: float
    basis: RadialFunctionBasis | None = None


def assemble_radial(
    breakpoints,
    angular_momenta,
    potential,
    *,
    degree=7,
    quadrature=12,
    source=None,
    radial_basis=None,
):
    """Assemble the Bloch Hamiltonian for an arbitrary real local coupled potential.

    potential(r) returns (npoint,nchannel,nchannel) in Hartree, excluding
    centrifugal terms. source(r) returns (npoint,nchannel,nsource), already
    projected onto angular channels and expressed as reduced radial functions.
    Breakpoints may include potential kinks. Quadrature/basis convergence must
    be checked for the supplied potential; no model-specific fitting is done.
    """
    return _assemble_radial(
        breakpoints,
        angular_momenta,
        potential,
        degree=degree,
        quadrature=quadrature,
        source=source,
        sparse=False,
        radial_basis=radial_basis,
    )


def assemble_sparse_radial(
    breakpoints,
    angular_momenta,
    potential,
    *,
    degree=7,
    quadrature=12,
    source=None,
    radial_basis=None,
):
    """Same Galerkin integrals as assemble_radial, stored directly as CSC.

    Exact B-spline support zeros are retained throughout assembly. No dense
    full-space Hamiltonian/overlap is constructed and no magnitude cutoff is
    applied. Potential samples still use the existing radial/channel contract.
    """
    return _assemble_radial(
        breakpoints,
        angular_momenta,
        potential,
        degree=degree,
        quadrature=quadrature,
        source=source,
        sparse=True,
        radial_basis=radial_basis,
    )


def _assemble_radial(
    breakpoints,
    angular_momenta,
    potential,
    *,
    degree,
    quadrature,
    source,
    sparse,
    radial_basis=None,
):
    radial = (
        RadialBasis(breakpoints, degree=degree, quadrature=quadrature)
        if radial_basis is None
        else radial_basis
    )
    if (
        not np.array_equal(radial.breakpoints, np.asarray(breakpoints))
        or radial.degree != degree
        or radial.quadrature != quadrature
    ):
        raise ValueError("Supplied radial basis must match breakpoints, degree and quadrature")
    raw_l = np.asarray(angular_momenta)
    if (
        raw_l.ndim != 1
        or not len(raw_l)
        or not np.all(np.isfinite(raw_l))
        or np.any(raw_l < 0)
        or np.any(raw_l != raw_l.astype(int))
    ):
        raise ValueError("Angular momenta must be nonnegative integers")
    r, weights = radial.nodes, radial.weights
    basis, derivative = radial.values, radial.derivatives
    if sparse:
        b, db = sp.csr_matrix(basis), sp.csr_matrix(derivative)
        gram = lambda w: (b.T @ b.multiply(w[:, None])).tocsc()
        overlap_radial = gram(weights)
        kinetic = .5 * (db.T @ db.multiply(weights[:, None])).tocsc()
        centrifugal = gram(weights / r**2)
    else:
        gram = lambda w: basis.T @ (w[:, None] * basis)
        overlap_radial = gram(weights)
        kinetic = 0.5 * derivative.T @ (weights[:, None] * derivative)
        centrifugal = gram(weights / r**2)
    nc, nr = len(raw_l), radial.size
    v = np.asarray(potential(r))
    if np.iscomplexobj(v) or v.shape != (len(r), nc, nc) or not np.all(np.isfinite(v)):
        raise ValueError("Potential must be a finite real channel matrix at every radial point")
    if not np.allclose(v, v.swapaxes(1, 2), rtol=1e-12, atol=1e-13):
        raise ValueError("Potential channel matrices must be symmetric")
    h = [[None] * nc for _ in range(nc)] if sparse else np.empty((nc, nr, nc, nr))
    for i in range(nc):
        for j in range(i + 1):
            block = gram(weights * v[:, i, j])
            if i == j:
                block += kinetic + 0.5 * raw_l[i] * (raw_l[i] + 1) * centrifugal
            if sparse:
                h[i][j], h[j][i] = block, block.T
            else:
                h[i, :, j, :] = block
                h[j, :, i, :] = block.T
    if sparse:
        h = sp.bmat(h, format="csc")
        overlap = sp.kron(sp.eye(nc), overlap_radial, format="csc")
        boundary = sp.kron(sp.eye(nc), sp.csc_matrix(radial.boundary[:, None]), format="csc")
    else:
        h = h.reshape(nc * nr, nc * nr)
        overlap = np.kron(np.eye(nc), overlap_radial)
        boundary = np.kron(np.eye(nc), radial.boundary[:, None])
    d = project_radial_sources(radial, nc, source)
    return RadialOperators(
        h,
        overlap,
        boundary,
        d,
        radial.radius,
        radial,
    )


def project_radial_sources(radial, nchannels, source):
    """Project arbitrary real sources in an existing radial/channel space.

    This is the same source quadrature used by Hamiltonian assembly. It can
    be reused for alternative orbitals without rebuilding the local field.
    """
    if int(nchannels) != nchannels or nchannels < 1:
        raise ValueError("Require a positive integer channel count")
    nc = int(nchannels)
    if source is None:
        return np.empty((nc*radial.size, 0))
    values = np.asarray(source(radial.nodes))
    if (np.iscomplexobj(values) or values.ndim != 3
            or values.shape[:2] != (len(radial.nodes),nc)
            or not np.all(np.isfinite(values))):
        raise ValueError("Source must have finite real shape (points,channels,sources)")
    return np.einsum("ri,rcs,r->cis",radial.values,values,radial.weights,
                     optimize=True).reshape(nc*radial.size,values.shape[2])


def free_boundary(energy, radius, angular_momenta):
    """Energy-normalized neutral-channel Riccati-Bessel waves, with G=-rho*y_l."""
    if not np.isfinite(energy) or energy <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Open-channel energy and radius must be finite and positive")
    raw_l = np.asarray(angular_momenta)
    if raw_l.ndim != 1 or np.any(raw_l < 0) or np.any(raw_l != raw_l.astype(int)):
        raise ValueError("Angular momenta must be nonnegative integers")
    l = raw_l.astype(int)
    k = np.sqrt(2 * energy)
    rho = k * radius
    norm = np.sqrt(2 / (np.pi * k))
    j, y = spherical_jn(l, rho), spherical_yn(l, rho)
    return BoundaryWaves(
        norm * rho * j,
        -norm * rho * y,
        norm * k * (j + rho * spherical_jn(l, rho, derivative=True)),
        -norm * k * (y + rho * spherical_yn(l, rho, derivative=True)),
    )
