"""Compact atomic B-spline enrichment of the central spectral continuum basis.

Atomic supports are disjoint and strictly inside the outer boundary. Local
functions vanish with their first derivative at the support radius. By default
u_l(r)=O(r^(l+1)) at each nucleus; explicit radial selections also support
less smooth finite-energy H1 trial spaces for reference comparisons. No
potential softening or overlap truncation is introduced. Off-centre blocks are integrated only on their compact support.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import eigh
from scipy import sparse as sp
from scipy.special import roots_legendre

from .basis import RadialBasis, _bound_eigenpair
from .molecular import NuclearPotential, ProjectedLocalPotential
from .radial import assemble_radial


class SphericalSplineFunctions:
    """Cartesian evaluation of B_i(r)/r Y_lm about an explicit centre."""

    def __init__(
        self, radial, channels, centre_bohr=(0, 0, 0), *, compact=False, radial_indices=None
    ):
        centre = np.asarray(centre_bohr, dtype=float)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError("Centre must be a finite Cartesian vector")
        self.radial, self.channels, self.centre = radial, channels, centre.copy()
        self.compact = bool(compact)
        indices = []
        if radial_indices is not None and len(radial_indices) != len(channels.labels):
            raise ValueError("Provide radial indices for every angular channel")
        for c, (l, _) in enumerate(channels.labels):
            chosen = np.arange(l, radial.size - 2) if compact else np.arange(radial.size)
            if radial_indices is not None:
                raw = np.asarray(radial_indices[c])
                if raw.ndim != 1 or not np.all(np.isfinite(raw)) or np.any(raw != raw.astype(int)):
                    raise ValueError("Radial indices must be finite integer vectors")
                chosen = raw.astype(int)
                limit = radial.size - 2 if compact else radial.size
                if (
                    np.any(chosen < 0)
                    or np.any(chosen >= limit)
                    or len(np.unique(chosen)) != len(chosen)
                ):
                    raise ValueError("Radial selection violates basis/support constraints")
            if not len(chosen):
                raise ValueError("Atomic radial basis is too small for this angular momentum")
            indices.extend(c * radial.size + chosen)
        self.indices = np.asarray(indices, dtype=int)
        self.size = len(indices)
        self.angular_labels = np.repeat(channels.labels, radial.size, axis=0)[self.indices]
        self._channel_indices, self._radial_indices = np.divmod(self.indices, radial.size)

    def on_support(self, centre_bohr, radius):
        """Restrict columns to functions intersecting an integration sphere.

        Exact compact-support screening, with no magnitude cutoff. Bases
        without support metadata retain their full existing trial space.
        """
        centre = np.asarray(centre_bohr, dtype=float)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)) or not np.isfinite(radius) or radius < 0:
            raise ValueError("Integration sphere requires a finite centre and nonnegative radius")
        bounds = getattr(self.radial, "support_bounds", None)
        if bounds is None:
            return self
        distance = np.linalg.norm(centre - self.centre)
        lo, hi = max(0.0, distance - radius), distance + radius
        active = (bounds[:, 0] <= hi) & (bounds[:, 1] >= lo)
        selected = [
            self._radial_indices[(self._channel_indices == c) & active[self._radial_indices]]
            for c in range(len(self.channels.labels))
        ]
        if any(not len(indices) for indices in selected):
            # The present callers have overlapping central/atomic supports.
            # Preserve the established nonempty-channel API for other cases.
            return self
        return SphericalSplineFunctions(
            self.radial, self.channels, self.centre,
            compact=self.compact, radial_indices=selected,
        )

    def evaluate(self, points_bohr, *, gradients=False):
        points = np.asarray(points_bohr, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("Require finite (N,3) Cartesian points")
        xyz = points - self.centre
        r = np.linalg.norm(xyz, axis=1)
        inside = (r > 0) & (r <= self.radial.radius)
        output = np.zeros((len(r), self.size))
        grad = np.zeros((len(r), self.size, 3)) if gradients else None
        if np.any(inside):
            x, ri = xyz[inside], r[inside]
            radial = self.radial.evaluate(ri)
            if gradients:
                y, gy = self.channels.evaluate_with_gradient(x)
                dr = self.radial.evaluate_derivative(ri)
            else:
                y = self.channels.evaluate(x)
            rb = radial[:, self._radial_indices] / ri[:, None]
            yb = y[:, self._channel_indices]
            output[inside] = yb * rb
            if gradients:
                n = x / ri[:, None]
                gr = dr[:, self._radial_indices] / ri[:, None] - rb / ri[:, None]
                for axis in range(3):
                    grad[inside, :, axis] = (
                        yb * gr * n[:, axis, None] + gy[:, self._channel_indices, axis] * rb
                    )
        origin = r == 0
        if np.any(origin):
            if gradients:
                raise ValueError("Cusp gradients at the exact centre are undefined")
            # For compact functions l>0 vanish at the centre; s values follow
            # the analytic B_i(r)/r -> B'_i(0) limit.
            slope = self.radial.evaluate_derivative(np.array([0.0]))[0]
            vals = np.zeros((len(self.channels.labels), self.radial.size))
            for c, (l, m) in enumerate(self.channels.labels):
                if l == 0 and m == 0:
                    vals[c] = slope / np.sqrt(4 * np.pi)
            output[origin] = vals.ravel()[self.indices]
        return (output, grad) if gradients else output


class AtomicSplineBlock(SphericalSplineFunctions):
    """A compact atomic basis with explicitly chosen angular quadrature."""

    def __init__(
        self, centre_bohr, breakpoints, channels, *, degree=5, quadrature=12, radial_indices=None
    ):
        radial = RadialBasis(breakpoints, degree=degree, quadrature=quadrature)
        super().__init__(radial, channels, centre_bohr, compact=True, radial_indices=radial_indices)

    def quadrature_batches(self):
        """One radial shell at a time, with true three-dimensional weights."""
        for r, w in zip(self.radial.nodes, self.radial.weights):
            points = self.centre + r * self.channels.directions
            yield points, w * r * r * self.channels.weights


@dataclass(frozen=True)
class MulticentreBasis:
    central: SphericalSplineFunctions
    atoms: tuple[AtomicSplineBlock, ...]
    local_transform: np.ndarray

    @property
    def size(self):
        return self.central.size + self.local_transform.shape[1]

    def evaluate(self, points_bohr):
        central = self.central.evaluate(points_bohr)
        local = np.concatenate([atom.evaluate(points_bohr) for atom in self.atoms], axis=1)
        return np.column_stack((central, local @ self.local_transform))


@dataclass(frozen=True)
class MulticentreOperators:
    hamiltonian: np.ndarray
    overlap: np.ndarray
    boundary: np.ndarray
    sources: np.ndarray
    basis: MulticentreBasis
    radius: float


@dataclass(frozen=True)
class MulticentreBoundState:
    energy: float
    coefficients: np.ndarray
    basis: MulticentreBasis

    def __call__(self, points_bohr):
        return self.basis.evaluate(points_bohr) @ self.coefficients


def _validate_atoms(atoms, radius):
    for i, atom in enumerate(atoms):
        d = np.linalg.norm(atom.centre)
        if d < atom.radial.radius:
            raise ValueError("Local support must not contain the central origin in its interior")
        if d + atom.radial.radius >= radius:
            raise ValueError("Atomic support must lie strictly inside the outer boundary")
        for other in atoms[:i]:
            if (
                np.linalg.norm(atom.centre - other.centre)
                < atom.radial.radius + other.radial.radius
            ):
                raise ValueError(
                    "This initial compact enrichment requires non-overlapping atomic supports"
                )


def assemble_multicentre(
    central_operators,
    central_channels,
    atoms,
    charges,
    positions_bohr,
    *,
    local_transform=None,
    electronic_potential=None,
    reuse_electronic_samples=True,
    overlap_only=False,
):
    """Enrich existing central operators, retaining all central/atomic couplings.

    The optional transform maps raw atomic functions into a supplied symmetry
    subspace. It must have full column rank. No overlap eigenvector is silently
    deleted; an ill-conditioned basis needs an explicit caller decision.
    overlap_only constructs the same overlap without spatial potentials or
    Cartesian gradients, for an inexpensive check before electronic preparation.
    The Hamiltonian in that diagnostic mode must not be used for propagation.
    electronic_potential is a scalar spatial contribution consistent with the
    already assembled central block; no exchange model is selected here.
    """
    atoms = tuple(atoms)
    if not atoms or central_operators.basis is None:
        raise ValueError("Require atomic blocks and the central radial basis")
    _validate_atoms(atoms, central_operators.radius)
    central = SphericalSplineFunctions(central_operators.basis, central_channels)
    if central.size != central_operators.hamiltonian.shape[0]:
        raise ValueError("Central angular labels and operator dimensions differ")
    charges, positions = np.asarray(charges, float), np.asarray(positions_bohr, float)
    if (
        positions.shape != (len(charges), 3)
        or not np.all(np.isfinite(charges))
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("Require finite point charges and nuclear coordinates")
    nraw = sum(atom.size for atom in atoms)
    transform = np.eye(nraw) if local_transform is None else np.asarray(local_transform)
    if (
        np.iscomplexobj(transform)
        or transform.ndim != 2
        or transform.shape[0] != nraw
        or transform.shape[1] == 0
        or not np.all(np.isfinite(transform))
        or np.linalg.matrix_rank(transform) != transform.shape[1]
    ):
        raise ValueError("Atomic symmetry transform must be finite, real and have full column rank")
    transform = transform.astype(float, copy=True)
    sparse = sp.issparse(central_operators.hamiltonian)
    zeros = sp.lil_matrix if sparse else np.zeros
    hcl, scl = zeros((central.size, nraw)), zeros((central.size, nraw))
    hll, sll = zeros((nraw, nraw)), zeros((nraw, nraw))
    offset = 0
    for atom in atoms:
        selection = slice(offset, offset + atom.size)
        supported_central = central.on_support(atom.centre, atom.radial.radius)
        central_rows = supported_central.indices
        cross_h = np.zeros((len(central_rows), atom.size))
        cross_s = np.zeros_like(cross_h)
        if overlap_only:
            def local_v(r):
                return np.zeros((len(r), len(atom.channels.labels), len(atom.channels.labels)))
        else:
            local_v = NuclearPotential(atom.channels, charges, positions - atom.centre)
        if electronic_potential is not None and not overlap_only:
            projected = ProjectedLocalPotential(
                atom.channels, electronic_potential, origin_bohr=atom.centre
            )

            if reuse_electronic_samples:
                projected_values,electronic_samples=projected(atom.radial.nodes,return_samples=True)
                def potential(r,nuclear=local_v,values=projected_values,nodes=atom.radial.nodes):
                    if not np.array_equal(r,nodes):
                        raise ValueError('Atomic assembly and potential sample nodes differ')
                    return nuclear(r)+values
            else:
                def potential(r, nuclear=local_v, electronic=projected):
                    return nuclear(r) + electronic(r)
        else:
            potential = local_v
        # Exact separable kinetic/overlap and finite-space nuclear multipoles.
        local = assemble_radial(
            atom.radial.breakpoints,
            atom.channels.angular_momenta,
            potential,
            degree=atom.radial.degree,
            quadrature=atom.radial.quadrature,
        )
        ix = np.ix_(atom.indices, atom.indices)
        hll[selection, selection], sll[selection, selection] = (
            local.hamiltonian[ix],
            local.overlap[ix],
        )
        for shell,(points, weights) in enumerate(atom.quadrature_batches()):
            if overlap_only:
                c = supported_central.evaluate(points)
                a = atom.evaluate(points)
                cross_s += c.T @ (weights[:, None] * a)
                continue
            c, gc = supported_central.evaluate(points, gradients=True)
            a, ga = atom.evaluate(points, gradients=True)
            distances = np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2)
            if np.any(distances == 0):
                raise ValueError("Quadrature hits a nucleus; choose a non-singular rule")
            v = -np.sum(charges / distances, axis=1)
            if electronic_potential is not None:
                ve = (electronic_samples[shell] if reuse_electronic_samples else
                      np.asarray(electronic_potential(points)))
                if np.iscomplexobj(ve) or ve.shape != (len(points),) or not np.all(np.isfinite(ve)):
                    raise ValueError("Electronic potential must return finite real point values")
                v += ve
            cross_s += c.T @ (weights[:, None] * a)
            cross_h += c.T @ ((weights * v)[:, None] * a)
            for axis in range(3):
                cross_h += 0.5 * gc[:, :, axis].T @ (weights[:, None] * ga[:, :, axis])
        hcl[central_rows, selection] = cross_h
        scl[central_rows, selection] = cross_s
        offset += atom.size
    contraction = sp.csc_matrix(transform) if sparse else transform
    hcl, scl = hcl @ contraction, scl @ contraction
    hll, sll = contraction.T @ hll @ contraction, contraction.T @ sll @ contraction
    if sparse:
        h = sp.bmat([[central_operators.hamiltonian, hcl], [hcl.T, hll]], format="csc")
        s = sp.bmat([[central_operators.overlap, scl], [scl.T, sll]], format="csc")
        local_boundary = sp.csc_matrix((hll.shape[0], len(central_channels.labels)))
        b = sp.vstack([central_operators.boundary, local_boundary], format="csc")
    else:
        h = np.block([[central_operators.hamiltonian, hcl], [hcl.T, hll]])
        s = np.block([[central_operators.overlap, scl], [scl.T, sll]])
        b = np.vstack((central_operators.boundary, np.zeros((len(hll), len(central_channels.labels)))))
    return MulticentreOperators(
        h,
        s,
        b,
        np.empty((h.shape[0], 0)),
        MulticentreBasis(central, atoms, transform),
        central_operators.radius,
    )


def solve_multicentre_bound_state(operators, *, state_index=0):
    """Bound-state solve in the full enriched basis, with zero outer boundary."""
    if state_index < 0 or int(state_index) != state_index:
        raise ValueError("Bound state index must be a nonnegative integer")
    central = operators.basis.central
    nr = central.radial.size
    keep_c = np.arange(central.size).reshape(-1, nr)[:, :-1].ravel()
    keep = np.r_[keep_c, np.arange(central.size, operators.hamiltonian.shape[0])]
    if state_index >= len(keep):
        raise ValueError("Bound state index exceeds the trial space")
    ix = np.ix_(keep, keep)
    e, c = _bound_eigenpair(operators.hamiltonian[ix], operators.overlap[ix], state_index)
    if e >= 0:
        raise ValueError("Requested eigenstate is not bound")
    coeff = np.zeros(operators.hamiltonian.shape[0])
    coeff[keep] = c
    return MulticentreBoundState(float(e), coeff, operators.basis)


def with_multicentre_spatial_sources(final, spatial_sources, *, central_sources):
    """Project a supplied spatial source onto every compact atomic block.

    Central source integrals are supplied explicitly from the central owner.
    Atomic terms are integrated on their own supports, retaining the complete
    source rather than its truncated central harmonic expansion.
    """
    central = np.asarray(central_sources)
    if (central.ndim != 2 or central.shape[0] != final.basis.central.size
            or central.shape[1] == 0 or np.iscomplexobj(central)
            or not np.all(np.isfinite(central))):
        raise ValueError('Finite real central source matrix required')
    local = np.zeros((final.basis.local_transform.shape[0], central.shape[1]))
    offset = 0
    for atom in final.basis.atoms:
        for points, weights in atom.quadrature_batches():
            values = np.asarray(spatial_sources(points))
            if (values.shape != (len(points), central.shape[1])
                    or np.iscomplexobj(values) or not np.all(np.isfinite(values))):
                raise ValueError('Spatial source must return finite real point/source columns')
            local[offset:offset+atom.size] += atom.evaluate(points).T @ (weights[:,None]*values)
        offset += atom.size
    return replace(final, sources=np.vstack((central, final.basis.local_transform.T@local)))


def with_multicentre_dipole_sources(final, initial):
    """Contract the complete central+atomic bound state into all three dipoles.

    Central/central angular integrals are analytic finite harmonic contractions;
    terms containing an atomic function are integrated on that compact support.
    This retains the local bound-state contribution rather than reprojecting it
    into a truncated central angular expansion.
    """
    fb, ib = final.basis, initial.basis
    if fb.central.radial.radius != ib.central.radial.radius:
        raise ValueError("Initial/final inner boxes must coincide for source contraction")
    f, i = fb.central, ib.central
    knots = np.unique(np.r_[f.radial.breakpoints, i.radial.breakpoints])
    nodes, weights = roots_legendre(f.radial.degree + i.radial.degree + 2)
    half = np.diff(knots) / 2
    r = (knots[:-1, None] + half[:, None] * (nodes + 1)).ravel()
    w = (half[:, None] * weights).ravel()
    moment = f.radial.evaluate(r).T @ ((w * r)[:, None] * i.radial.evaluate(r))
    initial_c = initial.coefficients[: i.size].reshape(len(i.channels.labels), i.radial.size)
    angular = f.channels.dipole_couplings(i.channels)
    dc = np.einsum("fmi,ri->frm", angular, moment @ initial_c.T, optimize=True).reshape(f.size, 3)
    initial_local = ib.local_transform @ initial.coefficients[i.size :]
    offset = 0
    for atom in ib.atoms:
        coeff = initial_local[offset : offset + atom.size]
        supported_final = f.on_support(atom.centre, atom.radial.radius)
        for points, weights in atom.quadrature_batches():
            psi = atom.evaluate(points) @ coeff
            dc[supported_final.indices] += supported_final.evaluate(points).T @ ((weights * psi)[:, None] * points)
        offset += atom.size
    dl = np.zeros((fb.local_transform.shape[0], 3))
    offset = 0
    for atom in fb.atoms:
        supported_initial = i.on_support(atom.centre, atom.radial.radius)
        central_coeff = initial.coefficients[supported_initial.indices]
        for points, weights in atom.quadrature_batches():
            psi = supported_initial.evaluate(points) @ central_coeff
            initial_offset = 0
            for initial_atom in ib.atoms:
                end = initial_offset + initial_atom.size
                if np.linalg.norm(atom.centre - initial_atom.centre) <= (
                    atom.radial.radius + initial_atom.radial.radius
                ):
                    psi += initial_atom.evaluate(points) @ initial_local[initial_offset:end]
                initial_offset = end
            dl[offset : offset + atom.size] += atom.evaluate(points).T @ (
                (weights * psi)[:, None] * points
            )
        offset += atom.size
    return replace(final, sources=np.vstack((dc, fb.local_transform.T @ dl)))


def compact_overlap_diagnostics(operators, *, require_positive=False):
    """Diagnose compact enrichment using its normalized overlap Schur complement.

    The central radial overlap is assumed positive. Only the local Schur matrix
    is dense; no global eigensolve, basis deletion or overlap modification occurs.
    Underintegrated translated cross integrals can violate positivity even when
    both diagonal blocks are positive. Such a space cannot certify a continuum.
    """
    from scipy.linalg import cho_factor,cho_solve,eigvalsh
    from scipy.sparse.linalg import splu

    nc=operators.basis.central.size
    overlap=operators.overlap
    if overlap.shape[0]==nc:
        return dict(positive=True,local_dimension=0,minimum_normalized_schur_eigenvalue=None,
                    maximum_normalized_schur_eigenvalue=None,negative_directions=0)
    if sp.issparse(overlap):
        cross=overlap[:nc,nc:].toarray();local=overlap[nc:,nc:].toarray()
        solved=splu(overlap[:nc,:nc].tocsc()).solve(cross)
    else:
        cross=overlap[:nc,nc:];local=overlap[nc:,nc:]
        solved=cho_solve(cho_factor(overlap[:nc,:nc]),cross)
    diagonal=np.diag(local)
    if np.any(diagonal<=0):raise ValueError('Compact overlap diagonal must be positive')
    schur=local-cross.T@solved
    scale=np.sqrt(diagonal)
    normalized=schur/scale[:,None]/scale[None,:]
    eigen=eigvalsh((normalized+normalized.T)*.5)
    report=dict(positive=bool(eigen[0]>0),local_dimension=len(eigen),
        minimum_normalized_schur_eigenvalue=float(eigen[0]),
        maximum_normalized_schur_eigenvalue=float(eigen[-1]),
        negative_directions=int(np.count_nonzero(eigen<-1e-10)))
    if require_positive and not report['positive']:
        raise ValueError('Compact overlap is not positive: refine cross-integral quadrature or basis; '
                         f"normalized Schur minimum {eigen[0]:.6g}")
    return report
