"""Retarded independent-particle and screened continuum response.

All energies and operators are in atomic units, with exp(-i*omega*t) time
convention. Potential coefficients and induced-density moments are dual:
chi maps potential coefficients to density moments, and kernel maps those
moments back to potential coefficients. A nonorthogonal auxiliary basis must
therefore supply the metric-transformed kernel, not bare two-electron integrals.
No XC functional, frequency broadening or boundary condition is selected here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import splu

from .resolvent import _real_sparse_matrix, _symmetric


def _finite_matrix(value, name):
    result = np.asarray(value, dtype=complex if np.iscomplexobj(value) else float)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


@dataclass(frozen=True)
class ScreenedResponseResult:
    """Effective potentials, induced moments and normwise backward error."""

    effective_potential: np.ndarray
    induced_density: np.ndarray
    backward_error: float

    def amplitudes(self, potential_vertices):
        """Contract <continuum|P_alpha|initial> without conjugating V_alpha.

        Vertices have shape (..., auxiliary). Result has shape (..., external).
        The caller retains the continuum phases and physical normalization.
        """
        vertices = np.asarray(potential_vertices, dtype=complex)
        if (
            vertices.ndim < 1
            or vertices.shape[-1] != len(self.effective_potential)
            or not np.all(np.isfinite(vertices))
        ):
            raise ValueError("Vertices must end in the potential-basis dimension")
        return vertices @ self.effective_potential


def screen_response(susceptibility, kernel, external_potential):
    """Solve (I-K chi) V_eff=V_ext for every external field together.

    Unlike iterated screening, the direct solve has no iteration tolerance and
    never substitutes a shifted frequency at a response pole. Singular systems
    raise the underlying linear-algebra exception.
    """
    chi = _finite_matrix(susceptibility, "susceptibility")
    k = _finite_matrix(kernel, "kernel")
    ext = _finite_matrix(external_potential, "external potential")
    n = len(chi)
    if not n or chi.shape != (n, n) or k.shape != (n, n) or ext.shape[0] != n:
        raise ValueError(
            "Response/kernel must be equal nonempty squares; fields need matching rows"
        )
    if not np.iscomplexobj(k) and np.iscomplexobj(chi):
        # A real adiabatic kernel needs two real BLAS products, not a general
        # complex product with an identically zero imaginary kernel.
        product = k @ chi.real + 1j * (k @ chi.imag)
    else:
        product = k @ chi
    operator = np.eye(n) - product
    effective = solve(operator, ext, check_finite=False)
    residual = operator @ effective - ext
    scale = np.linalg.norm(operator) * np.linalg.norm(effective) + np.linalg.norm(ext)
    error = float(np.linalg.norm(residual) / scale) if scale else 0.0
    return ScreenedResponseResult(effective, chi @ effective, error)


class GalerkinSternheimerResponse:
    """Occupied-projected response in the existing real Galerkin trial space.

    orbital_sources[i] contains <basis|P_alpha|occupied_i>, and occupation[i]
    is its actual electron occupation (no implicit spin factor). The real
    occupied columns must be S-orthonormal stationary eigenvectors of H.

    boundary_self_energy(z) must return the retarded outgoing self-energy for
    Im(z)>=0 and its advanced conjugate for Im(z)<0. It is subtracted from
    z*S-H. Omitting it explicitly describes a finite closed box, not an open
    molecular continuum. The caller must converge that model separately.

    Each projected resolvent is solved with an augmented sparse system imposing
    C.T*S*x=0. All potential columns share one factorization at each frequency
    and occupied orbital. No virtual eigenstates are constructed or truncated.

    For symmetry blocks, occupied columns describe the TARGET projection space,
    whereas orbital_energies/occupations/sources may describe distinct initial
    states. In that case projection_energies_hartree explicitly supplies the
    stationary target eigenvalues. An empty target projector is supported.
    """

    def __init__(
        self,
        hamiltonian,
        overlap,
        occupied,
        orbital_energies,
        occupations,
        orbital_sources,
        *,
        boundary_self_energy=None,
        stationary_tolerance=1e-8,
        projection_energies_hartree=None,
    ):
        h = _real_sparse_matrix(hamiltonian, "Hamiltonian")
        s = _real_sparse_matrix(overlap, "overlap")
        c = np.asarray(occupied)
        eps = np.asarray(orbital_energies, dtype=float)
        occ = np.asarray(occupations, dtype=float)
        sources = np.asarray(orbital_sources)
        n = h.shape[0]
        if n == 0 or h.shape != (n, n) or s.shape != (n, n):
            raise ValueError("H and S must be equal nonempty square matrices")
        if not _symmetric(h) or not _symmetric(s):
            raise ValueError("H and S must be symmetric")
        if (
            np.iscomplexobj(c)
            or c.ndim != 2
            or c.shape[0] != n
            or not np.all(np.isfinite(c))
        ):
            raise ValueError("Occupied orbitals must be finite real columns")
        m = c.shape[1]
        initial_count = len(eps) if eps.ndim == 1 else 0
        if (
            not initial_count
            or eps.shape != (initial_count,)
            or occ.shape != (initial_count,)
            or not np.all(np.isfinite(eps))
            or not np.all(np.isfinite(occ))
            or np.any(occ <= 0)
            or np.any(occ > 2)
        ):
            raise ValueError("Require one finite energy and occupation in (0,2] per orbital")
        if (
            np.iscomplexobj(sources)
            or sources.ndim != 3
            or sources.shape[:2] != (initial_count, n)
            or not sources.shape[2]
            or not np.all(np.isfinite(sources))
        ):
            raise ValueError("Sources must be real (occupied,basis,potential) arrays")
        if not np.isfinite(stationary_tolerance) or stationary_tolerance <= 0:
            raise ValueError("Stationary tolerance must be positive and finite")
        sc = s @ c
        projection_eps = eps if projection_energies_hartree is None else np.asarray(
            projection_energies_hartree, dtype=float)
        if projection_eps.shape != (m,) or not np.all(np.isfinite(projection_eps)):
            raise ValueError('Explicit stationary target projection energies required')
        if not np.allclose(c.T @ sc, np.eye(m), atol=1e-10, rtol=1e-10):
            raise ValueError("Occupied orbitals must be S-orthonormal")
        hc = h @ c
        if np.linalg.norm(hc - sc * projection_eps) > stationary_tolerance * max(1.0, np.linalg.norm(hc)):
            raise ValueError(
                "Occupied orbitals are not stationary eigenvectors of this Hamiltonian"
            )
        self.hamiltonian, self.overlap = h, s
        self.occupied_overlap = csc_matrix(sc)
        self.energies, self.occupations = eps.copy(), occ.copy()
        self.projection_energies = projection_eps.copy()
        self.sources = np.array(sources, dtype=float, copy=True)
        self.boundary_self_energy = boundary_self_energy
        self.last_diagnostics = None

    def _resolvent(self, z, source):
        n = self.hamiltonian.shape[0]
        # Exactly real closed-channel systems use real sparse LU. No small
        # imaginary part is dropped and no damping or pole shift is inserted.
        frequency = z.real if complex(z).imag == 0 else z
        operator = frequency * self.overlap - self.hamiltonian
        constraint_provider = getattr(self.boundary_self_energy, 'constraint', None)
        physical_boundary = None
        if callable(constraint_provider):
            physical_boundary = self.boundary_self_energy.boundary
            constraint = csc_matrix(constraint_provider(z))
            nc = physical_boundary.shape[1]
            if (physical_boundary.shape[0] != n or constraint.shape != (nc, n)
                    or not np.all(np.isfinite(constraint.data))):
                raise ValueError('Boundary constraint must match the trial space')
        elif self.boundary_self_energy is not None:
            sigma = csc_matrix(self.boundary_self_energy(z), dtype=complex)
            if sigma.shape != (n, n) or not np.all(np.isfinite(sigma.data)):
                raise ValueError("Boundary self-energy must be a finite basis-sized matrix")
            if not np.any(sigma.data.imag):
                sigma = sigma.real
            operator = operator - sigma
        coupling = self.occupied_overlap
        augmented = (bmat([[operator, coupling], [coupling.T, None]], format="csc")
                     if coupling.shape[1] else operator.tocsc())
        if physical_boundary is not None:
            if coupling.shape[1]:
                augmented = bmat([[operator, coupling, physical_boundary],
                                  [coupling.T, None, csc_matrix((coupling.shape[1], nc))],
                                  [constraint, csc_matrix((nc, coupling.shape[1])), None]], format='csc')
            else:
                augmented = bmat([[operator, physical_boundary], [constraint, None]], format='csc')
        factor = splu(augmented)
        rhs = np.zeros((augmented.shape[0], source.shape[1]), dtype=augmented.dtype)
        rhs[:n] = source
        full = factor.solve(rhs)
        scale = np.linalg.norm(augmented.data) * np.linalg.norm(full) + np.linalg.norm(rhs)
        error = np.linalg.norm(augmented @ full - rhs) / scale if scale else 0.0
        return full[:n], float(error), factor.nnz

    def evaluate(self, frequency_hartree):
        """chi(omega)=sum_i n_i M_i.T [G(e_i+w)+G(e_i-w)] M_i.

        A supplied positive Im(omega) is an explicit damping/model parameter;
        zero damping is preserved, including genuine singular response poles.
        """
        omega = complex(frequency_hartree)
        if not np.isfinite(omega) or omega.real < 0 or omega.imag < 0:
            raise ValueError("Require finite nonnegative retarded frequency and damping")
        p = self.sources.shape[2]
        response = np.zeros((p, p), complex)
        errors, fills = [], []
        energies = np.unique(self.energies)
        for energy in energies:
            indices = np.flatnonzero(self.energies == energy)
            # Exactly degenerate orbitals have the same projected operator.
            # Batch their distinct sources; preserve individual occupations.
            sources = self.sources[indices].transpose(1, 0, 2).reshape(
                self.sources.shape[1], -1
            )
            plus, errp, fillp = self._resolvent(energy + omega, sources)
            if omega == 0:
                minus, errm, fillm = plus, errp, fillp
            else:
                minus, errm, fillm = self._resolvent(energy - omega, sources)
            for block, index in enumerate(indices):
                sl = slice(block * p, (block + 1) * p)
                response += self.occupations[index] * (
                    self.sources[index].T @ (plus[:, sl] + minus[:, sl])
                )
            errors.extend((errp, errm))
            fills.extend((fillp, fillm))
        self.last_diagnostics = dict(
            maximum_backward_error=max(errors),
            maximum_factor_nonzeros=max(fills),
            factorizations=len(energies) * (1 if omega == 0 else 2),
            boundary="explicit self-energy" if self.boundary_self_energy else "closed box",
        )
        return response


def kernel_from_auxiliary_integrals(overlap, coulomb_integrals, xc_integrals):
    """Map accumulated real auxiliary integrals to the dual response convention."""
    from scipy.linalg import cho_factor, cho_solve
    matrices = [np.asarray(value) for value in (overlap,coulomb_integrals,xc_integrals)]
    metric,j,f = matrices
    if (metric.ndim != 2 or metric.shape[0] == 0 or metric.shape[0] != metric.shape[1]
            or any(value.shape != metric.shape or np.iscomplexobj(value)
                   or not np.all(np.isfinite(value))
                   or not np.allclose(value,value.T,rtol=1e-12,atol=1e-13) for value in matrices)):
        raise ValueError('Finite real symmetric auxiliary metric/Coulomb/XC integrals required')
    factor=cho_factor(metric,lower=True,check_finite=False)
    left=cho_solve(factor,j+f,check_finite=False)
    return cho_solve(factor,left.T,check_finite=False).T


def project_adiabatic_kernel(coulomb_integrals, auxiliary_values, weights, xc_density_derivative):
    """Represent Hartree+local XC in a common nonorthogonal auxiliary basis.

    Values are (quadrature, auxiliary), weights are physical volume weights,
    and xc_density_derivative is the explicit local d(v_xc)/d(rho). The caller
    supplies Coulomb integrals, computed analytically or by a separately checked
    Poisson solver. Point-charge self-interactions are never guessed here.

    chi maps potential coefficients to <phi|delta rho> moments. Accordingly,
    this returns M^-1*(J+integral phi*f_xc*phi)*M^-1, with M the auxiliary
    overlap. There is no threshold, rank deletion, or density regularization.
    """
    values = np.asarray(auxiliary_values)
    w = np.asarray(weights)
    fxc = np.asarray(xc_density_derivative)
    j = np.asarray(coulomb_integrals)
    if (
        any(np.iscomplexobj(x) for x in (values, w, fxc, j))
        or values.ndim != 2
        or not values.shape[0]
        or not values.shape[1]
        or w.shape != (len(values),)
        or fxc.shape != w.shape
        or j.shape != (values.shape[1], values.shape[1])
        or any(not np.all(np.isfinite(x)) for x in (values, w, fxc, j))
        or np.any(w <= 0)
    ):
        raise ValueError(
            "Require finite real basis, Coulomb matrix, positive weights and local XC kernel"
        )
    if not np.allclose(j, j.T, rtol=1e-12, atol=1e-13):
        raise ValueError("Coulomb integrals must be symmetric")
    metric = values.T @ (w[:, None] * values)
    xc = values.T @ ((w * fxc)[:, None] * values)
    return kernel_from_auxiliary_integrals(metric,j,xc)


def orbital_potential_sources(
    basis_values, occupied_values, potential_values, weights, *, block_points=1024
):
    """Stream <basis|P_alpha|occupied_i> on a shared physical quadrature.

    Large temporary (points,basis,potential) products are avoided. Output has
    shape (occupied,basis,potential), matching GalerkinSternheimerResponse.
    Values must share the same molecular frame and quadrature; no AO ordering,
    orbital phase or volume factor is inferred from array dimensions.
    """
    arrays = [np.asarray(x) for x in (basis_values, occupied_values, potential_values)]
    w = np.asarray(weights)
    if (
        any(
            np.iscomplexobj(x) or x.ndim != 2 or not x.shape[1] or not np.all(np.isfinite(x))
            for x in arrays
        )
        or w.ndim != 1
        or not len(w)
        or np.iscomplexobj(w)
        or not np.all(np.isfinite(w))
        or np.any(w <= 0)
        or any(x.shape[0] != len(w) for x in arrays)
        or not isinstance(block_points, (int, np.integer))
        or block_points <= 0
    ):
        raise ValueError(
            "Require real matrices on one positive-weight quadrature and positive block size"
        )
    basis, occupied, potential = arrays
    output = np.zeros((occupied.shape[1], basis.shape[1], potential.shape[1]))
    for start in range(0, len(w), block_points):
        sl = slice(start, start + block_points)
        for i in range(occupied.shape[1]):
            output[i] += basis[sl].T @ ((w[sl] * occupied[sl, i])[:, None] * potential[sl])
    return output


class SpectralSternheimerResponse:
    """Complete virtual spectral resolvent with channel-size outgoing solves.

    The occupied space must span the lowest occupied eigenstates of the supplied
    real local H/S. No virtual eigenpairs are discarded. A non-Aufbau or otherwise
    incompatible occupied subspace is rejected; GalerkinSternheimerResponse is
    available for explicitly different models.

    boundary must expose its real Galerkin boundary columns and
    log_derivative(z), as CoulombResponseBoundary does. Dense preparation is
    explicit and measured separately; long photon scans amortize it. At a
    virtual inner-region eigenvalue the original sparse projected solve is used
    without moving the energy or deleting that state.
    """

    def __init__(
        self,
        hamiltonian,
        overlap,
        occupied,
        orbital_energies,
        occupations,
        orbital_sources,
        *,
        boundary=None,
        projection_energies_hartree=None,
    ):
        from scipy.linalg import eigh
        from time import perf_counter

        start = perf_counter()
        self._sparse = GalerkinSternheimerResponse(
            hamiltonian,
            overlap,
            occupied,
            orbital_energies,
            occupations,
            orbital_sources,
            boundary_self_energy=boundary,
            projection_energies_hartree=projection_energies_hartree,
        )
        h, s = self._sparse.hamiltonian, self._sparse.overlap
        energies, states = eigh(h.toarray(), s.toarray(), check_finite=False)
        nocc = np.asarray(occupied).shape[1]
        virtual = states[:, nocc:]
        if np.linalg.norm(virtual.T @ (s @ np.asarray(occupied))) > 1e-8:
            raise ValueError("Spectral response requires the complete lowest occupied subspace")
        self.virtual_energies = energies[nocc:]
        self.sources = np.asarray([virtual.T @ source for source in self._sparse.sources])
        self.boundary = boundary
        self.projected_boundary = None
        self.virtual_vectors = None
        if boundary is not None:
            constrained = callable(getattr(boundary, 'constraint', None))
            if not constrained and not callable(getattr(boundary, "log_derivative", None)):
                raise ValueError("Spectral response requires an explicit boundary log_derivative")
            if boundary.boundary.shape[0] != h.shape[0]:
                raise ValueError("Boundary and inner trial space differ")
            self.projected_boundary = virtual.T @ boundary.boundary
            if constrained:
                self.virtual_vectors = virtual
        self.preparation_seconds = perf_counter() - start
        self.last_diagnostics = None

    def _contract(self, z, source):
        # Before boundary dressing a real frequency has an exactly real
        # diagonal inner resolvent, including at open continuum energies.
        # Keep the large virtual-space products in real BLAS; the small
        # outgoing channel correction still retains its full complex phase.
        frequency = z.real if complex(z).imag == 0 else z
        gap = frequency - self.virtual_energies
        inverse = 1 / gap
        free = inverse[:, None] * source
        value = source.T @ free
        if self.boundary is None:
            return value, 0.0
        if self.virtual_vectors is not None:
            constraint = self.boundary.constraint(z) @ self.virtual_vectors
            weighted = inverse[:, None] * self.projected_boundary
            operator = constraint @ weighted
            rhs = constraint @ free
            answer = solve(operator, rhs, check_finite=False)
            coupling = self.projected_boundary.T @ free
            value = value - coupling.T @ answer
            scale = np.linalg.norm(operator)*np.linalg.norm(answer) + np.linalg.norm(rhs)
            error = np.linalg.norm(operator@answer-rhs)/scale if scale else 0.
            return value, float(error)
        q = np.asarray(self.boundary.log_derivative(z), complex)
        w = self.projected_boundary
        if q.shape != (w.shape[1], w.shape[1]) or not np.all(np.isfinite(q)):
            raise ValueError("Invalid channel logarithmic derivative")
        weighted = inverse[:, None] * w
        operator = np.eye(w.shape[1]) + 0.5 * q @ (w.T @ weighted)
        coupling = w.T @ free
        rhs = q @ coupling
        answer = solve(operator, rhs, check_finite=False)
        # Real source/boundary columns and a diagonal complex resolvent imply
        # source.T @ weighted == coupling.T, without complex conjugation.
        value = value - 0.5 * coupling.T @ answer
        scale = np.linalg.norm(operator) * np.linalg.norm(answer) + np.linalg.norm(rhs)
        error = np.linalg.norm(operator @ answer - rhs) / scale if scale else 0.0
        return value, float(error)

    def evaluate(self, frequency_hartree):
        omega = complex(frequency_hartree)
        if not np.isfinite(omega) or omega.real < 0 or omega.imag < 0:
            raise ValueError("Require finite nonnegative retarded frequency and damping")
        arguments = [energy + sign * omega for energy in self._sparse.energies for sign in (1, -1)]
        for z in arguments:
            scale = max(1.0, abs(z), float(np.max(abs(self.virtual_energies), initial=0.0)))
            if (
                np.min(abs(z - self.virtual_energies), initial=np.inf)
                <= 128 * np.finfo(float).eps * scale
            ):
                result = self._sparse.evaluate(omega)
                self.last_diagnostics = dict(
                    self._sparse.last_diagnostics, method="sparse inner-pole fallback"
                )
                return result
        p = self.sources.shape[-1]
        chi = np.zeros((p, p), complex)
        errors = []
        for energy, occupation, source in zip(
            self._sparse.energies, self._sparse.occupations, self.sources
        ):
            plus, ep = self._contract(energy + omega, source)
            if omega == 0:
                minus, em = plus, ep
            else:
                minus, em = self._contract(energy - omega, source)
            chi += occupation * (plus + minus)
            errors.extend((ep, em))
        self.last_diagnostics = dict(
            method="complete virtual spectrum",
            virtual_states=len(self.virtual_energies),
            maximum_channel_backward_error=max(errors),
            preparation_seconds=self.preparation_seconds,
        )
        return chi
