"""Sparse outgoing-boundary solves in the existing Galerkin trial space.

No eigenspectrum, artificial broadening, pole shift or amplitude approximation
is introduced. The same ContinuumBackend/BoundaryWaves contracts are used by
the full spectral backend. A factorization is discarded after each energy.
"""

from time import perf_counter

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import splu

from .spectral import ContinuumResult, _real_matrix


def _real_sparse_matrix(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real for this backend")
    array = csc_matrix(value, dtype=float, copy=True)
    array.sum_duplicates()
    array.eliminate_zeros()
    if not np.all(np.isfinite(array.data)):
        raise ValueError(f"{name} must be finite")
    return array


def _symmetric(matrix):
    # Compare each stored pair, without creating a dense N-by-N array.
    difference = (matrix - matrix.T).tocoo()
    if not difference.nnz:
        return True
    transposed = np.asarray(matrix.T[difference.row, difference.col]).ravel()
    return bool(np.all(abs(difference.data) <= 1e-13 + 1e-12 * abs(transposed)))


class _ColumnOrderedFactor:
    """Expose solves in the original coordinates after fixed column ordering."""
    def __init__(self, factor, order):
        self.factor, self.order = factor, order
        self.nnz = factor.nnz

    def solve(self, rhs, trans="N"):
        if trans == "N":
            ordered = self.factor.solve(rhs)
            result = np.empty_like(ordered)
            result[self.order] = ordered
            return result
        return self.factor.solve(np.asarray(rhs)[self.order], trans=trans)


class SparseResolventBackend:
    """Solve (H-E*S-B*Q*B.T/2) C = B*(F'-Q*F)/2 at each energy.

    H/S/B may already be sparse; D is a small collection of dense sources.
    H/S symmetry is checked without densification. The caller supplies a valid
    positive-definite Galerkin overlap; this backend does not certify its
    smallest eigenvalue by an additional global eigensolve.

    This method avoids complete diagonalization. Its cost depends on sparse LU
    fill and is repeated at every requested energy; the spectral backend can
    remain preferable for a long energy scan. No backend is selected silently.
    The default minimum-degree ordering uses the graph of A+A.T to reduce LU
    fill. It does not symmetrize the outgoing operator or change its entries;
    callers can explicitly request another supported SuperLU permutation.
    The fixed H/S/boundary graph and first column ordering are reused by
    default. Every energy still receives an independent numerical LU with
    fresh row pivoting. Set reuse_structure=False to retain per-energy graph
    construction; large boundary support uses that sparse path automatically.
    An explicit source_transform T represents output sources D @ T. Only the
    supplied independent columns D are solved; all output amplitudes are
    restored by T.T. This exact linear map introduces no rank cutoff.
    """

    def __init__(self, hamiltonian, overlap, boundary, sources, *, permutation="MMD_AT_PLUS_A", reuse_structure=True, source_transform=None):
        if permutation not in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}:
            raise ValueError("Unsupported sparse column permutation")
        self.permutation = permutation
        h = _real_sparse_matrix(hamiltonian, "Hamiltonian")
        s = _real_sparse_matrix(overlap, "overlap")
        b = _real_sparse_matrix(boundary, "boundary")
        d = _real_matrix(sources, "sources")
        n = h.shape[0]
        if n == 0 or h.shape != (n, n) or s.shape != (n, n):
            raise ValueError("Hamiltonian and overlap must be equal nonempty square matrices")
        if b.shape[0] != n or d.shape[0] != n or b.shape[1] == 0:
            raise ValueError("Boundary/source rows must match the inner-region basis")
        for name, matrix in (("Hamiltonian", h), ("overlap", s)):
            if not _symmetric(matrix):
                raise ValueError(f"{name} must be symmetric")
        if np.any(s.diagonal() <= 0):
            raise ValueError("Overlap diagonal must be strictly positive")
        self._hamiltonian, self._overlap = h, s
        self._boundary, self._sources = b, d
        self._source_transform = None
        if source_transform is not None:
            transform = _real_matrix(source_transform,"source transform")
            if transform.shape[0] != d.shape[1]:
                raise ValueError("Source transform rows must match source columns")
            self._source_transform = transform
        self.output_sources = d.shape[1] if self._source_transform is None else self._source_transform.shape[1]
        self.nchannels = b.shape[1]
        self.last_diagnostics = None
        self._column_order = None
        self._structure = None
        self._ordering_reused = False
        if reuse_structure:
            self._prepare_structure()

    def _prepare_structure(self):
        """Reserve H/S and all possible boundary couplings, including exact zeros."""
        h, s, b = self._hamiltonian, self._overlap, self._boundary
        rows = np.unique(b.nonzero()[0])
        # Keep unrestricted full-support boundaries on the existing sparse path;
        # reserving a dense N-by-N graph would defeat the memory advantage.
        if len(rows)**2 > 4_000_000:
            return
        n = h.shape[0]
        boundary_rows, boundary_cols = np.tile(rows,len(rows)), np.repeat(rows,len(rows))
        template=h.copy()
        template.data.fill(1)
        spattern=s.copy()
        spattern.data.fill(1)
        template=template+spattern+coo_matrix(
            (np.ones(len(boundary_rows)),(boundary_rows,boundary_cols)),shape=h.shape).tocsc()
        keys = np.repeat(np.arange(n),np.diff(template.indptr))*n + template.indices
        def slots(matrix):
            coo = matrix.tocoo()
            return np.searchsorted(keys,coo.col*n+coo.row),coo.data
        si,sv=slots(s)
        if (np.array_equal(h.indptr,template.indptr)
                and np.array_equal(h.indices,template.indices)):
            baseline=h.data.astype(complex)
        else:
            hi,hv=slots(h)
            baseline=np.zeros(template.nnz,dtype=complex)
            baseline[hi]=hv
        boundary_slots=np.searchsorted(keys,boundary_cols*n+boundary_rows)
        self._structure = (template,baseline,si,sv,boundary_slots,b[rows].toarray())

    def _assemble_operator(self, energy, q):
        if self._structure is None:
            operator = (self._hamiltonian-energy*self._overlap).astype(complex)
            return (operator-.5*self._boundary@csc_matrix(q)@self._boundary.T).tocsc()
        template,baseline,si,sv,bi,b = self._structure
        data=baseline.copy()
        data[si]-=energy*sv
        data[bi]-=(.5*(b@q@b.T)).ravel(order="F")
        return csc_matrix((data,template.indices,template.indptr),shape=template.shape)

    def _factor(self, operator):
        self._ordering_reused = False
        if self._structure is None or self.permutation == "NATURAL":
            return splu(operator,permc_spec=self.permutation)
        if self._column_order is None:
            factor=splu(operator,permc_spec=self.permutation)
            self._column_order=np.argsort(factor.perm_c)
            # Construct this CSC index map only once, not on each energy.
            mapping=csc_matrix((np.arange(operator.nnz),operator.indices,operator.indptr),
                               shape=operator.shape)[:,self._column_order]
            self._ordered_pattern=(mapping.data.copy(),mapping.indices.copy(),mapping.indptr.copy())
            return factor
        slots,indices,indptr=self._ordered_pattern
        ordered=csc_matrix((operator.data[slots],indices,indptr),shape=operator.shape)
        factor=splu(ordered,permc_spec="NATURAL")
        self._ordering_reused=True
        return _ColumnOrderedFactor(factor,self._column_order)

    def _outgoing_system(self, energy, waves):
        energy = float(energy)
        if not np.isfinite(energy):
            raise ValueError("Energy must be finite")
        values = []
        for value in (waves.regular, waves.irregular, waves.regular_derivative,
                      waves.irregular_derivative):
            value = np.asarray(value)
            if (np.iscomplexobj(value) or not np.all(np.isfinite(value))
                    or value.shape not in ((self.nchannels,), (self.nchannels, self.nchannels))):
                raise ValueError("Boundary waves must be finite real channel vectors or matrices")
            values.append(np.diag(value) if value.ndim == 1 else value)
        f, g, fp, gp = values
        outgoing = g + 1j * f
        q = solve(outgoing.T, (gp + 1j * fp).T, check_finite=False).T
        driving = fp - q @ f
        boundary = self._boundary
        start = perf_counter()
        operator = self._assemble_operator(energy,q)
        assembled = perf_counter()
        factor = self._factor(operator)
        self._phase_seconds = dict(operator_assembly=assembled-start, factorization=perf_counter()-assembled)
        return energy, operator, factor, f, outgoing, driving

    def evaluate(self, energy, waves):
        energy, operator, factor, f, outgoing, driving = self._outgoing_system(energy, waves)
        boundary = self._boundary
        rhs = .5 * (boundary @ driving)
        coefficients = factor.solve(rhs)
        boundary_values = boundary.T @ coefficients
        scattering = np.eye(self.nchannels) + 2j * solve(
            outgoing, boundary_values - f, check_finite=False,
        )
        self._record_diagnostics(operator, factor, coefficients, rhs)
        return ContinuumResult(
            energy, self._expand_sources(self._sources.T @ coefficients), scattering, boundary_values, False,
        )

    def evaluate_amplitudes(self, energy, waves):
        """Return dipoles without constructing scattering states or S matrices.

        For fewer sources than boundary channels, A.T X = D gives
        D.T A^-1 B driving / 2 = X.T B driving / 2. This is a transpose,
        never a conjugate transpose: the complex outgoing phases are retained.
        All supplied sources and channels remain present. The same sparse LU
        factorization and boundary conditions are used by the full evaluation.
        """
        _, operator, factor, _, _, driving = self._outgoing_system(energy, waves)
        started = perf_counter()
        transpose = 0 < self._sources.shape[1] < self.nchannels
        if transpose:
            rhs = np.asarray(self._sources, dtype=complex)
            coefficients = factor.solve(rhs, trans="T")
            amplitudes = .5 * (self._boundary.T @ coefficients).T @ driving
        else:
            rhs = .5 * (self._boundary @ driving)
            coefficients = factor.solve(rhs)
            amplitudes = self._sources.T @ coefficients
        self._phase_seconds['source_solve_and_contraction'] = perf_counter()-started
        self._record_diagnostics(operator, factor, coefficients, rhs, transpose=transpose)
        return self._expand_sources(amplitudes)

    def _expand_sources(self, amplitudes):
        return (amplitudes if self._source_transform is None else
                self._source_transform.T @ amplitudes)

    def _record_diagnostics(self, operator, factor, coefficients, rhs, *, transpose=False):
        residual = (operator.T if transpose else operator) @ coefficients - rhs
        scale = np.linalg.norm(operator.data) * np.linalg.norm(coefficients) + np.linalg.norm(rhs)
        self.last_diagnostics = dict(
            dimension=operator.shape[0], operator_nonzeros=operator.nnz,
            factor_nonzeros=factor.nnz, column_permutation=self.permutation,
            structure_reused=self._structure is not None,
            ordering_reused=self._ordering_reused,
            backward_error=float(np.linalg.norm(residual) / scale) if scale else 0.0,
            solve_right_hand_sides=rhs.shape[1], transpose_solve=transpose,
            independent_source_columns=self._sources.shape[1], output_source_columns=self.output_sources,
            phase_seconds=dict(self._phase_seconds),
        )
