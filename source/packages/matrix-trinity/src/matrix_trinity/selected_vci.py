"""Selected VCI restrictions and complete external residuals of one Hamiltonian.

Reuses the canonical modal operator compiler. No perturbative self energy is
added to retained matrices. The denominator floor affects selection only;
resonant external partners remain visible and must enter the retained space.
"""
from collections import OrderedDict
import numpy as np
from .targeted_polyads import CubicDressedWorkspace
from .curvilinear_vpt2 import _local_operator_transitions
from .vci_operator_action import CompiledModalAction


class SelectedVCIWorkspace(CubicDressedWorkspace):
    """H0+V1+V3+V4 with complete coherent Q H P residual selection.

    All supplied kinetic and scalar terms in the modal compiler are retained.
    An explicitly reduced source polynomial remains a reduced physical model.
    Residual estimates are diagnostics, not rigorous eigenvalue error bounds.
    """
    def __init__(self, field, *, selection_gap_floor_cm=1., **kwargs):
        super().__init__(field, **kwargs)
        if not np.isfinite(selection_gap_floor_cm) or selection_gap_floor_cm <= 0:
            raise ValueError('positive finite selection denominator floor required')
        self.selection_gap_floor_cm = float(selection_gap_floor_cm)
        self.terms = self.cubic_terms + self.quartic_terms
        self.compiled_action = CompiledModalAction(self.terms)
        self.full_cache = OrderedDict()
        self.full_cached_count = 0
        # Odd local strings have zero diagonal matrix element by parity.
        self.diagonal_terms = tuple((key, value) for key, value in self.terms
                                    if all(len(ops) % 2 == 0 for _, ops in key))
        self.last_residual_diagnostics = {}

    def couplings(self, state):
        state = tuple(state)
        if state in self.full_cache:
            self.full_cache.move_to_end(state)
            return self.full_cache[state]
        values = {q: v for q, v in self.compiled_action(state).items() if v != 0}
        while self.full_cache and self.full_cached_count + len(values) > self.maximum_cached_couplings:
            _, old = self.full_cache.popitem(last=False)
            self.full_cached_count -= len(old)
        if len(values) <= self.maximum_cached_couplings:
            self.full_cache[state] = values
            self.full_cached_count += len(values)
        return values

    def diagonal(self, states):
        rows = np.asarray(states, int)
        result = rows @ self.frequencies + self.pseudopotential_cm
        result = result.astype(complex)
        for key, coefficient in self.diagonal_terms:
            factor = np.full(len(rows), coefficient, dtype=complex)
            for mode, ops in key:
                occupations = rows[:, mode]
                lookup = np.array([dict(_local_operator_transitions(ops, n)).get(n, 0j)
                                   for n in range(int(occupations.max()) + 1)])
                factor *= lookup[occupations]
            result += factor
        if np.max(abs(result.imag), initial=0) > 1e-9:
            raise ValueError('nonreal Hamiltonian diagonal')
        return result.real

    def matrix(self, states):
        states = tuple(map(tuple, states))
        indices = {state: i for i, state in enumerate(states)}
        if len(indices) != len(states):
            raise ValueError('distinct VCI occupations required')
        matrix = np.diag(np.asarray(states) @ self.frequencies + self.pseudopotential_cm).astype(complex)
        for i, state in enumerate(states):
            for target, value in self.couplings(state).items():
                j = indices.get(target)
                if j is not None:
                    matrix[j, i] += value
        if np.max(abs(matrix - matrix.conj().T), initial=0) > 1e-8:
            raise ValueError('compiled VCI Hamiltonian must be Hermitian')
        return matrix

    def external_corrections(self, states, columns, eigenvalues):
        members = set(map(tuple, states))
        columns = np.asarray(columns, complex)
        eigenvalues = np.asarray(eigenvalues, float)
        if columns.shape != (len(states), len(eigenvalues)):
            raise ValueError('compatible retained eigenvectors and energies required')
        residual = {}
        for i, state in enumerate(states):
            for target, value in self.couplings(state).items():
                if target not in members:
                    if target not in residual:
                        residual[target] = np.zeros(len(eigenvalues), complex)
                    residual[target] += value * columns[i]
        if not residual:
            self.last_residual_diagnostics = dict(norm_cm=[0.] * len(eigenvalues), external_states=0)
            return {}
        external = tuple(residual)
        values = np.asarray([residual[q] for q in external])
        gaps = eigenvalues[None, :] - self.diagonal(external)[:, None]
        near = abs(gaps) < self.selection_gap_floor_cm
        safe = np.where(near, np.where(gaps < 0, -1., 1.) * self.selection_gap_floor_cm, gaps)
        self.last_residual_diagnostics = dict(
            norm_cm=np.linalg.norm(values, axis=0).tolist(), external_states=len(external),
            near_denominator_entries=int(np.count_nonzero(near & (values != 0))),
            selection_gap_floor_cm=self.selection_gap_floor_cm,
            full_external_quartics=True, retained_hamiltonian_dressed=False)
        return {q: value for q, value in zip(external, values / safe) if np.any(value)}

    def lifted_wavefunctions(self, states, columns, _eigenvalues):
        """The retained normalized VCI vectors; no unselected PT2 tails added."""
        return {tuple(state): row for state, row in zip(states, np.asarray(columns, complex))}
