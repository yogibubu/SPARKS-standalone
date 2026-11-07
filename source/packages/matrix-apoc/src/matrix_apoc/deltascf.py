"""Continuation of independently optimized single determinants.

No TD amplitudes or optical transition densities are inferred. Overlaps refer
only to real integer-occupation determinants, with explicit alpha/beta sectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .state import ElectronicState, StateMatch, StateContinuityError, _maximum_overlap_assignment


@dataclass(frozen=True)
class DeterminantOrbitals:
    """AO determinant contract also supporting explicitly frozen valence cores.

    ``core_electrons`` counts excluded electrons on each physical nucleus; it
    must never be represented by changing nuclear charges or molecular charge.
    No transition properties are implied by this valence-only descriptor.
    """
    atomic_numbers: np.ndarray
    coordinates_bohr: np.ndarray
    overlap_ao: np.ndarray
    mo_coefficients: tuple[np.ndarray, ...]
    mo_occupations: tuple[np.ndarray, ...]
    core_electrons: np.ndarray
    charge: int = 0
    multiplicity: int = 1
    total_energy_hartree: float | None = None
    model: str = ""

    def __post_init__(self):
        z = np.asarray(self.atomic_numbers, dtype=int)
        core = np.asarray(self.core_electrons, dtype=int)
        xyz = np.asarray(self.coordinates_bohr, dtype=float)
        metric = np.asarray(self.overlap_ao, dtype=float)
        if (not np.array_equal(z, self.atomic_numbers)
                or not np.array_equal(core, self.core_electrons)
                or z.ndim != 1 or not len(z) or core.shape != z.shape
                or np.any(z <= 0) or np.any(core < 0) or np.any(core > z)
                or xyz.shape != (len(z), 3) or not np.all(np.isfinite(xyz))
                or metric.ndim != 2 or metric.shape[0] != metric.shape[1]
                or not np.all(np.isfinite(metric))
                or not np.allclose(metric, metric.T, atol=1e-12, rtol=0)
                or len(self.mo_coefficients) != 2 or len(self.mo_occupations) != 2
                or self.multiplicity < 1 or not self.model.strip()):
            raise ValueError("invalid valence determinant metadata")
        np.linalg.cholesky(metric)
        for c, occ in zip(self.mo_coefficients, self.mo_occupations, strict=True):
            if np.asarray(c).ndim != 2 or np.asarray(c).shape != (len(metric), len(occ)):
                raise ValueError("incompatible valence determinant orbitals")
        counts = [np.sum(o) for o in self.mo_occupations]
        if not np.isclose(sum(counts), np.sum(z-core)-self.charge, atol=1e-8, rtol=0):
            raise ValueError("valence determinant electron count is inconsistent")
        if not np.isclose(counts[0]-counts[1], self.multiplicity-1, atol=1e-8, rtol=0):
            raise ValueError("valence determinant spin counts are inconsistent")
        for name, value in (("atomic_numbers", z), ("core_electrons", core),
                            ("coordinates_bohr", xyz), ("overlap_ao", metric)):
            array = np.array(value, copy=True); array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("mo_coefficients", "mo_occupations"):
            arrays = tuple(np.array(v, copy=True) for v in getattr(self, name))
            for a in arrays:
                a.setflags(write=False)
            object.__setattr__(self, name, arrays)
        _occupied_channels(self)

    @property
    def spin_channels(self):
        return 2

    @property
    def nao(self):
        return len(self.overlap_ao)


@dataclass(frozen=True)
class DeltaSCFState:
    state_id: int
    wavefunction: ElectronicState | DeterminantOrbitals

    def __post_init__(self):
        if isinstance(self.state_id, bool) or int(self.state_id) != self.state_id or self.state_id < 1:
            raise ValueError("DeltaSCF state identifiers must be positive integers")
        _occupied_channels(self.wavefunction)


def _occupied_channels(state):
    channels = []
    for coefficients, occupations in zip(state.mo_coefficients, state.mo_occupations, strict=True):
        c = np.asarray(coefficients)
        if np.iscomplexobj(c) and np.any(np.abs(c.imag) > 1.e-12):
            raise ValueError("DeltaSCF tracking currently requires real orbitals")
        c = np.asarray(c.real, dtype=float)
        occ = np.asarray(occupations)
        limit = 2 if state.spin_channels == 1 else 1
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(occ)):
            raise ValueError("DeltaSCF orbitals and occupations must be finite")
        if not np.all(np.isclose(occ, 0, atol=1.e-8, rtol=0) | np.isclose(occ, limit, atol=1.e-8, rtol=0)):
            raise ValueError("DeltaSCF requires integer spin occupations; split open-shell orbitals into alpha/beta")
        occupied = c[:, occ > 0.5]
        if not np.allclose(occupied.T @ state.overlap_ao @ occupied, np.eye(occupied.shape[1]), atol=2.e-6, rtol=0):
            raise ValueError("DeltaSCF occupied orbitals must be AO-metric orthonormal")
        channels.append(occupied)
    result = tuple(channels * 2 if len(channels) == 1 else channels)
    if result[0].shape[1] - result[1].shape[1] != state.multiplicity - 1:
        raise ValueError("DeltaSCF occupied spin counts disagree with the nominal spin sector")
    return result


def determinant_overlap(reference: DeltaSCFState, candidate: DeltaSCFState,
                        overlap_reference_candidate: np.ndarray) -> tuple[float, float]:
    """Absolute normalized Slater overlap and real phase (not its square).

    The cross metric must be evaluated between the actual two AO bases. The
    determinant is invariant to occupied orbital rotations and permutations.
    """
    left, right = reference.wavefunction, candidate.wavefunction
    if (left.charge != right.charge or left.multiplicity != right.multiplicity
            or not np.array_equal(left.atomic_numbers, right.atomic_numbers)):
        raise ValueError("DeltaSCF tracking requires the same atoms, charge and spin sector")
    core_left = getattr(left, "core_electrons", np.zeros_like(left.atomic_numbers))
    core_right = getattr(right, "core_electrons", np.zeros_like(right.atomic_numbers))
    if not np.array_equal(core_left, core_right):
        raise ValueError("DeltaSCF frozen-core definitions differ")
    if isinstance(left, DeterminantOrbitals) != isinstance(right, DeterminantOrbitals):
        raise ValueError("cannot mix full and valence determinant representations")
    if isinstance(left, DeterminantOrbitals) and left.model != right.model:
        raise ValueError("DeltaSCF valence models differ")
    cross = np.asarray(overlap_reference_candidate, dtype=float)
    if cross.shape != (left.nao, right.nao) or not np.all(np.isfinite(cross)):
        raise ValueError("invalid cross-AO overlap for DeltaSCF tracking")
    log_score, phase = 0., 1.
    for a, b in zip(_occupied_channels(left), _occupied_channels(right), strict=True):
        if a.shape[1] != b.shape[1]:
            raise ValueError("DeltaSCF alpha/beta electron counts differ")
        block = a.T @ cross @ b
        # A physical cross metric cannot increase the norm of an orbital.
        if block.size and np.linalg.svd(block, compute_uv=False)[0] > 1. + 1.e-5:
            raise ValueError("cross-AO overlap is inconsistent with the occupied metrics")
        sign, logabs = np.linalg.slogdet(block)
        phase *= sign
        log_score += logabs
    return float(min(1., np.exp(log_score))), float(phase if phase else 1.)


def match_deltascf_states(reference_states: Iterable[DeltaSCFState],
                         candidate_states: Iterable[DeltaSCFState],
                         overlap_reference_candidate: np.ndarray, *,
                         minimum_overlap: float = .70,
                         ambiguity_margin: float = .05) -> tuple[StateMatch, ...]:
    """One-to-one assignment by determinant character; energies are metadata."""
    if not (0 <= minimum_overlap <= 1 and 0 <= ambiguity_margin <= 1):
        raise ValueError("tracking thresholds must lie in [0, 1]")
    references, candidates = tuple(reference_states), tuple(candidate_states)
    for states in (references, candidates):
        if len({s.state_id for s in states}) != len(states):
            raise ValueError("DeltaSCF state identifiers must be unique")
    if not references or not candidates:
        return ()
    scores = np.empty((len(references), len(candidates)))
    phases = np.empty_like(scores)
    for i, ref in enumerate(references):
        for j, cand in enumerate(candidates):
            scores[i, j], phases[i, j] = determinant_overlap(ref, cand, overlap_reference_candidate)
    matches = []
    for i, j in _maximum_overlap_assignment(scores):
        runner = float(np.max(np.delete(scores[i], j), initial=0.))
        margin = float(scores[i, j] - runner)
        e0, e1 = references[i].wavefunction.total_energy_hartree, candidates[j].wavefunction.total_energy_hartree
        matches.append(StateMatch(references[i].state_id, candidates[j].state_id,
                                  float(scores[i, j]), float(phases[i, j]),
                                  None if e0 is None or e1 is None else e1 - e0,
                                  runner, margin, bool(scores[i, j] >= minimum_overlap),
                                  bool(margin < ambiguity_margin)))
    return tuple(matches)


def follow_deltascf_state(reference_state, candidate_states, overlap_reference_candidate,
                          *, minimum_overlap=.70, ambiguity_margin=.05, strict=True):
    """Reject a collapsed or ambiguous step before an optimizer accepts it."""
    candidates = tuple(candidate_states)
    matches = match_deltascf_states((reference_state,), candidates, overlap_reference_candidate,
                                   minimum_overlap=minimum_overlap, ambiguity_margin=ambiguity_margin)
    if not matches:
        raise StateContinuityError("no DeltaSCF candidates supplied")
    match = matches[0]
    if strict and (not match.continuous or match.ambiguous):
        raise StateContinuityError(f"DeltaSCF state continuity failed: overlap={match.overlap:.6f}, margin={match.margin:.6f}")
    return next(s for s in candidates if s.state_id == match.candidate_state_id), match


@dataclass(frozen=True)
class SpinAdaptedSinglet:
    """Normalized two-determinant S=0 CSF in alpha-then-beta ordering.

    The components have common spatial orbitals and exactly two open shells.
    Both alpha/beta-block ordered determinants carry coefficient +1/sqrt(2).
    This is a wavefunction descriptor; it does not define a DFT projection energy.
    """
    state_id: int
    mixed: DeltaSCFState
    energy_hartree: float | None = None

    def __post_init__(self):
        w = self.mixed.wavefunction
        if self.state_id < 1 or w.spin_channels != 2 or w.multiplicity != 1:
            raise ValueError('spin-adapted singlet requires two explicit Ms=0 spin channels')
        a,b = w.mo_coefficients
        oa,ob = w.mo_occupations
        if (not np.allclose(a,b,atol=1e-10,rtol=0)
                or np.count_nonzero(oa-ob == 1) != 1
                or np.count_nonzero(ob-oa == 1) != 1
                or not np.allclose(a.T @ w.overlap_ao @ a,np.eye(a.shape[1]),atol=1e-7,rtol=0)):
            raise ValueError('S=0 CSF requires common orthonormal orbitals with exactly two open shells')

    @property
    def components(self):
        from dataclasses import replace
        w = self.mixed.wavefunction
        updates = dict(mo_occupations=w.mo_occupations[::-1])
        if isinstance(w,ElectronicState):
            for name in ('mo_energies_hartree','density_matrices_ao','fock_matrices_ao'):
                updates[name] = getattr(w,name)[::-1]
        partner = replace(w,**updates)
        return (self.mixed,DeltaSCFState(self.mixed.state_id,partner))

    @property
    def spin_squared(self):
        return 0.

    @property
    def spin_densities_ao(self):
        w = self.mixed.wavefunction
        total = sum((c*o) @ c.T for c,o in zip(w.mo_coefficients,w.mo_occupations,strict=True))
        return np.array([.5*total]*2)


def singlet_csf_overlap(reference, candidate, overlap_reference_candidate):
    """Signed coherent determinant sum, including cross terms between CSFs."""
    amplitude = 0.
    for a in reference.components:
        for b in candidate.components:
            score,phase = determinant_overlap(a,b,overlap_reference_candidate)
            amplitude += .5*score*phase
    if abs(amplitude) > 1.+1e-5:
        raise ValueError('singlet CSF overlap exceeds its normalized bound')
    return min(1.,abs(amplitude)), 1. if amplitude >= 0 else -1.


def follow_spin_adapted_singlet(reference, candidates, overlap_reference_candidate, *,
                                 minimum_overlap=.70, ambiguity_margin=.05, strict=True):
    candidates = tuple(candidates)
    if not candidates:
        raise StateContinuityError('no spin-adapted singlet candidates supplied')
    if len({s.state_id for s in candidates}) != len(candidates):
        raise ValueError('singlet candidate identifiers must be unique')
    values = [singlet_csf_overlap(reference,c,overlap_reference_candidate) for c in candidates]
    order = np.argsort([-v[0] for v in values],kind='stable')
    i = order[0];runner = values[order[1]][0] if len(order)>1 else 0.
    score,phase = values[i];margin = score-runner
    e0,e1 = reference.energy_hartree,candidates[i].energy_hartree
    match = StateMatch(reference.state_id,candidates[i].state_id,score,phase,
        None if e0 is None or e1 is None else e1-e0,runner,margin,
        score >= minimum_overlap,margin < ambiguity_margin)
    if strict and (not match.continuous or match.ambiguous):
        raise StateContinuityError(f'singlet CSF continuity failed: overlap={score:.6f}, margin={margin:.6f}')
    return candidates[i],match
