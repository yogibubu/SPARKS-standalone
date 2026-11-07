"""Exact polynomial-operator bridge from continuum amplitudes to SATI/TRINITY."""

from dataclasses import dataclass

import numpy as np


def cached_harmonic_overlap_matrix(model, initial_basis, final_basis, *, cache_directory,
        maximum_grid_entries=5000000, maximum_harmonic_pairs=2000000):
    """Reuse exact nuclear overlaps by state label, independently of the continuum.

    Frequencies, the complete mode map, displacement and initial states identify
    each file. Missing final states use the existing TRINITY overlap owner.
    Electronic operators, photon energies and line widths do not affect this
    identity. Selection and completeness tests remain the caller's responsibility.
    Writes replace the saved file atomically; altered contents are rejected.
    """
    import hashlib
    import json
    import os
    from pathlib import Path
    import tempfile
    from matrix_trinity.vibronic_variational import harmonic_overlap_matrix

    n = len(model.initial_frequencies_cm)
    bases = []
    for value in (initial_basis, final_basis):
        value = np.asarray(value)
        if (value.ndim != 2 or not len(value) or value.shape[1] != n
                or value.dtype.kind not in 'iu' or np.any(value < 0)
                or np.any(value > np.iinfo(np.int32).max)):
            raise ValueError('Compatible nonnegative integer harmonic basis rows required')
        bases.append(value.astype(np.int64))
    initial, final = bases
    for budget in (maximum_grid_entries, maximum_harmonic_pairs):
        if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1:
            raise ValueError('Positive integer harmonic workspace budgets required')
    if len(initial)*len(final) > maximum_harmonic_pairs:
        raise MemoryError('harmonic overlap pair budget exceeded')

    def digest(arrays, prefix):
        h = hashlib.sha256(prefix.encode())
        for array in arrays:
            v = np.ascontiguousarray(array)
            h.update(json.dumps((v.dtype.str, v.shape)).encode())
            h.update(v.tobytes())
        return h.hexdigest()

    key = digest([np.asarray(v, dtype=np.float64) for v in
        (model.initial_frequencies_cm, model.final_frequencies_cm,
         model.mode_map, model.final_shift_dimensionless)] + [initial],
         'sparks.harmonic_nuclear_overlaps.v1')
    folder = Path(cache_directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (key + '.npz')
    stored = np.empty((0, n), dtype=np.int64)
    values = np.empty((0, len(initial)))
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            stored, values = data['final_basis'], data['overlaps']
            valid = (str(data['identity']) == key and
                     str(data['contents_sha256']) == digest([stored, values], key))
        if (not valid or stored.ndim != 2 or stored.shape[1] != n
                or stored.dtype.kind not in 'iu' or np.any(stored < 0)
                or values.shape != (len(stored), len(initial))
                or len({tuple(row) for row in stored}) != len(stored)
                or not np.all(np.isfinite(values))):
            raise ValueError('Incompatible or altered nuclear-overlap cache')
    lookup = {tuple(row): i for i, row in enumerate(stored)}
    missing = list(dict.fromkeys(tuple(row) for row in final if tuple(row) not in lookup))
    if missing:
        new_basis = np.asarray(missing, dtype=np.int64)
        new_values, _ = harmonic_overlap_matrix(model, initial, new_basis,
            maximum_grid_entries=maximum_grid_entries,
            maximum_harmonic_pairs=maximum_harmonic_pairs)
        offset = len(stored)
        lookup.update((row, offset+i) for i, row in enumerate(missing))
        stored = np.vstack((stored, new_basis))
        values = np.vstack((values, new_values))
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(stream, identity=key, final_basis=stored,
                    overlaps=values, contents_sha256=digest([stored, values], key))
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    result = values[[lookup[tuple(row)] for row in final]]
    missing_set = set(missing)
    return result, dict(cache_file=str(path), nuclear_identity=key,
        computed_final_states=len(missing), requested_final_states=len(final),
        reused_final_states=sum(tuple(row) not in missing_set for row in final),
        full_duschinsky_retained=True, amplitude_pruning_used=False)


@dataclass(frozen=True)
class ContinuumOperatorBank:
    constants: np.ndarray
    gradients: np.ndarray
    forms: np.ndarray
    labels: tuple[str, ...] = ("FC", "HT", "interference", "FCHT")


def continuum_fc_ht_operator_bank(amplitudes, derivatives):
    """Sum all Cartesian/continuum channels through (1,q1,...,qn).

    Amplitudes have shape (3,channels); derivatives add a final initial-normal-
    mode axis, in dimensionless coordinates. Forms follow TRINITY's convention
    O_a W_ab O_b*. This is exact algebra for a linear HT operator, with no rank
    threshold or channel selection. Photon, orientation and spin factors are
    not added: returned strengths sum the supplied electronic channels.
    """
    c = np.asarray(amplitudes, complex)
    h = np.asarray(derivatives, complex)
    if (
        c.ndim != 2
        or c.shape[0] != 3
        or not c.shape[1]
        or h.ndim != 3
        or h.shape[:2] != c.shape
        or not h.shape[2]
        or not np.all(np.isfinite(c))
        or not np.all(np.isfinite(h))
    ):
        raise ValueError(
            "Require finite (3,channels) amplitudes and (3,channels,modes) derivatives"
        )
    n = h.shape[2]
    coefficients = np.column_stack((c.reshape(-1), h.reshape(-1, n)))
    gram = coefficients.T @ coefficients.conj()
    forms = np.zeros((4, n + 1, n + 1), complex)
    forms[0, 0, 0] = gram[0, 0]
    forms[1, 1:, 1:] = gram[1:, 1:]
    forms[2, 0, 1:] = gram[0, 1:]
    forms[2, 1:, 0] = gram[1:, 0]
    forms[3] = forms[0] + forms[1] + forms[2]
    constants = np.zeros(n + 1)
    constants[0] = 1.0
    gradients = np.vstack((np.zeros((1, n)), np.eye(n)))
    return ContinuumOperatorBank(constants, gradients, forms)


def harmonic_photoionization_spectrum(
    model, amplitudes, derivatives, *, electron_energies_hartree, **controls
):
    """Dress energy-resolved continuum amplitudes with native harmonic FC/HT.

    Input arrays have shape (energy,3,channels) and (energy,3,channels,modes).
    The caller transports gradients to the model's initial normal coordinates.
    TRINITY owns all nuclear TI/TD algorithms; this wrapper supplies exact small
    Hermitian forms and reshapes the observable axis to (energy,FC/HT/...).
    Returned spectra are dipole strengths at fixed outgoing-electron energy,
    with no photon, spin or orientation prefactors inserted.
    """
    from matrix_trinity.vibronic_observables import harmonic_operator_spectrum

    c = np.asarray(amplitudes, complex)
    h = np.asarray(derivatives, complex)
    energies = np.asarray(electron_energies_hartree, float)
    n = len(model.initial_frequencies_cm)
    if (
        c.ndim != 3
        or c.shape[1] != 3
        or h.shape != c.shape + (n,)
        or energies.shape != (len(c),)
        or not np.all(np.isfinite(energies))
        or np.any(energies <= 0)
    ):
        raise ValueError("Require matching energy-resolved amplitudes and initial-normal gradients")
    banks = [continuum_fc_ht_operator_bank(ci, hi) for ci, hi in zip(c, h, strict=True)]
    if not banks:
        raise ValueError("At least one electron energy is required")
    result = harmonic_operator_spectrum(
        model,
        banks[0].constants,
        np.concatenate([b.forms for b in banks]),
        gradients=banks[0].gradients,
        **controls,
    )
    result["nuclear_schema"] = result["schema"]
    result["schema"] = "matrix.sparks.harmonic_photoionization.v1"
    for key in ("strengths", "densities_per_cm", "correlations", "sum_rules"):
        if key in result:
            value = np.asarray(result[key])
            result[key] = value.reshape((len(c), 4, *value.shape[1:]))
    result.update(
        electron_energies_hartree=energies,
        operator_components=banks[0].labels,
        cartesian_sources=3,
        continuum_channels=c.shape[2],
        polynomial_operator_count=n + 1,
        strength_convention="sum of supplied Cartesian and continuum-channel dipole strengths; fixed electron kinetic energy",
    )
    return result


def vpt2_photoionization_spectrum(model, initial_field, final_field, amplitudes,
                                 derivatives, *, electron_energies_hartree, **controls):
    """Contract the same continuum bank with native two-surface SAM VPT2.

    Mechanical strengths are strictly truncated at order two. No global QFF
    confinement or anharmonic completeness is inferred from selected targets.
    """
    from matrix_trinity.vibronic_vpt2_observables import vpt2_operator_spectrum
    c = np.asarray(amplitudes, complex)
    h = np.asarray(derivatives, complex)
    energies = np.asarray(electron_energies_hartree, float)
    n = len(model.initial_frequencies_cm)
    if (c.ndim != 3 or c.shape[1] != 3 or h.shape != c.shape+(n,)
            or energies.shape != (len(c),) or not len(c)
            or not np.all(np.isfinite(energies)) or np.any(energies <= 0)):
        raise ValueError('Require matching positive energies and normal-coordinate amplitudes')
    banks = [continuum_fc_ht_operator_bank(ci, hi) for ci, hi in zip(c, h, strict=True)]
    result = vpt2_operator_spectrum(model, initial_field, final_field,
        banks[0].constants, np.concatenate([b.forms for b in banks]),
        gradients=banks[0].gradients, **controls)
    result['nuclear_schema'] = result['schema']
    result['schema'] = 'matrix.sparks.vpt2_photoionization.v1'
    for key in ('strength_orders', 'strengths', 'weighted_strengths', 'completed_strengths'):
        value = np.asarray(result[key])
        result[key] = value.reshape((len(c), 4, *value.shape[1:]))
    result.update(electron_energies_hartree=energies,
                  operator_components=banks[0].labels,
                  strength_convention='conditional dipole strengths at fixed electron kinetic energy; strict mechanical order two')
    return result
