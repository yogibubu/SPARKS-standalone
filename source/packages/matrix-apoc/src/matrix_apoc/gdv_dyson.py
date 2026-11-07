"""Import complete GDV EPT diagnostic vectors; rounded weights are insufficient."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dyson import DysonOrbital


def locate_saved_ept_vector(path, active_components, *, ionization_energy_ev,
                            printed_pole_strength, block_words=8_000_000):
    """Locate a unique signed saved vector using native pole/norm evidence.

    This is recovery from a retained little-endian RWF, not a new electronic
    calculation. Identical repeated records are allowed; distinct matching
    vectors require explicit inspection. Canonical MO mapping and AO norm
    still need independent checks before exporting an orbital.
    """
    if (type(active_components) is not int or active_components < 1
            or type(block_words) is not int or block_words < active_components+1
            or not np.isfinite([ionization_energy_ev, printed_pole_strength]).all()
            or min(ionization_energy_ev, printed_pole_strength) <= 0):
        raise ValueError("Positive dimensions and native pole/norm required")
    path = Path(path)
    if path.stat().st_size % 8:
        raise ValueError("Saved RWF must contain complete eight-byte words")
    data = np.memmap(path, dtype='<f8', mode='r')
    length = active_components+1
    offsets, candidate = [], None
    for start in range(0, len(data)-length+1, block_words):
        stop = min(start+block_words, len(data)-length+1)
        indices = np.flatnonzero(abs(data[start:stop]+ionization_energy_ev) <= 5.1e-7)+start
        for offset in indices:
            vector = np.array(data[offset:offset+length])
            if not np.isfinite(vector).all():
                continue
            norm = float(vector[1:] @ vector[1:])
            if abs(norm-printed_pole_strength) > 5.1e-5:
                continue
            if candidate is not None and not np.allclose(vector, candidate, atol=1e-12, rtol=1e-12):
                raise ValueError("Multiple distinct saved EPT vectors match the printed pole; inspect explicitly")
            candidate = vector
            offsets.append(int(offset))
    if not offsets:
        raise ValueError("No complete saved EPT vector matches the native pole and norm")
    return offsets[0], tuple(offsets)


def write_ept_dyson_mo_export(path, state, coefficients_mo, *, frozen_core,
                             ionization_energy_ev, pole_strength, symmetry, root,
                             provenance):
    """Write complete saved EPT components in the existing checked interchange.

    The caller must recover the producer's signed vector and canonical MO map;
    rounded component weights cannot supply these data. Coefficients cover the
    complete same-run MO space, including explicit zero frozen-core entries.
    The AO-metric norm is checked before writing; no rescaling is performed.
    Per-orbital symmetry is marked zero (unavailable), independently of the
    explicit electronic channel symmetry in the header.
    """
    import json

    if (len(state.mo_coefficients) != 1 or not state.mo_energies_hartree
            or type(frozen_core) is not int or frozen_core < 0
            or type(symmetry) is not int or symmetry < 1
            or type(root) is not int or root < 1 or not provenance):
        raise ValueError('Require restricted same-run orbitals and explicit EPT identities')
    c = np.asarray(coefficients_mo)
    nmo = state.mo_coefficients[0].shape[1]
    if (c.shape != (nmo,) or np.iscomplexobj(c) or frozen_core >= nmo
            or np.any(c[:frozen_core] != 0)):
        raise ValueError('Complete real MO coefficients with zero frozen-core entries required')
    occupations = state.mo_occupations[0]
    if np.any((occupations != 0) & (occupations != 2)):
        raise ValueError('Restricted integer occupations required for EPT export')
    no = int(np.count_nonzero(occupations[frozen_core:] == 2))
    if (np.any(occupations[:frozen_core] != 2)
            or np.any(occupations[frozen_core:frozen_core + no] != 2)
            or np.any(occupations[frozen_core + no:] != 0)):
        raise ValueError('Require canonical occupied-before-virtual MO ordering')
    target = Path(path)
    DysonOrbital.from_mo(state, c, pole_strength=pole_strength,
                        ionization_energy_ev=ionization_energy_ev,
                        channel=f'{symmetry}:{root}', method='EPT saved-vector export',
                        source=str(target))
    count = nmo - frozen_core
    rows = ['# Saved signed-vector provenance: ' + json.dumps(provenance, sort_keys=True),
            f'SPARKS_EPT_DYSON_V1 {symmetry} {root} {count} {no} {count-no}',
            f'SPARKS_EPT_POLE {-ionization_energy_ev:.17e} {pole_strength:.17e}']
    energies = state.mo_energies_hartree[0]
    for i in range(count):
        j = i + frozen_core
        rows.append(f'SPARKS_EPT_COMPONENT {i+1} 0 {energies[j]:.17e} {c[j]:.17e}')
    rows.append('SPARKS_EPT_END')
    with target.open('x') as stream:
        stream.write('\n'.join(rows) + '\n')
    return target


def read_gdv_ept_dysons(path, state, *, frozen_core, method, energy_tolerance=1e-7):
    """Read SPARKS_EPT_DYSON_V1 blocks from an instrumented GDV L909 log.

    The matching checkpoint must come from the same run: MO phases cannot be
    recovered from energies. ``frozen_core`` is explicit, not guessed from the
    channel number. Only restricted electron-removal poles are supported here.
    All active orbitals and their signed coefficients must be present. Orbital
    energies qualify the active-to-full index map against the imported state.
    """
    if len(state.mo_coefficients) != 1 or not state.mo_energies_hartree:
        raise ValueError("GDV EPT import requires restricted MOs and their energies")
    if int(frozen_core) != frozen_core or frozen_core < 0:
        raise ValueError("Frozen core count must be a nonnegative integer")
    if not np.isfinite(energy_tolerance) or energy_tolerance <= 0:
        raise ValueError("Energy tolerance must be positive")
    lines = iter(Path(path).read_text().splitlines())
    result = []
    nmo = state.mo_coefficients[0].shape[1]
    for line in lines:
        if not line.startswith("SPARKS_EPT_DYSON_V1 "):
            continue
        try:
            symmetry, root, count, occupied, virtual = map(int, line.split()[1:])
            if count != occupied + virtual or count < 1 or count + frozen_core != nmo:
                raise ValueError("Active-space dimensions do not match the checkpoint")
            pole_line = next(lines).split()
            if len(pole_line) != 3 or pole_line[0] != "SPARKS_EPT_POLE":
                raise ValueError("Missing full pole data")
            pole_ev, strength = map(float, pole_line[1:])
            coefficients = np.zeros(nmo)
            seen = set()
            for _ in range(count):
                parts = next(lines).split()
                if len(parts) != 5 or parts[0] != "SPARKS_EPT_COMPONENT":
                    raise ValueError("Missing full signed Dyson components")
                index = int(parts[1]) - 1
                energy, coefficient = map(float, parts[3:])
                if index in seen or index not in range(count):
                    raise ValueError("Invalid or duplicated active MO index")
                seen.add(index)
                full_index = index + int(frozen_core)
                if not np.isfinite(energy) or abs(energy - state.mo_energies_hartree[0][full_index]) > energy_tolerance:
                    raise ValueError("Dyson orbital energies do not match checkpoint ordering")
                coefficients[full_index] = coefficient
            if next(lines).strip() != "SPARKS_EPT_END":
                raise ValueError("Missing Dyson block terminator")
            result.append(DysonOrbital.from_mo(
                state, coefficients, pole_strength=strength,
                ionization_energy_ev=-pole_ev, channel=f"{symmetry}:{root}",
                method=method, source=str(Path(path).resolve()),
            ))
        except (StopIteration, TypeError) as error:
            raise ValueError("Truncated GDV Dyson export") from error
    if not result:
        raise ValueError("No complete GDV Dyson export; printed pole weights are not a Dyson vector")
    return tuple(result)
