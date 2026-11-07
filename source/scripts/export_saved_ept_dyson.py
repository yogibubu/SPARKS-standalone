"""Recover a complete saved EPT vector without an electronic rerun.

The caller supplies an inspected offset or requests a unique-vector search. Record 694
must be dumped with the producer's rwfdump utility. The native printed orbital
energies independently check its symmetry-to-canonical MO map. A raw search
candidate alone is insufficient: all map, energy, pole and AO-norm checks apply.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from matrix_apoc import electronic_state_from_gaussian_fchk, read_gdv_ept_dysons
from matrix_apoc.gdv_dyson import write_ept_dyson_mo_export, locate_saved_ept_vector
from matrix_sparks.execution import require_compute_host


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('rwf', 'checkpoint', 'log', 'mapping-dump', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--word-offset', type=int,
                   help='Omit to locate a unique saved vector from native pole/norm evidence')
    p.add_argument('--frozen-core', type=int, required=True)
    p.add_argument('--symmetry', type=int, required=True)
    p.add_argument('--root', type=int, required=True)
    p.add_argument('--ionization-energy-ev', type=float, required=True)
    p.add_argument('--printed-pole-strength', type=float, required=True)
    a = p.parse_args()
    host = require_compute_host()
    if (a.word_offset is not None and a.word_offset < 0) or a.frozen_core < 0:
        raise ValueError('Offsets and frozen core must be nonnegative')
    state = electronic_state_from_gaussian_fchk(a.checkpoint)
    nmo = state.mo_coefficients[0].shape[1]
    nactive = nmo - a.frozen_core
    matching_offsets = ()
    if a.word_offset is None:
        a.word_offset, matching_offsets = locate_saved_ept_vector(
            a.rwf, nactive, ionization_energy_ev=a.ionization_energy_ev,
            printed_pole_strength=a.printed_pole_strength)
    no = int(np.count_nonzero(state.mo_occupations[0] == 2)) - a.frozen_core
    text = a.mapping_dump.read_text()
    match = re.search(r'Dump of file\s+694 length\s+(\d+).*?:', text)
    if not match or int(match[1]) != 2 * nactive:
        raise ValueError('Record 694 dimensions differ from the explicit active space')
    integers = np.array([int(x) for x in text[match.end():].split()], dtype=int)
    if len(integers) != 2 * nactive:
        raise ValueError('Incomplete symmetry-to-canonical map')
    mapping = integers[:nactive] - 1
    mapping[no:] += no
    if not np.array_equal(np.sort(mapping), np.arange(nactive)):
        raise ValueError('Active MO map is not a complete canonical permutation')
    log = a.log.read_text()
    heading = 'Orbital energies by symmetry in eV'
    if heading not in log:
        raise ValueError('Native orbital-energy table required to check the MO map')
    section = log.rsplit(heading, 1)[1].split('The 1st number', 1)[0]
    values = np.array([float(x) for x in section.split()])
    reference = state.mo_energies_hartree[0][mapping + a.frozen_core] * 27.211386245988
    if values.shape != reference.shape or not np.allclose(values, reference, atol=2e-5, rtol=2e-7):
        raise ValueError('Printed orbital energies do not match the same-run checkpoint map')
    with a.rwf.open('rb') as stream:
        stream.seek(8 * a.word_offset)
        raw = stream.read(8 * (nactive + 1))
    vector = np.frombuffer(raw, dtype='<f8')
    if len(vector) != nactive + 1 or not np.all(np.isfinite(vector)):
        raise ValueError('Missing complete finite signed-vector record')
    pole = float(vector[1:] @ vector[1:])
    if abs(vector[0] + a.ionization_energy_ev) > 5.1e-7:
        raise ValueError('Saved pole does not match the native printed root')
    if abs(pole - a.printed_pole_strength) > 5.1e-5:
        raise ValueError('Saved vector norm does not match the native pole strength')
    coefficients = np.zeros(nmo)
    coefficients[mapping + a.frozen_core] = vector[1:]
    provenance = dict(host=str(host), source=str(a.rwf), word_offset=a.word_offset,
                      matching_saved_offsets=matching_offsets,
                      raw_vector_sha256=hashlib.sha256(raw).hexdigest(),
                      checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                      mapping_sha256=hashlib.sha256(a.mapping_dump.read_bytes()).hexdigest(),
                      native_log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),
                      mo_energy_max_error_ev=float(abs(reference-values).max()),
                      no_electronic_rerun=True)
    write_ept_dyson_mo_export(a.output, state, coefficients,
                             frozen_core=a.frozen_core, ionization_energy_ev=-vector[0],
                             pole_strength=pole, symmetry=a.symmetry, root=a.root,
                             provenance=provenance)
    (dyson,) = read_gdv_ept_dysons(a.output, state, frozen_core=a.frozen_core, method='NR2')
    result = dict(provenance, ionization_energy_ev=dyson.ionization_energy_ev,
                  pole_strength=pole, ao_norm=dyson.norm_squared,
                  ao_norm_error=abs(dyson.norm_squared-pole), active_components=nactive)
    a.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
