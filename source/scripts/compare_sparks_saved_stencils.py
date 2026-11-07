"""Compare saved continuum stencils with identical portable nuclear moments.

This analysis starts no electronic or continuum calculation. Complex amplitudes
are interpolated and contracted by the existing fixed-photon implementation.
A JSON specification supplies the nuclear archives, requested photon energies,
line indices and derivative folders for any molecule or stencil sequence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from matrix_sparks.execution import require_compute_host
from matrix_sparks.continuum import RealSphericalChannels
from matrix_sparks.photoelectron import ContinuumDipoleTable, fixed_photon_photoelectron_lines


def load_bank(folder, derivative):
    with np.load(folder / 'center.npz') as data:
        energies = data['energies'].copy()
        constants = data['amplitudes'].copy()
        labels = [tuple(map(int, pair)) for pair in data['channels']]
    files = sorted(folder.glob('derivative-mode*.npz'))
    if not files:
        raise ValueError('No saved normal-coordinate derivatives')
    derivatives = []
    for file in files:
        with np.load(file) as data:
            if not np.array_equal(data['energies'], energies):
                raise ValueError('Stencil electron-energy grids differ')
            derivatives.append(data[derivative].copy())
    coefficients = np.concatenate((constants[..., None], np.stack(derivatives, axis=-1)), axis=-1)
    return ContinuumDipoleTable(energies, coefficients), RealSphericalChannels(labels)


def summarize(answer, line_indices):
    sigma = answer['cross_sections_Mb'][:, 3]
    beta = answer['beta']
    total = sigma.sum(axis=-1)
    if np.any((sigma > 0) & ~np.isfinite(beta)):
        raise ValueError('Positive-strength lines require finite anisotropies')
    weighted = np.sum(sigma * np.where(sigma > 0, beta, 0.0), axis=-1)
    average = np.divide(weighted, total, out=np.full_like(total, np.nan), where=total > 0)
    def finite_list(values):
        return [float(value) if np.isfinite(value) else None for value in values]

    reference = sigma[:, line_indices['reference']]
    lines = {}
    for label, index in line_indices.items():
        ratio = np.divide(sigma[:, index], reference,
                          out=np.full_like(reference, np.nan), where=reference > 0)
        lines[label] = dict(cross_section_Mb=finite_list(sigma[:, index]),
                            ratio_to_reference=finite_list(ratio),
                            beta=finite_list(beta[:, index]))
    return dict(summed_cross_sections_Mb=finite_list(total),
                strength_weighted_beta=finite_list(average), lines=lines)



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    host = require_compute_host()
    if a.output.exists():
        raise FileExistsError(a.output)
    spec = json.loads(a.specification.read_text())
    photon = np.asarray(spec['photon_energies_ev'], float) / 27.211386245988
    archives = {}
    for model, path in spec['nuclear_archives'].items():
        with np.load(path) as data:
            archives[model] = (data['transition_energies_hartree'].copy(), data['nuclear_moments'].copy())
    output = dict(host=host, photon_energies_ev=spec['photon_energies_ev'],
                  electronic_jobs=0, continuum_calculations=0, cases=[])
    for case in spec['cases']:
        folder = Path(case['folder'])
        table, channels = load_bank(folder, case.get('derivative', 'first'))
        row = dict(name=case['name'], derivative=case.get('derivative', 'first'), models={})
        for model, (binding, moments) in archives.items():
            result = fixed_photon_photoelectron_lines(photon, binding, moments, table,
                channel_multiplicity=spec['channel_multiplicity'], angular_channels=channels,
                residual_charge=spec['residual_charge'])
            row['models'][model] = summarize(result, spec['line_indices'][model])
        row['source_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [folder/'center.npz', *sorted(folder.glob('derivative-mode*.npz'))]}
        output['cases'].append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
