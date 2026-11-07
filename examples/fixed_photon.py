"""Small synthetic FC/HT example; no electronic program or reference data needed."""
from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np
from matrix_sparks.execution import require_compute_host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New directory for inputs and spectrum')
    args = parser.parse_args()
    require_compute_host()
    args.output.mkdir(parents=True, exist_ok=False)
    # Cartesian dipoles, one outgoing channel, constant + linear nuclear operators.
    coefficients = np.zeros((2, 3, 1, 2), dtype=complex)
    coefficients[:, 0, 0, 0] = [1 + .1j, .8 + .2j]
    coefficients[:, 1, 0, 1] = [.1j, .2j]
    identity = 'synthetic-example:fixed-Cartesian-frame:dimensionless-Q:v1'
    np.savez(args.output / 'continuum.npz', energies_hartree=[.1, .5],
             coefficients=coefficients, operator_basis_id=identity)
    np.savez(args.output / 'nuclear.npz', transition_energies_hartree=[.5, .51],
             nuclear_moments=[[.95, .02], [.15, .7]], operator_basis_id=identity)
    subprocess.run([sys.executable, '-m', 'sparks', 'photoelectron',
                    '--continuum', str(args.output / 'continuum.npz'),
                    '--nuclear', str(args.output / 'nuclear.npz'),
                    '--photon-energies-ev', '21.218',
                    '--output', str(args.output / 'spectrum.npz')], check=True)
    with np.load(args.output / 'spectrum.npz', allow_pickle=False) as data:
        print('FC, HT, interference and total cross sections / Mb:')
        print(data['cross_sections_Mb'])


if __name__ == '__main__':
    main()
