"""Compute fixed-photon photoelectron lines from portable numerical inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.constants import physical_constants

from .execution import require_compute_host
from .photoelectron import ContinuumDipoleTable, fixed_photon_photoelectron_lines


def _basis_identity(archive):
    """An archive must declare which signed nuclear operators it multiplies."""
    if 'operator_basis_id' not in archive:
        raise ValueError('Missing operator_basis_id: identify the common coordinate frame, origin, mode signs and units')
    value = np.asarray(archive['operator_basis_id'])
    if value.ndim != 0 or value.dtype.kind != 'U' or not str(value).strip():
        raise ValueError('operator_basis_id must be a nonempty Unicode scalar')
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continuum', type=Path, required=True,
        help='NPZ with energies_hartree and coefficients (energy,3,channel,operator)')
    parser.add_argument('--nuclear', type=Path, required=True,
        help='NPZ with transition_energies_hartree, nuclear_moments and optional populations')
    parser.add_argument('--photon-energies-ev', type=float, nargs='+', required=True)
    parser.add_argument('--channel-multiplicity', type=float, default=1.)
    parser.add_argument('--residual-charge', type=float,
        help='Also compute beta; continuum NPZ must contain integer (l,m) channels in the documented asymptotic convention')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    host = require_compute_host()
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    with np.load(args.continuum, allow_pickle=False) as data:
        operator_basis_id = _basis_identity(data)
        provider = ContinuumDipoleTable(data['energies_hartree'], data['coefficients'])
        angular = {}
        if args.residual_charge is not None:
            from .continuum import RealSphericalChannels
            angular = dict(angular_channels=RealSphericalChannels(data['channels']),
                           residual_charge=args.residual_charge)
    with np.load(args.nuclear, allow_pickle=False) as data:
        if _basis_identity(data) != operator_basis_id:
            raise ValueError('Continuum and nuclear operator bases differ; transport the complete operator before contraction')
        result = fixed_photon_photoelectron_lines(
            np.asarray(args.photon_energies_ev)/physical_constants['Hartree energy in eV'][0],
            data['transition_energies_hartree'], data['nuclear_moments'], provider,
            populations=data['populations'] if 'populations' in data else None,
            channel_multiplicity=args.channel_multiplicity, **angular)
    if args.output.exists():
        raise FileExistsError('Preserve completed output; choose another filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    metadata.update(host=host, operator_basis_id=operator_basis_id,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in (args.continuum, args.nuclear)})
    np.savez_compressed(args.output, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
                        metadata_json=json.dumps(metadata))
    print(json.dumps(dict(metadata, summed_cross_sections_Mb=result['cross_sections_Mb'].sum(axis=-1).tolist()), indent=2))


if __name__ == '__main__':
    main()
