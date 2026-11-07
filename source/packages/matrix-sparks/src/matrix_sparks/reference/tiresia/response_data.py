"""Checked serial raw Galerkin inputs for independent response reconstruction."""

from pathlib import Path
import numpy as np


def read_response_operators(path):
    with Path(path).open('rb') as stream:
        header = np.fromfile(stream, '<i4', 4)
        if len(header) != 4 or np.any(header < 1):
            raise ValueError('Invalid response operator dimensions')
        n, nc, nk, order = map(int, header)
        scalar = np.fromfile(stream, '<f8', 2)
        angular = np.fromfile(stream, '<i4', nc)
        counts = np.fromfile(stream, '<i4', nc)
        values = np.fromfile(stream, '<f8')
    if (len(scalar) != 2 or len(angular) != nc or len(counts) != nc
            or len(values) != nk + 2*n*n or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(scalar))):
        raise ValueError('Truncated or nonfinite response operators')
    return dict(dimension=n, charge=float(scalar[0]), asymptotic_shift=float(scalar[1]),
                angular_momenta=angular, radial_counts=counts, degree=order-1,
                knots=values[:nk], overlap=values[nk:nk+n*n].reshape(n, n, order='F'),
                strong_hamiltonian=values[nk+n*n:].reshape(n, n, order='F'))


def read_response_source(path):
    with Path(path).open('rb') as stream:
        header = np.fromfile(stream, '<i4', 8)
        if len(header) != 8 or np.any(header < 1):
            raise ValueError('Invalid raw response source dimensions')
        n, p, symmetry, orbital, dipole, nc, angular_pair, component = map(int, header)
        scalar = np.fromfile(stream, '<f8', 2)
        values = np.fromfile(stream, '<f8')
    if (len(scalar) != 2 or not np.all(np.isfinite(scalar)) or scalar[1] <= 0
            or len(values) != n*p or not np.all(np.isfinite(values))):
        raise ValueError('Truncated or nonfinite raw response source')
    return dict(target_symmetry=symmetry, orbital=orbital, dipole=dipole,
                angular_pair=angular_pair, component=component,
                energy=float(scalar[0]), occupation=float(scalar[1]),
                sources=values.reshape(n, p, order='F'))
