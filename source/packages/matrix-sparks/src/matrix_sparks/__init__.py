"""SPARKS — Scattering and Photoionization: Amplitudes, Response Kernels and Spectra."""

from .grid import ElectronicGrid
from .photoelectron import ContinuumDipoleTable, fixed_photon_photoelectron_lines, gaussian_line_profiles
from .angular_observables import (
    photoelectron_angular_parameters,
    molecular_frame_photoelectron_distribution,
)
from .vibronic import (
    ContinuumOperatorBank,
    continuum_fc_ht_operator_bank,
    harmonic_photoionization_spectrum,
    vpt2_photoionization_spectrum,
    cached_harmonic_overlap_matrix,
)

__all__ = [
    "ElectronicGrid",
    "ContinuumDipoleTable",
    "fixed_photon_photoelectron_lines",
    "gaussian_line_profiles",
    "cached_harmonic_overlap_matrix",
    "photoelectron_angular_parameters",
    "molecular_frame_photoelectron_distribution",
    "ContinuumOperatorBank",
    "continuum_fc_ht_operator_bank",
    "harmonic_photoionization_spectrum",
    "vpt2_photoionization_spectrum",
]
