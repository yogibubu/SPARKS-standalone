# SPARKS standalone

Scattering and Photoionization: Amplitudes, Response Kernels and Spectra.

The archive includes the independent Python continuum solver, C++ AO grid
source, response operators, complex-amplitude FC/HT contraction, nuclear solvers,
electronic-data importers, tests and all internal Python dependencies at one
recorded revision. It does not require a nano-MATRIX checkout or installation.
Gaussian and ORCA executables are separately licensed external programs.
TIRESIA remains an optional numerical reference and is not required to run SPARKS.

## Installation

Python 3.11 or newer is required. Use a separate virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python install.py
sparks --help
sparks photoelectron --help
sparks response --help
```

The offline archive includes internal and third-party Python wheels for the
Python version and operating system recorded in `wheelhouse-manifest.json`.
Installation checks the source and wheel SHA-256 values and runs `pip check`.
It never contacts a package index. An existing nano-MATRIX installation is
unnecessary. Binary dependencies such as NumPy, SciPy and PySCF are included
in the platform-specific wheel archive, with their distribution metadata.

The portable source archive is available for other platforms. On a compute
host with internet access run `python install.py --source`, or prepare an
offline archive using:

```sh
python source/tools/build/build_sparks_standalone.py build /absolute/path/to/SPARKS_standalone
```

A C++17 compiler and platform development headers are needed for the native
AO-grid and vibrational kernels, which compile on first use. On macOS install
the Command Line Tools (`xcode-select --install`); on Linux provide a C++17
compiler. These system tools and licensed electronic-structure executables
are separate from the Python distribution. Numerical execution in this
research release retains the Oracle/MAC-STUDIO host policy. Source inspection,
transfer and command help can run on the laptop.

## Included components

- Independent continuum solver, molecular potentials, response and angular observables.
- Complex Franck--Condon (FC) and Herzberg--Teller (HT) amplitudes, harmonic
  nuclear overlaps, anharmonic contraction and bounded-memory spectral profiles.
- Electronic-data readers and nuclear-coordinate/force-field services, with
  the complete mandatory internal dependency closure.
- Native C++ sources, package data, license, scientific checks, and import and
  preparation scripts under `source/scripts/`.
- An executable synthetic example under `examples/`; it requires no molecular
  files, electronic job, TIRESIA installation or original repository.

The complete internal package list and exact source revision are recorded in
`source-manifest.json`. The directory may be moved or renamed after unpacking.
TIRESIA Fortran overlays are excluded; optional Python reference readers are
included for users who already have reference data. Production calculations
use the independent SPARKS implementation.

## Example and input data

On a compute host, with the environment activated:

```sh
python examples/fixed_photon.py example-output
```

This small synthetic example writes portable continuum/nuclear inputs, invokes
the installed `sparks photoelectron` command, and prints FC, HT, interference
and total cross sections. It illustrates units and coherent contraction;
its numbers are not molecular predictions. Each output directory must be new.

For molecular calculations supply an electronic density/checkpoint, a
channel-specific Dyson orbital, nuclear fields and the desired electron/photon
energies. These are scientific inputs, not missing software dependencies.
The supplied preparation scripts document their inputs through `--help`.
Gaussian formatted checkpoints and saved electron-propagator results can be
imported without rerunning electronic calculations. Generating new electronic
data requires the corresponding separately installed backend.

## Commands

- `sparks photoelectron --help`: fixed-photon photoelectron lines from complex
  continuum coefficients and nuclear transition moments.
- `sparks response --help`: solve `(I-K chi)V=Vext` from exported response
  matrices and contract the effective vertices. The input susceptibility and
  kernel must already use the same declared dual auxiliary basis. This command
  does not generate missing electronic response data.
- `python -m pytest source/tests/test_sparks_response.py`: algebra, full virtual
  sum and occupied-projection controls. Numerical work is restricted to the
  configured Oracle or Mac Studio compute hosts.

The continuum phase, coordinate frame, expansion origin and units are part of
`operator_basis_id`. Nuclear and electronic files with different identities
cannot be contracted. A Dyson norm and spin multiplicity are distinct inputs;
no empirical extra pole-strength multiplier is applied.

## Python API

`matrix_sparks.continuum.GalerkinSternheimerResponse` evaluates both frequency
terms in a projected sparse resolvent. An explicit boundary self-energy is
required for an open continuum. Its absence specifies a finite closed box.
`screen_response` accepts the Coulomb/XC kernel explicitly and solves all
external fields together using compiled LAPACK. See the supplied tests for
minimal reproducible examples and normalization conventions.

`matrix_sparks.continuum.LocalMolecularTDDFT` constructs the molecular response
from a `LocalPhotoionizationCalculation`, a closed-shell electronic density
and an explicit `GaussianResponseAuxiliary`. All occupied projectors, source
integrals, auxiliary Coulomb/XC matrices, susceptibility, screened fields and
complex amplitudes are generated by the independent code. Cartesian dipoles
remain separate exact operators, and the auxiliary response space conserves
total charge. Select the XC response derivative explicitly; LB94 does not
define it. Sparse and complete-spectrum response backends share the same
physical exterior. The current integration owner accepts central spherical
operators and rejects atomic-enriched response assembly.

The supplied `source/scripts/qualify_sparks_checkpoint_response.py` runs a
complete checkpoint-based example with TIRESIA imports, TIRESIA file access
and external executable launches blocked. Run `--help` for the explicit
Gaussian checkpoint/Dyson inputs, auxiliary basis, response kernel and
numerical dimensions. It also checks the absorptive response against the sum
of outgoing-channel strengths. No new electronic structure job is launched.
