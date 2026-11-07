# Standalone release 0.1.0rc8

Scientific source revision: `aff75c631`.

The public source distribution includes 23 internal Python components, native
C++ sources, required package data, scientific checks, input preparation tools,
an installer and a synthetic FC/HT example. The source and wheel inventories
are SHA-256 verified by the installer.

## Recorded installation check

The matching macOS arm64 / CPython 3.12 wheelhouse contains 53 wheels. It was
installed without a package index in a newly created virtual environment on
MAC-STUDIO. `pip check` reported no broken requirements. With no inherited
PYTHONPATH and with a working directory outside the source distribution,
all 23 internal modules imported from the isolated installation. General,
photoelectron and response command help succeeded. The supplied fixed-photon
example produced FC, HT, interference and total cross sections, with no
TIRESIA modules loaded.

This is installation and example verification. Molecular validation is
reported separately in the accompanying scientific work. Licensed electronic
executables and a C++17 compiler are separately installed system tools.

## Distribution layout

- Git repository and GitHub source archives: sources, tests, installer,
  examples, license and integrity inventories.
- Release wheelhouse asset: platform-specific third-party and internal wheels.
- Development code: maintained independently; no development history or
  automatic synchronization is included in this repository.
