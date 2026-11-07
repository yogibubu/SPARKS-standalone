# SPARKS

**Scattering and Photoionization: Amplitudes, Response Kernels and Spectra.**

SPARKS owns the production continuum implementation. TIRESIA is retained in
`matrix_sparks.reference.tiresia` solely as a numerical reference, with its
build, qualification and benchmark harness. It is not required by the new
continuum solver. Full molecular photoionization equivalence is not yet qualified.

The independently assembled H2+ spectrum agrees with TIRESIA over 160 energies
to 2.80e-8 in strength. The CO solver comparison covers 150 energies and three
symmetries with shared operators. Independently constructed CO LB94 and exact
Gaussian Hartree components have separate controls; the reference's finite
Poisson Hartree representation is explicitly distinguished. These comparisons
do not equate local one-particle DFT with the additional response models of
the reference program.

## Independent continuum core

### Repeated spectra and bounded workspace

`cached_harmonic_overlap_matrix` reuses the existing TRINITY overlap evaluator
and stores complete nuclear overlaps by final-state labels. The frequencies,
full mode map, displacement and initial states identify the saved data. A
different electronic operator or photon energy reuses the same overlaps;
new final states alone require evaluation. Input identities and content
checksums prevent reuse with a different nuclear model or altered arrays.
The adaptive TI selection and intensity recovery are checked for each operator.

`gaussian_line_profiles(axis, centers, strengths, sigma=...)` broadens real
line strengths, including signed interference contributions, in bounded blocks.
All lines and Gaussian tails are retained. `maximum_kernel_entries` limits
the temporary Gaussian matrix, independently of the number of spectral points
and transitions. The molecular harmonic/VCI comparison and the saved-VCI
window workflow use this same routine.

Compact-enriched `LocalPhotoionizationCalculation` checks overlap positivity
before constructing the Hartree/XC field. The geometry-only assembly avoids
spatial potentials and Cartesian gradients. The complete electronic assembly
retains its independent final overlap check. `overlap_preflight_seconds`
records the preliminary check separately from total preparation time.

### Polarization-resolved molecular-frame distributions

`molecular_frame_photoelectron_distribution` reconstructs angular strengths
from the same complex Cartesian/channel dipoles used for total cross sections
and beta. It accepts a unit-trace positive Hermitian polarization density
matrix, including linear, circular, elliptical and incoherent mixtures. The
result retains partial-wave interference and includes an analytic angular
integral; arbitrary observation directions are optional.

```python
from matrix_sparks import molecular_frame_photoelectron_distribution

result = molecular_frame_photoelectron_distribution(
    dipoles, channels, electron_energy_hartree=energy,
    residual_charge=1, polarization_density=polarization_density,
)
# result['angular_strength_per_steradian']: batch axes followed by direction
# result['integrated_dipole_strength']: exact angular integral
```

The photon polarization and directions are in the molecular Cartesian frame.
The caller supplies a polarization transverse to the photon propagation.
Multiplication by `4*pi**2*alpha*photon_energy_hartree` gives bohr squared per
steradian. Population and channel multiplicity remain explicit. No isotropic
one-third factor is used for this fixed-orientation observable. This interface
does not perform a chiral molecular-orientation average.

`matrix_sparks.continuum` provides a replaceable `ContinuumBackend` contract and
an independent `SpectralRMatrixBackend`. A real inner-region Bloch Hamiltonian,
overlap, boundary values and physical source integrals form its input. It
solves the generalized eigenproblem once, then contracts the full spectral
resolvent and solves only a channel-sized outgoing-boundary system at each
energy. An exact inner-region pole uses a direct modal solve without shifting
the energy. All eigenpairs are retained.

`assemble_radial` supplies general coupled local radial potentials using
SciPy B-splines, quadrature and BLAS/LAPACK. It imposes u(0)=0 and an open outer
boundary. `free_boundary` supplies energy-normalized neutral-channel waves.
`coulomb_boundary` supplies charged waves through a controlled outgoing
asymptotic expansion, SciPy DOP853 propagation and angular recurrence where
oscillatory; mpmath remains an independent test oracle only.
`propagate_boundary` retains coupled molecular tails through full channel
matrices. Full molecular TIRESIA amplitude equivalence, nonlocal exchange and TDDFT
response remain to be qualified or implemented; the present code is a local one-particle continuum solver.

```python
from matrix_sparks.continuum import SpectralRMatrixBackend, free_boundary

solver = SpectralRMatrixBackend(H_bloch, S, boundary_values, source_integrals)
result = solver.evaluate(energy_hartree, free_boundary(energy_hartree, radius, l))
# result.amplitudes: (sources, channels), complex, energy normalized
# result.scattering: channel scattering matrix in the G+iF convention
```

All operator and boundary units are atomic. Sources include their physical
angular projection and normalization. Angular Coulomb phases are not applied
automatically. Matching waves must represent the actual potential outside the
boundary. Pure Coulomb matching is insufficient for an unconverged molecular
dipole tail. Real local operators and open channels are the initial scope;
complex/nonlocal/closed-channel models require appropriate extensions.

The remote qualification command is `scripts/qualify_sparks_continuum.py`.
It checks free scattering, coupled-channel amplitudes against a full solve,
unitarity, reciprocity, an exact inner-region pole, and the analytic hydrogen
1s photoionization cross section. Numerical reports live in
`docs/reports/sparks_continuum_20261026/`.

## Molecular operators

* `RealSphericalChannels`: explicit real (l,m) labels, positive Cartesian p
  functions, finite-space angular quadrature and dipole contractions.
* `NuclearPotential`: off-centre point charges projected analytically through
  the finite multipole expansion. Include nuclear radii among radial knots.
  Coordinates are relative to the common expansion origin.
* `solve_bound_state` and `bound_dipole_source`: reconstruct a normalized
  finite-box one-particle bound state and its three Cartesian dipole sources.
* `AODysonOrbital`: explicit signed AO Dyson coefficients from an APOC state;
  actual norm is retained. It accepts Cartesian and spherical AO conventions
  via the existing compiled PySCF/libcgto evaluator.
* `AOHartreePotential`: exact batched PySCF/libcint `int1e_grids` contractions
  with the actual total density from APOC, including spin sums and an explicit
  memory budget. It is only the electronic Coulomb repulsion; it does not imply
  an exchange or polarization approximation.
* `ProjectedLocalPotential` and `ProjectedDipoleSource`: stream scalar electronic
  potentials or spatial orbitals into angularly projected radial operators.
  A callable lets another electronic backend replace the AO provider.

No default electronic XC model or new electronic keywords are introduced.
The existing C++ grid can supply density/gradient arguments to a separately
specified local-potential callback; APOC continues to own all electronic parsers.

The molecular qualification is `scripts/qualify_sparks_molecular.py`. Tests run
only on Oracle/MAC-STUDIO. The first molecular application is H2+ with two point
nuclei at z=+-1 bohr. The central angular/radial space can now be enriched by compact atomic-centred
B-splines through `assemble_multicentre`. H2+ cross sections have
been computed, but their agreement with the distributed TIRESIA spectrum is
not established. See `docs/reports/sparks_molecular_20261026/` for results,
normalization conventions and convergence controls.

## Electronic grid service

A reusable electronic density/gradient grid core in C++ is exposed through
Python and connected to the reference TIRESIA quadratures by a Fortran adapter.

## Reuse and numerical contract

* APOC owns electronic states, basis conventions and existing FCHK/Molden/
  PySCF/TREXIO imports. `ElectronicGrid.from_electronic_state` reconstructs its
  validated AO space through APOC and uses its actual total density, summing
  spin channels. It does not run a new SCF calculation or parse a new file format.
* The existing PySCF/libcgto implementation evaluates Gaussian AOs and their
  derivatives. The C++ core contracts batches with existing BLAS `dgemm` and
  computes density and gradient magnitude. No new AO evaluator is introduced.
* Blocks contain at most 256 points. All shell masks are enabled; no screening
  threshold is added. Current qualification covers real Cartesian s–g functions.
  Pure spherical AOs and complex coefficients are rejected by this initial API.
* A context owns copies of the state and reusable workspace. Recreate it after
  changing geometry/basis/density. Independent contexts are separate; concurrent
  calls to the same context are unsupported. Use a context manager or `close()`.

## Native build and use

On a compute host with the existing GNU/MPI/BLAS and PySCF installations:

```sh
python scripts/build_tiresia_gnu.py \
  --source /path/to/official/tiresia-cpip \
  --mpi /path/to/mpi --scalapack /path/to/libscalapack.a \
  --blas /path/to/libopenblas.so \
  --gto-batch --hartree-batch --sparse-blocks \
  --sparks-source packages/matrix-sparks/src/matrix_sparks/native/gto_grid.cpp \
  --targets main
```

The builder records the immutable native library path and hashes in
`gnu_compat/gto_batch/native-library.json`. Older executables retain their
original dependencies. The optional `--cgto-library` identifies the existing
libcgto explicitly; otherwise the compute-host PySCF installation supplies it.
This does not install packages, change electronic keywords or launch jobs.
Build the separately qualified spectrum postprocessor using the same builder
and toolchain arguments with `--angular-cache --targets sigma`.

```python
from matrix_sparks import ElectronicGrid

with ElectronicGrid.from_electronic_state(native_library_path, apoc_state) as grid:
    density, gradient_magnitude = grid.evaluate(points_in_bohr)
```

For a Cartesian PySCF molecule and matching real coefficients/occupations,
`ElectronicGrid(native_library_path, molecule, coefficients, occupations)` is
also available. AO normalization and ordering must already match that molecule.

## TIRESIA reference harness

`scripts/tiresia_gto_batch.py` converts the already-read Tiresia basis once,
matching Cartesian angular components and normalization explicitly. It batches
the existing two-centre quadrature coordinates in the same order and checks
that scalar consumers use every cached result exactly once. The spherical
Tiresia path retains its original evaluator.

`scripts/tiresia_hartree_batch.py` factors radial Hartree contractions before
their repeated azimuthal use. This changes floating-point association, with
agreement checked against the literal upstream loop; it omits no term.
`scripts/tiresia_angular_cache.py` reuses exact upstream dipole Wigner values
across energies, with a reference fallback outside its cached range.

See `docs/reports/tiresia_20261026/` and the `qualify_*` / `benchmark_*` scripts
for numerical checks and measured timings. Run substantial qualification and
production jobs only on the authorized compute hosts.

The reference Python code is organized under `reference/tiresia/` into
`overlays`, `qualification`, `benchmarks`, shared execution utilities and
separate Fortran resources. Top-level commands remain compatibility entry
points. `qualify_tiresia_layout.py` verifies that relocation generates exactly
the previously qualified Fortran bytes. Upstream source files remain unchanged.

## Atomic enrichment and angular quadrature

`AtomicSplineBlock` and `assemble_multicentre` add compact local bases, analytic
Cartesian gradients, full central/atomic overlap and Hamiltonian blocks, and a
caller-supplied atomic symmetry transform. `solve_multicentre_bound_state` and
`with_multicentre_dipole_sources` retain atomic contributions in the bound state
and dipoles. Local supports must be disjoint, exclude the central origin and
lie inside the matching sphere. Value and derivative vanish at their cutoff;
no overlap eigenvectors are silently dropped. These are real local operators.

`RealSphericalChannels(labels, quadrature="lebedev", lebedev_order=59)` reuses
PySCF's existing Lebedev tables; the original product rule remains available.
Translated cross blocks need angular convergence beyond finite harmonic-product
exactness. Run `scripts/qualify_sparks_multicentre.py` on Oracle/MAC-STUDIO.
See `docs/reports/sparks_multicentre_20261026/` for numerical results, measured
speed changes and the literature comparison with FE-DVR, overset grids,
multiwavelets and Gaussian alternatives. No wavelet speedup is claimed.

## Sparse outgoing resolvent

`assemble_sparse_radial` uses the same quadrature, angular coupling and source
contracts as `assemble_radial`, retaining exact radial support zeros in CSC
storage. Passing these operators through `assemble_multicentre` preserves sparse
storage for the full central/atomic Hamiltonian, overlap and boundary columns.
Bound-state reconstruction and the existing dipole contractions accept these
operators. No basis functions or small couplings are discarded.

`SparseResolventBackend(H, S, B, D)` implements the existing `ContinuumBackend`
interface. It solves the outgoing-boundary linear system at each requested
energy, without complete diagonalization or shifting inner-region poles.
Diagonal and full coupled boundary waves are supported. The caller supplies a
qualified positive-definite Galerkin overlap; symmetry/finiteness/positive
diagonal are checked, but no separate global SPD eigentest is performed.
`last_diagnostics` reports sparse LU nonzeros and a normwise backward residual.

Keep `SpectralRMatrixBackend` for scans where one eigensolve is amortized over
many energies. Sparse preparation is cheaper, but LU is repeated at each energy;
no automatic backend selection or universal speedup is claimed.
Qualifications: `scripts/qualify_sparks_resolvent.py` (CO, poles, full boundary
matrices), `scripts/qualify_sparks_sparse_assembly.py` (independent sparse
assembly, bound states and dipoles), with results in
`docs/reports/sparks_resolvent_20261026/`.

## Optional fitted Hartree provider

`AODensityFittedHartreePotential(state, auxiliary_basis, ...)` is an explicit
alternative to the exact Gaussian Hartree provider. It reuses PySCF three-centre
integrals and an auxiliary Coulomb metric, streams AO-pair tensors and constrains
the fitted density to preserve the actual imported electron count. It does not
alter the source AO density, add electronic keywords, or run SCF.

The optional `cholesky_tolerance` selects independent auxiliary functions with
PySCF's pivoted Cholesky of the unit-diagonal-scaled Coulomb metric. Its value is
explicit; no hidden jitter or metric cutoff is introduced. This is **not** a
direct Cholesky decomposition of the complete four-index AO ERI tensor.

The exact provider remains the reference/default. Auxiliary-basis, metric-rank,
field/matrix and observable convergence must be checked before adopting the
fit. Water analytic-probe checks and actual water/naphthalene/anthracene provider
benchmarks are archived in `docs/reports/sparks_hartree_fit_20261026/`.

## Explicit outer-region models

`CoulombTailPotential(potential, inner_radius=..., outer_radius=..., charge=...)`
joins a scalar spatial field to `-charge/r` with a C2 quintic switch. The input
field is unchanged inside the inner radius. Selecting the splice is an explicit
change of local model, with its own sensitivity check; the raw LB94 default is
unchanged. Applied to XC alone, it retains the separately evaluated nuclear and
Hartree multipoles. Coordinates, radii and potentials use atomic units.

`InterpolatedRadialPotential(radii, matrices)` caches symmetric channel matrices
with SciPy cubic interpolation over a finite interval and rejects extrapolation.
Pass this to the existing `propagate_boundary` to avoid AO integrals on every
ODE evaluation. Check interpolation by midpoint residuals and grid refinement.

The actual density trace is retained. A nominal neutral XC tail `-1/r` combined
with an imported density trace N has total Coulomb charge `sum(Z)-N+1`;
`scripts/qualify_sparks_dyson_continuum.py` records this charge explicitly when
`--xc-tail` is selected. It also accepts a distinct outer matching radius and
checks the sampled tail against directly evaluated midpoint potentials.
`AODensityGradient` uses a scaled hypot norm so finite gradients below 1e-162
are not lost through squaring underflow. Tests run on the permitted compute hosts.

`DipoleCoulombBoundary(l, D, charge)` treats the leading anisotropic tail
`-charge/r I + D/r^2` without a finite dipole cutoff. It diagonalises
`diag(l*(l+1))+2*D` once and integrates the outgoing scalar Coulomb equations
in that eigenbasis. The original spherical-channel phases are restored before
returning `BoundaryWaves`; a changing eigenvector gauge is not exposed to Dyson
amplitudes. Negative centrifugal eigenvalues, including values below -1/4,
are retained. Unit tests compare independently evaluated Whittaker functions
and verify box invariance of both amplitudes and scattering matrices. Higher
multipoles are kept by `propagate_boundary` between the outer reference radius
and the inner spline box. The water driver obtains D from exact AO position
integrals of the same supplied density and retains its actual trace.

## Geometry-derivative quadrature

`RadialBasis(..., quadrature_breakpoints=...)` separates integration cuts from
spline knots. Nuclear radii at each displaced geometry can split quadrature
intervals while all stencil points retain the same variational basis.
`assemble_radial` and `assemble_sparse_radial` accept this explicit basis as
`radial_basis=...`, checking its breakpoints, degree and quadrature. The default
assembly path is unchanged. This prevents moving/duplicated nuclear knots from
introducing trial-space changes into a finite-difference amplitude derivative.

## Continuum amplitudes and vibronic FC/HT

`LocalPhotoionizationCalculation` prepares one typed APOC density/Dyson pair,
fixed radial trial space, explicit XC callback and asymptotic boundary once.
Its `amplitudes(energies)` returns complex Cartesian/channel amplitudes; it does
not insert photon, spin or orientation prefactors. The qualification CLI uses
this same API. APOC aligns AO tensors and follows the Dyson overlap/sign before
normal-coordinate differentiation. `symmetric_amplitude_derivative` preserves
complex values and accepts explicit three-point (basis check) or five-point
(step check and fourth-order first derivative) symmetric stencils.

`continuum_fc_ht_operator_bank` sums all electronic channels exactly into the
polynomial basis `(1,q1,...,qn)`. Its small Hermitian forms separately retain FC,
HT, interference and their sum. No singular-value cutoff or channel pruning is
used. `harmonic_photoionization_spectrum` passes those operators to TRINITY's
existing harmonic TI/TD implementation (optional `vibronic` dependency). The
caller must transport gradients into the initial coordinates of the nuclear
model. Outputs are conditional dipole strengths at fixed electron kinetic
energy, not absolute or fixed-photon-energy photoionization spectra.

The reusable drivers `prepare_sparks_normal_derivatives.py`,
`compute_sparks_normal_derivatives.py` and `qualify_sparks_vibronic_bridge.py`
retain approved electronic templates, content-hashed per-point checkpoints,
normal-coordinate transport, sum-rule and full-grid TI/TD comparisons. Heavy
runs are restricted to Oracle/MAC-STUDIO. A resumed stencil reuses amplitudes;
its elapsed time describes the resume, not the original production calculation.

`vpt2_photoionization_spectrum` passes the same channel forms to TRINITY's
explicit-target, two-surface SAM operator bank. It returns strict order-two
strengths and separate completed squares. Continuum derivatives must match the
initial modal gauge of the supplied QFF/model. A rectilinear potential-only QFF
must be compared with its matching rectilinear harmonic model, rather than
silently substituted into an ESALC model with a different kinetic metric.

## Fixed-photon photoelectron cross sections

`fixed_photon_photoelectron_lines` connects continuum and nuclear calculations
at a specified photon energy. For every nuclear transition it evaluates the
complex continuum operator at `electron_energy = photon_energy - binding_energy`.
Closed channels return zero and are never sent to the continuum solver. Inputs
are portable nuclear transition energies and moments of `(1,q1,...,qn)`; the
same interface accepts harmonic or variational overlaps without changing the
continuum model. Populations and unresolved-channel multiplicities are explicit.

The returned isotropic cross sections in Mb use
`4*pi**2*alpha*photon_energy/3` times the summed energy-normalized dipole strength
and the Bohr-area conversion. The physical Dyson norm is already in its signed
AO coefficients. No additional pole strength, spin factor or electron momentum
is inserted automatically. FC, HT, signed interference and FCHT are returned
separately. For a closed-shell spatial electron-removal pole, summing the two
orthogonal ionic spin components requires an explicit multiplicity of two.

`ContinuumDipoleTable` interpolates complex amplitudes in one fixed channel/phase
gauge, before nuclear contraction and squaring. It rejects extrapolation.
Intermediate-energy checks establish interpolation accuracy for each use.
A direct continuum callable can replace it through the same interface.

The portable command, run on Oracle or MAC-STUDIO, is:

```sh
sparks-photoelectron --continuum continuum.npz --nuclear nuclear.npz \
  --photon-energies-ev 21.218 --channel-multiplicity 2 --output photoelectron.npz
```

The continuum archive stores `energies_hartree` and `coefficients` with axes
`(energy,3,channel,operator)`; the nuclear archive stores
`transition_energies_hartree`, `nuclear_moments` and optional `populations`. Both archives must contain the same nonempty `operator_basis_id`, identifying the common coordinate frame, origin, mode signs and units. The command rejects missing or mismatched identities before contraction.
The result preserves units, inputs' content hashes and the verified compute
host. This cross-section calculation is distinct from the earlier conditional
fixed-electron-energy strength distribution.

`photoelectron_angular_parameters` evaluates the linear-polarization anisotropy
beta by exact finite-space angular quadrature, with explicit centrifugal and
Coulomb phases. It reuses `RealSphericalChannels`; no new angular basis is
introduced. Tests include analytic beta=2 and beta=0 limits. Independent
postprocessing of reference amplitudes reproduces TIRESIA's H2+ and CO beta
over 160 and 900 energy/state combinations, respectively. This is an angular
postprocessor comparison, separate from the continuum-solver comparison.
The fixed-photon API optionally accepts `angular_channels` and `residual_charge`
to obtain beta separately for each nuclear line. The CLI enables this through
`--residual-charge`; the continuum archive must then also store `(l,m)` labels
under `channels`. Raw K-normalized real dipoles are not suitable inputs.

## Response optimization without model truncation

The projected sparse response batches sources of exactly degenerate occupied
orbitals through the same factorization, retaining their distinct occupations.
Exactly real operators use real sparse LU; any nonzero imaginary component
keeps the complex path. The complete spectral response uses real virtual-space
products at real frequencies and reuses the transposed source/boundary
contraction. No virtual state, pole or damping is removed.

In the oscillatory Coulomb region the response boundary propagates the outgoing
logarithmic derivative directly, then recurs on angular wave ratios. It reuses
the asymptotic-series owner of the normalized-wave solver. The full-wave path
remains in use near the angular turning region. Real adiabatic kernels use two
real BLAS products to construct complex screening, without unnecessary input
copies.

Matched and interleaved timing records, source hashes, independent Whittaker
checks and the complete regression log are in
`docs/reports/sparks_response_optimization_20261106/`.

## Dipoles without complete scattering states

`SparseResolventBackend.evaluate_amplitudes(energy, waves)` returns only the
complex source/channel amplitudes. For fewer sources than channels it solves
`A.T @ X = D` using the same outgoing sparse factorization, then contracts
`0.5 * (B.T @ X).T @ driving`. The transpose is unconjugated. Every source,
channel and outgoing phase is retained. When sources are more numerous, the
existing forward solve is used. `evaluate` still returns the complete
scattering matrix and boundary values.

`LocalPhotoionizationCalculation.amplitudes` uses this exact reduced-RHS path
when supported and retains compatibility with backends exposing only
`evaluate`. Diagnostics identify the solve direction and RHS count.

On the archived 1484-function CO problem (13 channels, three sources), a
three-energy scan with one numerical thread takes median 0.12879 s for the
full result and 0.11885 s for dipoles only over three alternating-order
repetitions. These times include sparse factorizations and exclude molecular
preparation. This is an 8.4% throughput increase for that workload, not a
measured full-furan speedup. For 144 channels and three dipole sources the
RHS count falls from 144 to three; factorization cost is unchanged.

The general saved-operator benchmark is
`scripts/benchmark_sparks_amplitudes.py`; numerical checks and measured
records are in `docs/reports/sparks_amplitude_optimization_20261107/`.

### Exact local-field and sparse-ordering optimizations

The outgoing sparse solver defaults to `permutation="MMD_AT_PLUS_A"` and
accepts `"COLAMD"`, `"MMD_ATA"` or `"NATURAL"` explicitly. This changes only
the elimination order, not the operator, retained entries or pivot tolerance.
The diagnostic record identifies the permutation. Real AO Coulomb grid
integrals use Hermitian shell-pair evaluation, and angular local potentials
use weighted matrix products with the unchanged quadrature and channel set.

On the saved water l=7 problem (5632 functions, 64 channels, three sources),
three alternating-order measurements at 0.30 hartree give a 19.2-fold
speedup for amplitudes-only/minimum-degree over full-result/COLAMD solves,
including factorization. Separate profiled single-geometry runs reduce
preparation from 20.6 to 10.7--11.3 s and total evaluation from 33.4 to
12.8 s. Those complete-geometry timings are individual profiles, not
repeated benchmark medians. Complex amplitudes retain the saved production
values within 1.2e-14 relative. The gains are workload-specific; the method
introduces no channel truncation, density fit or accuracy relaxation.

The benchmark accepts either a dense `--operators` archive or a
`--sparse-operators` directory, with separate reference and optimized
permutations. Records include numerical-library versions, thread settings,
source hashes, fill, backward residuals and interleaved timings.

The final optimized path also reproduces the saved 14049-function furan
centre, including nine compact atomic blocks and all three production
energies, within 3.59e-13 relative complex-amplitude norm. On MAC-STUDIO the
complete point takes 125.1 s (103.1 s preparation), one numerical thread.
This is an absolute timing and a cross-implementation numerical comparison,
not a cross-host speedup measurement. The extended regression has 62 passes.

### Streamed Coulomb contraction and repeated energy solves

`AOHartreePotential(..., contraction="auto")` contracts exact libcint integrals
inside a SPARKS C++ driver, one shell-pair/grid block at a time. This avoids the
full grid-by-AO-by-AO tensor. Real spherical and Cartesian AOs are supported;
only exactly zero density weights are skipped. The physical potential,
quadrature and integral engine are unchanged. The grid block respects
`workspace_bytes` (libcint scratch space and the output vector are additional).

The packaged C++ source is compiled once into the user cache using `CXX` or
`c++` and the installed PySCF/libcint headers. There is no TIRESIA dependency.
`contraction="direct"` requires the driver; `contraction="matrix"` explicitly
uses the full-matrix implementation. `auto` uses the matrix implementation if
the compiler or headers are unavailable. Compilation errors are reported.
`contraction_backend` identifies the selected path.

`SparseResolventBackend(..., reuse_structure=True)` prepares the union of
Hamiltonian, overlap and boundary-coupling entries once. Later energies update
values in that fixed structure and reuse the first column ordering. Numerical
factorization and row pivoting are performed anew at every energy. Diagnostics
report `structure_reused` and `ordering_reused`. Setting `reuse_structure=False`
restores independent sparse assembly. A boundary support exceeding four
million pair slots retains the independent path to avoid a dense graph cache.

Repeated one-thread MAC-STUDIO controls on existing molecular data give an
approximately 8% Coulomb-stage gain for the 103-AO furan density and a 5% total
solver gain over a three-energy, 5632-function water scan, including graph
preparation. These are comparisons with the preceding optimized implementation,
not speedups against a separate electronic-structure program.

### Matched removal-orbital comparisons

`LocalPhotoionizationCalculation.dipole_sources(orbital)` projects another real
spatial orbital into the already prepared central and atomic space, preserving
the field, H/S/B and channel conventions. `project_radial_sources` exposes the
same quadrature owner used by radial assembly; compact atomic contributions
reuse `with_multicentre_spatial_sources`.

`scripts/compare_sparks_orbital_sources.py` accepts a JSON specification of saved
Gaussian data and continuum settings. It compares correlated Dyson, canonical
unit-norm, and canonical norm-matched sources in a shared factorization at each
energy. Multiple removal channels may be supplied via `channel_indices`; poles
are indexed in ascending ionization energy. The reference occupied orbital is
chosen by normalized AO-metric overlap, or explicitly by `orbital_index`.
The correlated removal energy is held fixed in all source comparisons. The
report distinguishes shape from norm and checks sigma_Z=Z*sigma_1 and equal
canonical anisotropies. This command starts no electronic calculation.

`scripts/plot_sparks_energy_observables.py` plots saved band-energy or orbital
comparisons. Molecular labels and observed-energy intervals come from the
supplied data. No spectrum is energy-shifted or fitted to experiment.

### Multichannel sources without repeated AO evaluation

`AODysonOrbital(state, coefficients)` also accepts a real AO coefficient column
matrix. The same AO grid batch evaluates all columns, with individual norms
and phases preserved. `ProjectedDipoleSource` and `dipole_sources` return Cartesian
columns in orbital-major order, including every compact atomic contribution.

`SparseResolventBackend(..., source_transform=T)` represents requested sources
as `D @ T`, where `D` is the supplied source matrix. It solves only those
columns and reconstructs every requested complex amplitude with `T.T`.
The linear map is explicit: there is no numerical rank threshold or silent
source approximation. This is useful when several operators differ only by
known weights or normalizations. Diagnostics separate source count, actual
right-hand sides, matrix factorization and source solution costs.

The shared-field comparison defaults to `source_mode="batched"`; `"serial"`
retains independent orbital evaluations and explicit normalization columns
for verification. `benchmark_source_counts` and `benchmark_repetitions`
measure adding existing removal channels to one prepared field. Atomic
settings accept `element_overrides`, keyed by atomic number, to converge
heavy-atom supports independently of light atoms.

For multichannel reports, `plot_sparks_energy_observables.py` can overlay
figure-derived experimental strengths and anisotropies. A single total-
strength normalization at a declared photon energy is shared by all channels
within each dataset, retaining relative channel intensities. The optional
`extract_vector_plot_points.py` utility uses PyMuPDF and explicit linear/log
axis calibrations to recover circular symbols from vector PDF figures;
its output identifies the source figure, PDF hash and calibration. These
values are explicitly distinguished from an author's original numerical data.

Continuum integration rules can be varied independently of the trial space:
`continuum.angular_quadrature` supplies `polar_order` and `azimuthal_order`
(or an explicit supported Lebedev rule), while `atomic_blocks.angular_quadrature`
and its element overrides specify atomic integration. A central/atomic angular
basis comparison must retain the same integration rules; an integration check
retains the same basis. Larger supports can require finer atomic integration
for translated central/atomic cross integrals even at fixed local lmax.

### Positive compact overlap before propagation

`compact_overlap_diagnostics(operators, require_positive=True)` checks the
normalized local Schur complement of the central/atomic overlap. It uses
sparse central solves and a small dense local eigenproblem, never a global
eigensolve. Underresolved cross integrals can violate positivity even when
both separate diagonal blocks are positive. No direction is discarded and
no matrix entry is regularized. A failed check blocks local photoionization.

`scripts/check_sparks_continuum_overlap.py` applies the same basis and
quadrature settings without computing an electronic field. Its result can
be supplied as `overlap_qualification` in the source-comparison JSON to
reject an invalid space before field construction. The final assembled
calculation also checks its own overlap. Both central and atomic integration
must be refined independently of angular trial-space size.
