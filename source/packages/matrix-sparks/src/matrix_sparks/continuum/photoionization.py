"""Shared local photoionization preparation for spectra and geometry derivatives."""

from time import perf_counter

import numpy as np

from .basis import RadialBasis
from .coulomb import coulomb_boundary, propagate_boundary
from .dipole_boundary import DipoleCoulombBoundary
from .molecular import (
    AODysonOrbital,
    AOHartreePotential,
    NuclearPotential,
    ProjectedDipoleSource,
    ProjectedLocalPotential,
)
from .radial import assemble_sparse_radial, project_radial_sources
from .resolvent import SparseResolventBackend
from .tail import InterpolatedRadialPotential


class LocalPhotoionizationCalculation:
    """Prepare one local electronic field and signed Dyson source once.

    Electronic states already share the caller's declared molecular frame.
    XC is an explicit spatial callback; no functional or tail is selected here.
    The asymptotic charge and dipole reference are explicit model inputs.
    All heavy numerical operations reuse the existing SPARKS components.
    """

    def __init__(
        self,
        density,
        dyson,
        channels,
        breakpoints,
        xc_potential,
        *,
        asymptotic_charge,
        outer_radius,
        dipole_boundary=False,
        degree=7,
        quadrature=12,
        quadrature_breakpoints=None,
        tail_grid_points=129,
        atomic_blocks=(),
        atomic_transform=None,
    ):
        start = perf_counter()
        if not np.array_equal(
            density.atomic_numbers, dyson.state.atomic_numbers
        ) or not np.allclose(
            density.coordinates_bohr, dyson.state.coordinates_bohr, atol=1e-7, rtol=0
        ):
            raise ValueError("Density and Dyson must share the same ordered molecular frame")
        if density.charge != dyson.state.charge:
            raise ValueError("Density and Dyson parent charges differ")
        self.channels = channels
        self.dyson = dyson
        self.dyson_orbital = AODysonOrbital.from_apoc(dyson)
        self.radial = RadialBasis(
            breakpoints,
            degree=degree,
            quadrature=quadrature,
            quadrature_breakpoints=quadrature_breakpoints,
        )
        self.radius = self.radial.radius
        self.outer_radius = float(outer_radius)
        if (
            not np.isfinite(self.outer_radius)
            or self.outer_radius < self.radius
            or np.max(np.linalg.norm(density.coordinates_bohr, axis=1)) >= self.radius
        ):
            raise ValueError("Require enclosing inner sphere and outer>=inner radius")
        if (
            not np.isfinite(asymptotic_charge)
            or int(tail_grid_points) != tail_grid_points
            or tail_grid_points < 4
        ):
            raise ValueError("Require finite charge and at least four tail samples")
        self.charge = float(asymptotic_charge)
        if atomic_transform is not None and not atomic_blocks:
            raise ValueError('Atomic transform requires explicit compact atomic blocks')
        self.overlap_preflight_seconds = 0.0
        if atomic_blocks:
            from .multicentre import assemble_multicentre, compact_overlap_diagnostics
            tick = perf_counter()
            central_overlap = assemble_sparse_radial(
                breakpoints, channels.angular_momenta,
                lambda r: np.zeros((len(r), len(channels.labels), len(channels.labels))),
                degree=degree, quadrature=quadrature, radial_basis=self.radial)
            trial_overlap = assemble_multicentre(
                central_overlap, channels, atomic_blocks,
                density.atomic_numbers, density.coordinates_bohr,
                local_transform=atomic_transform, overlap_only=True)
            compact_overlap_diagnostics(trial_overlap, require_positive=True)
            self.overlap_preflight_seconds = perf_counter() - tick
            del trial_overlap, central_overlap
        self.hartree = AOHartreePotential(density)
        vn = NuclearPotential(channels, density.atomic_numbers, density.coordinates_bohr)

        def electronic_potential(points):
            xc = np.asarray(xc_potential(points))
            if (np.iscomplexobj(xc) or xc.shape != (len(points),)
                    or not np.isfinite(xc).all()):
                raise ValueError("Local potential must return finite real values at all points")
            return self.hartree(points) + xc

        ve = ProjectedLocalPotential(channels, electronic_potential)

        def potential(r):
            # Projection is linear: share the same angular contraction for
            # Coulomb and XC contributions rather than projecting each twice.
            return vn(r) + ve(r)

        self._potential = potential
        self.boundary_provider = None
        self.molecular_dipole_au = None
        if dipole_boundary:
            total = sum(density.density_ao(i) for i in range(density.spin_channels))
            position = self.hartree.molecule.intor_symmetric("int1e_r", comp=3)
            self.molecular_dipole_au = (
                density.atomic_numbers @ density.coordinates_bohr
                - np.einsum("xij,ji->x", position, total)
            )
            dipole_coupling = -np.einsum(
                "amb,m->ab", channels.dipole_couplings(channels), self.molecular_dipole_au
            )
            self.boundary_provider = DipoleCoulombBoundary(
                channels.angular_momenta, dipole_coupling, self.charge
            )
        self.sampled_tail = None
        self.tail_midpoint_error_hartree = None
        if self.outer_radius > self.radius:
            radii = np.geomspace(self.radius, self.outer_radius, int(tail_grid_points))
            self.sampled_tail = InterpolatedRadialPotential(radii, potential(radii))
            midpoints = np.sqrt(radii[:-1] * radii[1:])
            self.tail_midpoint_error_hartree = float(
                np.max(abs(self.sampled_tail(midpoints) - potential(midpoints)))
            )
        values = potential(self.radial.nodes)
        source = ProjectedDipoleSource(channels, self.dyson_orbital)
        self.operators = assemble_sparse_radial(
            breakpoints,
            channels.angular_momenta,
            lambda r: values,
            degree=degree,
            quadrature=quadrature,
            radial_basis=self.radial,
            source=source,
        )
        if atomic_blocks:
            from .multicentre import assemble_multicentre, with_multicentre_spatial_sources
            central_sources = self.operators.sources
            enriched = assemble_multicentre(
                self.operators, channels, atomic_blocks,
                density.atomic_numbers, density.coordinates_bohr,
                local_transform=atomic_transform,
                electronic_potential=electronic_potential)
            self.operators = with_multicentre_spatial_sources(
                enriched, lambda points: self.dyson_orbital(points)[:,None]*points,
                central_sources=central_sources)
        self.overlap_diagnostics = None
        if atomic_blocks:
            from .multicentre import compact_overlap_diagnostics
            self.overlap_diagnostics = compact_overlap_diagnostics(self.operators,require_positive=True)
        self.solver = SparseResolventBackend(
            self.operators.hamiltonian,
            self.operators.overlap,
            self.operators.boundary,
            self.operators.sources,
        )
        self.preparation_seconds = perf_counter() - start

    def dipole_sources(self, orbital):
        """Project real orbital(s), preserving H/S/B; batch columns are orbital-major."""
        central = project_radial_sources(self.radial,len(self.channels.labels),
                                        ProjectedDipoleSource(self.channels,orbital))
        if hasattr(self.operators.basis,"central"):
            from .multicentre import with_multicentre_spatial_sources
            def spatial(points):
                values = np.asarray(orbital(points))
                if values.ndim == 1:
                    values = values[:,None]
                return (values[:,:,None]*points[:,None,:]).reshape(len(points),-1)
            return with_multicentre_spatial_sources(
                self.operators,spatial,central_sources=central).sources
        return central

    def boundary_waves(self, energy):
        """Return the same propagated exterior used by every scattering solve."""
        waves = (
            coulomb_boundary(energy, self.outer_radius, self.channels.angular_momenta, self.charge)
            if self.boundary_provider is None
            else self.boundary_provider(energy, self.outer_radius)
        )
        if self.sampled_tail is not None:
            waves = propagate_boundary(
                energy,
                self.radius,
                self.outer_radius,
                self.channels.angular_momenta,
                self.sampled_tail,
                waves,
            )
        return waves

    def evaluate(self, energy):
        return self.solver.evaluate(energy, self.boundary_waves(energy))

    def amplitudes(self, energies):
        """Return complex (energy, Cartesian source, angular channel) amplitudes."""
        values = np.asarray(energies, float)
        if (
            values.ndim != 1
            or not len(values)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError("Require a nonempty array of positive open energies")
        evaluate_amplitudes = getattr(self.solver, "evaluate_amplitudes", None)
        if evaluate_amplitudes is None:
            return np.asarray([self.evaluate(e).amplitudes for e in values])
        return np.asarray([
            evaluate_amplitudes(e, self.boundary_waves(e)) for e in values
        ])
