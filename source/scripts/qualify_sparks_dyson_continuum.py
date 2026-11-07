"""Bounded central-continuum pilot using APOC Dyson and an explicit LB94 density."""

import argparse, json, socket, time
from pathlib import Path
import numpy as np
from matrix_apoc import electronic_state_from_gaussian_fchk, read_gdv_ept_dysons
from matrix_sparks.continuum import (
    AODensityGradient,
    LB94Potential,
    CoulombTailPotential,
    RealSphericalChannels,
    LocalPhotoionizationCalculation,
)
from matrix_sparks.execution import require_compute_host


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--density", type=Path, required=True)
    p.add_argument("--frozen-core", type=int, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--lmax", type=int, default=3)
    p.add_argument("--radius", type=float, default=20.0)
    p.add_argument("--intervals", type=int, default=80)
    p.add_argument("--degree", type=int, default=7)
    p.add_argument("--quadrature", type=int, default=12)
    p.add_argument("--polar-order", type=int)
    p.add_argument("--azimuthal-order", type=int)
    p.add_argument('--atomic-radius', type=float)
    p.add_argument('--atomic-lmax', type=int, default=2)
    p.add_argument('--atomic-intervals', type=int, default=4)
    p.add_argument(
        "--xc-tail",
        type=float,
        nargs=2,
        metavar=("INNER", "OUTER"),
        help="Explicit C2 XC splice to -1/r; changes the outer local model",
    )
    p.add_argument(
        "--outer-radius",
        type=float,
        help="Propagate existing coupled boundary waves inward from this radius",
    )
    p.add_argument(
        "--dipole-boundary",
        action="store_true",
        help="Include the actual molecular dipole in the asymptotic reference",
    )
    p.add_argument("--tail-grid-points", type=int, default=129)
    p.add_argument('--energies',type=float,nargs='+',default=[.05,.1,.2,.5,1.],
                   help='Explicit positive electron energies; compute only the requested spectral range')
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require_compute_host()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.lmax < 0 or a.radius <= 0 or a.intervals < 1 or a.degree < 2 or a.quadrature < a.degree + 1:
        raise ValueError("Invalid angular/radial pilot dimensions")
    if not np.isfinite(a.energies).all() or min(a.energies)<=0 or np.any(np.diff(a.energies)<=0):
        raise ValueError('Increasing positive electron energies required')
    outer_radius = a.radius if a.outer_radius is None else a.outer_radius
    if not np.isfinite(outer_radius) or outer_radius < a.radius or a.tail_grid_points < 4:
        raise ValueError("Invalid outer radius or tail interpolation grid")
    t0 = time.perf_counter()
    s = electronic_state_from_gaussian_fchk(a.checkpoint)
    (d,) = read_gdv_ept_dysons(a.log, s, frozen_core=a.frozen_core, method=a.method)
    density = electronic_state_from_gaussian_fchk(a.density)
    if not np.array_equal(density.atomic_numbers, s.atomic_numbers) or not np.allclose(
        density.coordinates_bohr, s.coordinates_bohr, rtol=0, atol=1e-7
    ):
        raise ValueError("Density and Dyson must share a molecular frame")
    if density.charge != s.charge or np.max(np.linalg.norm(s.coordinates_bohr, axis=1)) >= a.radius:
        raise ValueError("Charge mismatch or matching sphere inside molecule")
    channels = RealSphericalChannels(
        [(l, m) for l in range(a.lmax + 1) for m in range(-l, l + 1)],
        polar_order=a.polar_order, azimuthal_order=a.azimuthal_order,
    )
    edges = np.unique(
        np.r_[np.linspace(0, a.radius, a.intervals + 1), np.linalg.norm(s.coordinates_bohr, axis=1)]
    )
    xc = LB94Potential(AODensityGradient(density))
    if a.xc_tail is not None:
        if a.xc_tail[1] > a.radius:
            raise ValueError("XC splice must finish inside the inner matching sphere")
        xc = CoulombTailPotential(
            xc, inner_radius=a.xc_tail[0], outer_radius=a.xc_tail[1], charge=1.0
        )
    if a.dipole_boundary and a.xc_tail is None:
        raise ValueError("Dipole reference requires the explicit XC Coulomb tail")
    actual_electrons = sum(
        float(np.trace(density.density_ao(i) @ density.overlap_ao))
        for i in range(density.spin_channels)
    )
    charge = (
        float(np.sum(density.atomic_numbers)) - actual_electrons + 1
        if a.xc_tail is not None
        else density.charge + 1
    )
    atoms = ()
    if a.atomic_radius is not None:
        from matrix_sparks.continuum.multicentre import AtomicSplineBlock
        if a.atomic_radius <= 0 or a.atomic_lmax < 0 or a.atomic_intervals < 1:
            raise ValueError('Positive compact atomic support and valid dimensions required')
        local_channels = RealSphericalChannels(
            [(l,m) for l in range(a.atomic_lmax+1) for m in range(-l,l+1)],
            quadrature='lebedev', lebedev_order=max(17,4*a.atomic_lmax+1))
        atoms = tuple(AtomicSplineBlock(position, np.linspace(0,a.atomic_radius,a.atomic_intervals+1),
                                       local_channels) for position in density.coordinates_bohr)
    calculation = LocalPhotoionizationCalculation(
        density,
        d,
        channels,
        edges,
        xc,
        asymptotic_charge=charge,
        outer_radius=outer_radius,
        dipole_boundary=a.dipole_boundary,
        tail_grid_points=a.tail_grid_points,
        atomic_blocks=atoms,
        degree=a.degree,
        quadrature=a.quadrature,
    )
    op = calculation.operators
    molecular_dipole = calculation.molecular_dipole_au
    boundary_provider = calculation.boundary_provider
    rows = []
    for energy in a.energies:
        result = calculation.evaluate(energy)
        amplitude = result.amplitudes
        assert np.all(np.isfinite(amplitude))
        rows.append(
            dict(
                kinetic_energy_hartree=energy,
                amplitude_real=amplitude.real.tolist(),
                amplitude_imag=amplitude.imag.tolist(),
                sum_squared_amplitudes=float(np.sum(abs(amplitude) ** 2)),
            )
        )
    report = dict(
        host=socket.getfqdn(),
        elapsed_seconds=time.perf_counter() - t0,
        source=f"complete {a.method} Dyson imported through APOC",
        density=str(a.density),
        potential="exact AO Hartree + nuclear + LB94 XC",
        xc_tail_interval=a.xc_tail,
        boundary_charge=charge,
        asymptotic_reference="Coulomb"
        if boundary_provider is None
        else "Coulomb plus molecular dipole",
        molecular_dipole_au=None if molecular_dipole is None else molecular_dipole.tolist(),
        minimum_effective_centrifugal_eigenvalue=None
        if boundary_provider is None
        else float(boundary_provider.eigenvalues.min()),
        outer_radius_bohr=outer_radius,
        tail_grid_points=a.tail_grid_points if calculation.sampled_tail is not None else None,
        tail_midpoint_interpolation_error_hartree=calculation.tail_midpoint_error_hartree,
        lmax=a.lmax,
        rmax_bohr=a.radius,
        radial_intervals=len(edges) - 1,
        degree=a.degree,
        quadrature=a.quadrature,
        angular_integration_points=len(channels.weights),
        polar_order=a.polar_order if a.polar_order is not None else 2*a.lmax+2,
        azimuthal_order=a.azimuthal_order if a.azimuthal_order is not None else 4*a.lmax+3,
        operator_dimension=op.hamiltonian.shape[0],
        electron_number=actual_electrons,
        pole_strength=d.pole_strength,
        compact_atomic_blocks=len(atoms),
        atomic_support_radius_bohr=a.atomic_radius,
        atomic_lmax=a.atomic_lmax if atoms else None,
        status="finite central continuum amplitudes, NOT converged cross sections or FCHT",
        results=rows,
    )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
