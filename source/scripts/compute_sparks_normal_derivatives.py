"""Evaluate and checkpoint an explicit normal-coordinate continuum stencil."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import socket
from time import perf_counter

import numpy as np
from matrix_apoc import (
    electronic_state_from_gaussian_fchk,
    read_gdv_ept_dysons,
    reorient_dyson_orbital,
    reorient_electronic_state,
    follow_dyson_orbital,
)
from matrix_sparks.continuum import (
    RealSphericalChannels,
    LocalPhotoionizationCalculation,
    AODensityGradient,
    LB94Potential,
    CoulombTailPotential,
    symmetric_amplitude_derivative,
)
from matrix_sparks.execution import require_compute_host


def _load_unique_dysons(log, checkpoint, method, frozen_core):
    state = electronic_state_from_gaussian_fchk(checkpoint)
    return read_gdv_ept_dysons(log, state, method=method, frozen_core=frozen_core)


def _fingerprint(files, settings):
    return hashlib.sha256(
        json.dumps(
            dict(
                settings=settings,
                files=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files],
                checkpoint_version=2,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def evaluate_point(task):
    require_compute_host()
    start = perf_counter()
    settings = task["settings"]
    name = task["name"]
    folder = Path(task["output"])
    target = folder / f"{name}.npz"
    metadata = folder / f"{name}.json"
    files = [
        task[k]
        for k in [
            "reference_log",
            "reference_checkpoint",
            "dyson_log",
            "dyson_checkpoint",
            "density_checkpoint",
        ]
    ]
    fingerprint = _fingerprint(
        files, dict(settings=settings, coordinates=task["coordinates"], name=name)
    )
    if target.exists() and metadata.exists():
        report = json.loads(metadata.read_text())
        if report["fingerprint"] != fingerprint:
            legacy = hashlib.sha256(
                json.dumps(
                    dict(
                        settings=dict(
                            settings=settings, coordinates=task["coordinates"], name=name
                        ),
                        files={
                            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files
                        },
                    ),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if "checkpoint_version" in report or report["fingerprint"] != legacy:
                raise ValueError(
                    f"Preserve incompatible checkpoint {target}; choose a new output folder"
                )
            report.update(
                legacy_fingerprint=report["fingerprint"],
                fingerprint=fingerprint,
                checkpoint_version=2,
            )
            metadata.write_text(json.dumps(report, indent=2) + "\n")
        with np.load(target) as arrays:
            if not np.all(np.isfinite(arrays["amplitudes"])):
                raise ValueError("Invalid amplitude checkpoint")
        return dict(name=name, reused=True)
    references = _load_unique_dysons(
        task["reference_log"],
        task["reference_checkpoint"],
        settings["method"],
        settings["frozen_core"],
    )
    if len(references) != 1:
        raise ValueError("Declare an isolated reference channel before taking derivatives")
    reference, _ = reorient_dyson_orbital(references[0], settings["center_coordinates_bohr"])
    candidates = _load_unique_dysons(
        task["dyson_log"], task["dyson_checkpoint"], settings["method"], settings["frozen_core"]
    )
    candidates = [reorient_dyson_orbital(d, task["coordinates"])[0] for d in candidates]
    dyson, match = follow_dyson_orbital(reference, candidates)
    density = electronic_state_from_gaussian_fchk(task["density_checkpoint"])
    density, alignment = reorient_electronic_state(density, task["coordinates"])
    electrons = sum(
        float(np.trace(density.density_ao(i) @ density.overlap_ao))
        for i in range(density.spin_channels)
    )
    expected = int(np.sum(density.atomic_numbers)) - density.charge
    if abs(electrons - expected) > 2e-7:
        raise ValueError("Density trace fails equilibrium/current-density qualification")
    channels = RealSphericalChannels(
        [(l, m) for l in range(settings["lmax"] + 1) for m in range(-l, l + 1)]
    )
    edges = np.unique(
        np.r_[
            np.linspace(0, settings["radius"], settings["intervals"] + 1),
            np.linalg.norm(settings["center_coordinates_bohr"], axis=1),
        ]
    )
    cuts = np.linalg.norm(density.coordinates_bohr, axis=1)
    xc = CoulombTailPotential(
        LB94Potential(AODensityGradient(density)),
        inner_radius=settings["xc_tail"][0],
        outer_radius=settings["xc_tail"][1],
        charge=1.0,
    )
    calculation = LocalPhotoionizationCalculation(
        density,
        dyson,
        channels,
        edges,
        xc,
        asymptotic_charge=float(np.sum(density.atomic_numbers)) - electrons + 1,
        outer_radius=settings["outer_radius"],
        dipole_boundary=True,
        quadrature_breakpoints=cuts,
        quadrature=settings["quadrature"],
        atomic_blocks=atomic_blocks(settings, density.coordinates_bohr),
    )
    amplitudes = calculation.amplitudes(settings["energies"])
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, amplitudes=amplitudes, energies=settings["energies"], channels=channels.labels
    )
    report = dict(
        name=name,
        fingerprint=fingerprint,
        checkpoint_version=2,
        host=socket.getfqdn(),
        q=task["q"],
        mode=task["mode"],
        elapsed_seconds=perf_counter() - start,
        preparation_seconds=calculation.preparation_seconds,
        overlap_preflight_seconds=calculation.overlap_preflight_seconds,
        phase=match.phase,
        normalized_dyson_overlap=match.normalized_overlap,
        pole_strength=dyson.pole_strength,
        ionization_energy_ev=dyson.ionization_energy_ev,
        density_trace=electrons,
        geometry_error_bohr=alignment.maximum_geometry_error_bohr,
        operator_dimension=calculation.operators.hamiltonian.shape[0],
        quadrature_nodes=len(calculation.radial.nodes),
        tail_midpoint_error_hartree=calculation.tail_midpoint_error_hartree,
        compact_atomic_blocks=len(density.coordinates_bohr) if settings.get("atomic_radius") is not None else 0,
        overlap_diagnostics=calculation.overlap_diagnostics,
    )
    metadata.write_text(json.dumps(report, indent=2) + "\n")
    return report


def atomic_blocks(settings, coordinates):
    """Reuse compact atom-centred enrichment for moving-geometry derivatives."""
    radius = settings.get("atomic_radius")
    if radius is None:
        return ()
    lmax = settings["atomic_lmax"]
    intervals = settings["atomic_intervals"]
    if not np.isfinite(radius) or radius <= 0 or lmax < 0 or intervals < 1:
        raise ValueError("Positive atomic support and valid local basis dimensions required")
    from matrix_sparks.continuum.multicentre import AtomicSplineBlock
    channels = RealSphericalChannels(
        [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)],
        quadrature="lebedev", lebedev_order=settings.get("atomic_lebedev_order", max(17, 4 * lmax + 1)),
    )
    return tuple(AtomicSplineBlock(position, np.linspace(0, radius, intervals + 1), channels)
                 for position in coordinates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--reference-log", type=Path, required=True)
    p.add_argument("--reference-checkpoint", type=Path, required=True)
    p.add_argument("--reference-density", type=Path, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--frozen-core", type=int, required=True)
    p.add_argument('--dyson-export-name', default='job.log',
                   help='Signed complete export name, e.g. dyson.log after saved-vector recovery')
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--coarse-only",
        action="store_true",
        help="Use +/-h for a bounded basis check; no step-sensitivity estimate",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--lmax", type=int, default=5)
    p.add_argument("--atomic-radius", type=float)
    p.add_argument("--atomic-lmax", type=int, default=2)
    p.add_argument("--atomic-intervals", type=int, default=4)
    p.add_argument("--atomic-lebedev-order", type=int,
                   help="Explicit atom-centred integration order, stored in checkpoint identity")
    p.add_argument("--modes", type=int, nargs="+", help="Optional one-based subset for numerical basis checks")
    p.add_argument("--radius", type=float, default=20.0)
    p.add_argument("--intervals", type=int, default=80)
    p.add_argument("--outer-radius", type=float, default=120.0)
    p.add_argument("--xc-tail", type=float, nargs=2, required=True)
    p.add_argument("--quadrature", type=int, default=12)
    p.add_argument("--energies", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    a = p.parse_args()
    require_compute_host()
    if a.workers < 1 or a.workers > 4:
        raise ValueError("Use one to four bounded continuum workers")
    start = perf_counter()
    plan = json.loads(a.plan.read_text())
    root = a.plan.parent
    modes = {point["mode"] for point in plan["points"]}
    if a.modes is not None:
        if not set(a.modes) <= modes or len(set(a.modes)) != len(a.modes):
            raise ValueError("Select distinct existing one-based modes")
        modes = set(a.modes)
    settings = dict(
        method=a.method,
        frozen_core=a.frozen_core,
        lmax=a.lmax,
        radius=a.radius,
        intervals=a.intervals,
        outer_radius=a.outer_radius,
        xc_tail=a.xc_tail,
        energies=a.energies,
        quadrature=a.quadrature,
        center_coordinates_bohr=plan["center_coordinates_bohr"],
    )
    if a.atomic_radius is not None:
        settings.update(atomic_radius=a.atomic_radius, atomic_lmax=a.atomic_lmax,
                        atomic_intervals=a.atomic_intervals)
        if a.atomic_lebedev_order is not None:
            settings["atomic_lebedev_order"] = a.atomic_lebedev_order
        atomic_blocks(settings, plan["center_coordinates_bohr"])
    common = dict(
        settings=settings,
        output=str(a.output.resolve()),
        reference_log=str(a.reference_log.resolve()),
        reference_checkpoint=str(a.reference_checkpoint.resolve()),
    )
    tasks = [
        dict(
            common,
            name="center",
            mode=0,
            q=0.0,
            coordinates=plan["center_coordinates_bohr"],
            dyson_log=str(a.reference_log.resolve()),
            dyson_checkpoint=str(a.reference_checkpoint.resolve()),
            density_checkpoint=str(a.reference_density.resolve()),
        )
    ]
    selected_points = [
        point for point in plan["points"] if point["mode"] in modes
        and (not a.coarse_only or abs(point["q"]) == plan["step"])
    ]
    for point in selected_points:
        if 'symmetry_source' in point:
            continue
        dyson_dir = (root / point["inputs"]["dyson"]).parent
        density_dir = (root / point["inputs"]["density"]).parent
        checkpoints = [list(folder.glob("*.fchk")) for folder in [dyson_dir, density_dir]]
        if any(len(paths) != 1 for paths in checkpoints):
            raise ValueError("Require exactly one completed FCHK per electronic job")
        tasks.append(
            dict(
                common,
                name=point["name"],
                mode=point["mode"],
                q=point["q"],
                coordinates=point["coordinates_bohr"],
                dyson_log=str((dyson_dir / a.dyson_export_name).resolve()),
                dyson_checkpoint=str(checkpoints[0][0].resolve()),
                density_checkpoint=str(checkpoints[1][0].resolve()),
            )
        )
    a.output.mkdir(parents=True, exist_ok=True)
    executions = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context("spawn")) as executor:
        for report in executor.map(evaluate_point, tasks):
            executions.append(report)
            print(json.dumps(report), flush=True)
    center = np.load(a.output / "center.npz")["amplitudes"]
    # A reflected electronic point is reconstructed in amplitude space, with
    # the isolated reference Dyson's character fixing its phase. In particular
    # an odd out-of-plane derivative must not be erased by copying strengths.
    derived = [point for point in selected_points if 'symmetry_source' in point]
    if derived:
        from matrix_sparks.continuum.molecular import AODysonOrbital
        (reference,) = _load_unique_dysons(a.reference_log, a.reference_checkpoint,
                                         a.method, a.frozen_core)
        reference, _ = reorient_dyson_orbital(reference, plan['center_coordinates_bohr'])
        orbital = AODysonOrbital.from_apoc(reference)
        channels = RealSphericalChannels([(l,m) for l in range(a.lmax+1) for m in range(-l,l+1)])
        probes = (np.array([.5, 1., 2., 3., 5., 8.])[:,None,None]*channels.directions).reshape(-1,3)
        values = orbital(probes)
        for point in derived:
            operation = np.asarray(point['orthogonal_transform'])
            rotated = orbital(probes @ operation)
            overlap = float(values @ rotated)
            character = 1 if overlap >= 0 else -1
            error = float(np.linalg.norm(rotated-character*values)/np.linalg.norm(values))
            if error > 1e-6:
                raise ValueError('Reference Dyson is not an isolated eigenfunction of the declared symmetry')
            source = a.output / (point['symmetry_source']+'.npz')
            with np.load(source) as data:
                amplitudes = channels.transform_dipole_amplitudes(
                    data['amplitudes'], operation, dyson_character=character)
            identity = hashlib.sha256(source.read_bytes()+operation.tobytes()+str(character).encode()).hexdigest()
            target = a.output / (point['name']+'.npz')
            metadata = target.with_suffix('.json')
            if target.exists():
                if not metadata.exists() or json.loads(metadata.read_text()).get('fingerprint') != identity:
                    raise ValueError('Preserve incompatible symmetry checkpoint')
            else:
                np.savez_compressed(target, amplitudes=amplitudes, energies=a.energies, channels=channels.labels)
                metadata.write_text(json.dumps(dict(fingerprint=identity,
                    reconstructed_from=point['symmetry_source'], dyson_character=character,
                    dyson_character_relative_error=error, orthogonal_transform=operation.tolist()),indent=2)+'\n')
    rows = []
    for mode in sorted(modes):
        points = [point for point in selected_points if point["mode"] == mode]
        displacements = [0.0, *[point["q"] for point in points]]
        arrays = [
            center,
            *[np.load(a.output / f"{point['name']}.npz")["amplitudes"] for point in points],
        ]
        result = symmetric_amplitude_derivative(displacements, arrays)
        arrays_to_save = dict(
            first=result.first,
            second=result.second,
            coarse=result.coarse_first,
            energies=a.energies,
        )
        if result.fine_first is not None:
            arrays_to_save["fine"] = result.fine_first
        np.savez_compressed(a.output / f"derivative-mode{mode:02d}.npz", **arrays_to_save)
        energy_rows = []
        for index, energy in enumerate(a.energies):
            norm = float(np.linalg.norm(result.first[index]))
            difference = (None if result.fine_first is None else float(
                np.linalg.norm(result.fine_first[index] - result.coarse_first[index])
            ))
            intensity = float(np.sum(abs(center[index]) ** 2))
            slope = float(2 * np.vdot(center[index], result.first[index]).real)
            energy_rows.append(
                dict(
                    kinetic_energy_hartree=energy,
                    first_derivative_norm=norm,
                    relative_step_difference=None
                    if norm == 0 or difference is None
                    else difference / norm,
                    relative_strength_derivative=slope / intensity,
                )
            )
        rows.append(
            dict(
                mode=mode,
                frequency_cm=plan["frequencies_cm"][mode - 1],
                relative_step_difference=result.relative_step_difference,
                by_energy=energy_rows,
            )
        )
    report = dict(
        schema="matrix.sparks.normal_amplitude_derivatives.v1",
        host=socket.getfqdn(),
        elapsed_seconds=perf_counter() - start,
        workers=a.workers,
        stencil="central +/-h"
        if a.coarse_only
        else "central h and h/2 with fourth-order derivative",
        reused_points=sum(bool(r.get("reused", False)) for r in executions),
        calculated_points=sum(not r.get("reused", False) for r in executions),
        symmetry_reconstructed_points=len(derived),
        settings=settings,
        results=rows,
        status="phase-consistent local continuum amplitude derivatives, not a complete FCHT spectrum",
    )
    (a.output / "derivatives.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
