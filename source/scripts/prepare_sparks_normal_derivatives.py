"""Prepare a bounded normal-coordinate stencil from approved GDV templates."""

import argparse
import hashlib
import json
import socket
from pathlib import Path

import numpy as np
from matrix_gaussian.fchk import read_gaussian_fchk, lower_to_symmetric
from matrix_gaussian.modes import gaussian_cartesian_modes_to_mass_weighted
from matrix_gaussian.cartesian_template import write_gaussian_cartesian_template
from matrix_qm.normal_modes import mass_weighted_cartesian_hessian
from matrix_vpt2_vci.zaff_qff import dimensionless_normal_displacements


def prepare(fchk, templates, output, step, *, stencil_points=5, include_center=False,
            reflection_normal=None):
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Dimensionless step must be positive")
    if stencil_points not in (3, 5) or "dyson" not in templates:
        raise ValueError("Require a three/five-point stencil and a Dyson template")
    data = read_gaussian_fchk(fchk)
    modes = gaussian_cartesian_modes_to_mass_weighted(data.normal_modes, data.masses_amu)
    weighted = mass_weighted_cartesian_hessian(
        lower_to_symmetric(data.cartesian_hessian_lower), data.masses_amu
    )
    projected = modes @ weighted @ modes.T
    if np.max(np.abs(projected - np.diag(np.diag(projected)))) > 1e-6:
        raise ValueError("Stored Gaussian modes do not diagonalise the supplied Hessian")
    directions = dimensionless_normal_displacements(
        data.harmonic_frequencies_cm,
        np.diag(projected),
        modes.reshape(-1, len(data.atomic_numbers), 3),
        data.masses_amu,
    )
    output.mkdir(parents=True, exist_ok=True)
    points = []
    for mode in range(modes.shape[0]):
        offsets = [-step, -step / 2, step / 2, step] if stencil_points == 5 else [-step, step]
        for index, q in enumerate(offsets):
            name = f"mode{mode + 1:02d}_point{index:02d}"
            geometry = data.cartesian_coordinates_bohr + q * directions[:, mode].reshape(-1, 3)
            files = {}
            for label, template in templates.items():
                target = output / name / label / "job.gjf"
                if target.exists():
                    raise FileExistsError(f"Preserve existing prepared job: {target}")
                write_gaussian_cartesian_template(template, target, geometry * 0.529177210903)
                files[label] = str(target.relative_to(output))
            if "density" not in files:
                files["density"] = files["dyson"]
            points.append(
                dict(
                    name=name, mode=mode + 1, q=q, coordinates_bohr=geometry.tolist(), inputs=files
                )
            )
    center_inputs = {}
    if include_center:
        for label, template in templates.items():
            target = output / "center" / label / "job.gjf"
            if target.exists():
                raise FileExistsError(f"Preserve existing prepared job: {target}")
            write_gaussian_cartesian_template(template, target, data.cartesian_coordinates_bohr * 0.529177210903)
            center_inputs[label] = str(target.relative_to(output))
        if "density" not in center_inputs:
            center_inputs["density"] = center_inputs["dyson"]
    report = dict(
        schema="matrix.sparks.normal_derivative_plan.v1",
        host=socket.getfqdn(),
        step=step,
        coordinate_units="dimensionless",
        atomic_numbers=data.atomic_numbers.tolist(),
        center_coordinates_bohr=data.cartesian_coordinates_bohr.tolist(),
        frequencies_cm=data.harmonic_frequencies_cm.tolist(),
        directions_bohr=directions.tolist(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [fchk, *templates.values()]
        },
        points=points,
        job_count=(len(points) + int(include_center)) * len(templates),
        stencil_points=stencil_points,
        density_from_dyson_checkpoint="density" not in templates,
        center_reused=not include_center,
        center_inputs=center_inputs,
    )
    if reflection_normal is not None:
        normal = np.asarray(reflection_normal, float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
            raise ValueError("A finite nonzero plane normal is required")
        normal = normal / np.linalg.norm(normal)
        reflection = np.eye(3) - 2*np.outer(normal, normal)
        # Keep the scattering origin fixed: translations require a different
        # angular-channel transformation and cannot be inferred here.
        if np.max(abs(data.cartesian_coordinates_bohr @ normal)) > 1e-7:
            raise ValueError("Reference atoms must lie in the declared plane through the origin")
        odd = []
        for mode in range(modes.shape[0]):
            direction = directions[:, mode].reshape(-1, 3)
            if np.allclose(direction @ reflection, -direction, rtol=0, atol=1e-8):
                odd.append(mode + 1)
        by_mode_q = {(row['mode'], row['q']): row for row in points}
        for row in points:
            if row['mode'] in odd and row['q'] < 0:
                partner = by_mode_q[(row['mode'], -row['q'])]
                if not np.allclose(np.asarray(partner['coordinates_bohr']) @ reflection,
                                   row['coordinates_bohr'], atol=1e-7, rtol=0):
                    raise ValueError("Reflected displacement does not reproduce its partner")
                row['symmetry_source'] = partner['name']
                row['orthogonal_transform'] = reflection.tolist()
        unique = [row for row in points if 'symmetry_source' not in row]
        report.update(reflection_odd_modes=odd, electronic_points=[row['name'] for row in unique],
                      job_count=(len(unique)+int(include_center))*len(templates),
                      symmetry_reconstruction='Transform complex channel dipoles and include the reference Dyson character; copying scalar intensities is insufficient')
    (output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fchk", type=Path, required=True)
    p.add_argument("--dyson-template", type=Path, required=True)
    p.add_argument("--density-template", type=Path,
                   help="Omit only when the Dyson job also supplies the intended density")
    p.add_argument("--stencil-points", type=int, choices=(3, 5), default=5)
    p.add_argument("--include-center", action="store_true")
    p.add_argument("--reflection-normal", type=float, nargs=3)
    p.add_argument("--step", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from matrix_sparks.execution import require_compute_host
    require_compute_host()
    templates = dict(dyson=a.dyson_template)
    if a.density_template is not None:
        templates["density"] = a.density_template
    r = prepare(a.fchk, templates, a.output, a.step,
                stencil_points=a.stencil_points, include_center=a.include_center,
                reflection_normal=a.reflection_normal)
    print(json.dumps(dict(job_count=r["job_count"], points=len(r["points"]), step=a.step)))
