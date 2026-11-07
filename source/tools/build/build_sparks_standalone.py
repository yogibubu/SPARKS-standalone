#!/usr/bin/env python3
"""Export committed SPARKS sources and build a platform-specific offline archive."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib

from build_smith_release import _requirement_name, _stage_project

ROOTS = (
    "matrix-sparks",
    "matrix-apoc",
    "matrix-pyscf",
    "matrix-trinity",
    "matrix-gaussian",
    "matrix-orca",
    "matrix-dvr",
    "matrix-vpt2-vci",
    "matrix-zaff",
)
TESTS = (
    "test_sparks_response.py",
    "test_sparks_cli.py",
    "test_sparks_response_optimization.py",
    "test_sparks_symmetry_response.py",
    "test_sparks_atomic_sources.py",
    "test_sparks_molecular_response.py",
    "test_saved_ept_export.py",
    "test_gaussian_normal_field_gauge.py",
    "test_sparks_polarization.py",
    "test_sparks_photoelectron.py",
    "test_sparks_angular_observables.py",
    "test_sparks_execution.py",
    "test_sparks_amplitude_derivatives.py",
    "test_sparks_normal_derivative_basis.py",
    "test_sparks_vibronic_bridge.py",
    "test_sparks_vpt2_bridge.py",
    "test_sparks_source_projection.py",
    "test_sparks_tail.py",
    "test_sparks_reflection.py",
    "test_composite_full_hessian.py",
    "test_targeted_polyad_forces.py",
    "test_selected_vci.py",
    "test_selected_vci_overlaps.py",
    "test_spectral_window_convergence.py",
    "test_variational_operator_bank.py",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(output: Path, revision: str):
    root = Path(__file__).resolve().parents[2]
    if output.exists():
        raise FileExistsError(output)
    revision = subprocess.check_output(["git", "rev-parse", revision], cwd=root, text=True).strip()
    queue, projects = list(ROOTS), {}
    while queue:
        name = queue.pop()
        if name in projects:
            continue
        raw = subprocess.check_output(
            ["git", "show", f"{revision}:packages/{name}/pyproject.toml"], cwd=root
        )
        metadata = tomllib.loads(raw.decode())["project"]
        projects[name] = metadata["version"]
        queue.extend(
            _requirement_name(item)
            for item in metadata.get("dependencies", ())
            if _requirement_name(item).startswith("matrix-")
        )
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    paths = [f"packages/{name}" for name in sorted(projects)]
    paths += [
        "LICENSE",
        "VERSION",
        "pyproject.toml",
        "tools/standalone/sparks",
        "tools/build/build_smith_release.py",
        "tools/build/build_sparks_standalone.py",
        "scripts/qualify_sparks_checkpoint_response.py",
        "scripts/prepare_sparks_normal_derivatives.py",
        "scripts/compute_sparks_normal_derivatives.py",
        "scripts/export_saved_ept_dyson.py",
        "scripts/export_gaussian_ept_batch.py",
        "scripts/audit_sparks_dyson_stencil.py",
        "scripts/prepare_sparks_composite_reference.py",
        "scripts/qualify_sparks_composite_polyads.py",
        "scripts/evaluate_sparks_variational_window.py",
        "scripts/compare_sparks_nuclear_models.py",
        "scripts/compare_sparks_saved_stencils.py",
        "scripts/compare_sparks_orbital_sources.py",
        "scripts/plot_sparks_energy_observables.py",
        "scripts/extract_vector_plot_points.py",
        "scripts/check_sparks_continuum_overlap.py",
        "scripts/plot_sparks_nuclear_comparison.py",
        "scripts/qualify_sparks_nuclear_progression.py",
        "scripts/qualify_sparks_dyson_continuum.py",
        "scripts/run_frozen_gaussian_batch.py",
        *[f"tests/{name}" for name in TESTS],
    ]
    # Include all current SPARKS checks, including optimized continuum assembly.
    tracked_tests = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "tests"], cwd=root, text=True
    ).splitlines()
    paths.extend(path for path in tracked_tests
                 if Path(path).name.startswith("test_sparks_") and path not in paths)
    with tempfile.TemporaryFile() as stream:
        subprocess.run(["git", "archive", revision, *paths], cwd=root, stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            archive.extractall(source, filter="data")
    # TIRESIA adapter code may be supplied; third-party Fortran overlays need an
    # independently licensed reference installation and do not belong to production.
    reference = source / "packages/matrix-sparks/src/matrix_sparks/reference/tiresia/fortran"
    if reference.exists():
        shutil.rmtree(reference)
    shutil.copyfile(source / "tools/standalone/sparks/README.md", output / "README.md")
    shutil.copyfile(source / "tools/standalone/sparks/install.py", output / "install.py")
    shutil.copyfile(source / "LICENSE", output / "LICENSE")
    shutil.copytree(source / "tools/standalone/sparks/examples", output / "examples")
    record = dict(
        schema="sparks.standalone.source.v1",
        scientific_revision=revision,
        components=projects,
        python_requires=">=3.11",
        files={
            str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()
        },
    )
    (output / "source-manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(output / "source-manifest.json")


def build(output: Path):
    # The guard reads physical hardware identity, not a caller-supplied hostname.
    import platform
    import socket

    if socket.getfqdn() != "oracle.sns.it":
        if platform.system() != "Darwin":
            raise RuntimeError("Build on Oracle or Mac Studio")
        name = subprocess.check_output(["scutil", "--get", "ComputerName"], text=True).lower()
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
        if "mac studio" not in name or model not in {
            "Mac13,1",
            "Mac13,2",
            "Mac14,13",
            "Mac14,14",
            "Mac15,14",
            "Mac16,9",
        }:
            raise RuntimeError("Build on verified Mac Studio, never a laptop")
    record = json.loads((output / "source-manifest.json").read_text())
    for name, digest in record["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or sha256(path) != digest:
            raise ValueError(f"Source hash mismatch: {name}")
    wheels = output / "wheels"
    wheels.mkdir(exist_ok=False)
    source = output / "source"
    with tempfile.TemporaryDirectory(prefix="sparks-build-") as directory:
        staged = Path(directory)
        projects = [
            _stage_project(source / "packages" / name, staged) for name in record["components"]
        ]
        projects.append(_stage_project(source / "tools/standalone/sparks", staged))
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--wheel-dir", str(wheels), *map(str, projects)],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--dest",
            str(wheels),
            "pytest>=8,<9",
        ],
        check=True,
    )
    built = dict(
        schema="sparks.standalone.wheelhouse.v1",
        scientific_revision=record["scientific_revision"],
        python=sys.version,
        platform=platform.platform(),
        files={p.name: sha256(p) for p in sorted(wheels.glob("*.whl"))},
    )
    (output / "wheelhouse-manifest.json").write_text(
        json.dumps(built, indent=2, sort_keys=True) + "\n"
    )
    print(output / "wheelhouse-manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser("export")
    exp.add_argument("output", type=Path)
    exp.add_argument("--revision", default="HEAD")
    bld = commands.add_parser("build")
    bld.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        export(args.output.expanduser().resolve(), args.revision)
    else:
        build(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
