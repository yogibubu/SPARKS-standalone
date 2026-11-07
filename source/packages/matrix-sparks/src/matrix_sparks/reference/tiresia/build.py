"""Build the official Tiresia source with GNU Fortran, preserving upstream files.

Run on Oracle or MAC-STUDIO. Supply a ScaLAPACK built with the same MPI and
Fortran toolchain; no Intel compiler license or MKL dependency is required.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import socket
import subprocess
from pathlib import Path

from .resources import fortran_source

UNIT_NAME = fortran_source("build_tiresia_gnu_unit_name.f90")


def write_changed(path: Path, text: str) -> None:
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def portable_source(text: str) -> str:
    """Translate vendor syntax and make implicit NEW filenames explicit."""
    result = text.replace("month*3(12)", "month(12)*3")
    result = re.sub(
        r"character\s*,\s*allocatable\s*::\s*(\w+)\*(\d+)\(:\)",
        lambda m: f"character(len={m[2]}), allocatable :: {m[1]}(:)",
        result,
        flags=re.I,
    )
    result = result.replace(
        "call mpiddipfull (n0, d0,  xx, nspace, )",
        "call mpiddipfull (n0, d0,  xx, nspace)",
    )
    helper = False
    original = result

    def explicit_filename(match: re.Match) -> str:
        nonlocal helper
        line_start = original.rfind("\n", 0, match.start()) + 1
        if original[line_start : match.start()].lstrip().startswith("!"):
            return match[0]
        if not re.search(r"status\s*=\s*'new'", match[0], re.I):
            return match[0]
        if re.search(r"\bfile\s*=", match[0], re.I):
            return match[0]
        unit_match = re.match(r"open\s*\(\s*(?:unit\s*=\s*)?([^,]+),", match[0], re.I)
        if unit_match is None:
            raise ValueError("Unrecognized OPEN statement; inspect upstream source")
        unit = unit_match[1].strip()
        if unit.isdigit():
            filename = f"'fort.{unit}'"
        else:
            helper = True
            filename = f"gnu_tiresia_unit_name({unit})"
        return match[0][:-1] + ", FILE=" + filename + ")"

    result = re.sub(r"\bopen\s*\([^)]*\)", explicit_filename, result, flags=re.I)
    if helper:
        result, count = re.subn(
            r"(?im)^(\s*implicit[^\n]*\n)",
            r"\1      character(len=64), external :: gnu_tiresia_unit_name\n",
            result,
            count=1,
        )
        if count != 1:
            raise ValueError("Explicit filename helper needs a declaration location")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mpi", type=Path, required=True)
    parser.add_argument("--scalapack", type=Path, required=True)
    parser.add_argument("--blas", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4, choices=range(1, 9))
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument(
        "--gprof", action="store_true", help="Separate instrumented binary for source profiling"
    )
    profiles.add_argument(
        "--native", action="store_true", help="Separate O3/native binaries for measured comparison"
    )
    parser.add_argument(
        "--gto-cache",
        action="store_true",
        help="Cache point-independent GTO work in a separate candidate build",
    )
    parser.add_argument(
        "--sparse-blocks",
        action="store_true",
        help="Preserve exact angular sparsity in two-centre assembly",
    )
    parser.add_argument(
        "--gto-fast",
        action="store_true",
        help="Reuse Cartesian powers and exact MO sparsity over GTO cache",
    )
    parser.add_argument(
        "--angular-cache",
        action="store_true",
        help="Reuse exact dipole Wigner coefficients across energies",
    )
    parser.add_argument(
        "--gto-batch", action="store_true", help="Batch two-centre GTO grids through SPARKS/libcgto"
    )
    parser.add_argument(
        "--hartree-batch",
        action="store_true",
        help="Factor radial Hartree contractions before azimuthal integration",
    )
    parser.add_argument(
        "--sparks-source", type=Path, help="SPARKS native/gto_grid.cpp (required for batch build)"
    )
    parser.add_argument(
        "--cgto-library", type=Path, help="Existing PySCF libcgto.so (batch build only)"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("main", "sigma", "sdipole"),
        default=["main", "sigma", "sdipole"],
    )
    args = parser.parse_args()
    host = socket.gethostname()
    if host.split(".")[0].lower() not in ("oracle", "mac-studio"):
        raise RuntimeError("Build only on the authorized compute hosts")
    source = args.source.resolve()
    compatibility = source / "gnu_compat"
    compatibility.mkdir(exist_ok=True)
    changes = []
    patch = []
    for filename in sorted(source.rglob("*.f90")):
        if compatibility in filename.parents:
            continue
        old = filename.read_text()
        new = portable_source(old)
        if old == new:
            continue
        target = compatibility / filename.name
        if any(c["overlay"] == target.name for c in changes):
            raise ValueError("Duplicate source basename in compatibility overlay")
        write_changed(target, new)
        relative = str(filename.relative_to(source))
        changes.append(
            dict(
                source=relative,
                overlay=target.name,
                original_sha256=hashlib.sha256(old.encode()).hexdigest(),
                overlay_sha256=hashlib.sha256(new.encode()).hexdigest(),
            )
        )
        patch.extend(
            difflib.unified_diff(
                old.splitlines(True),
                new.splitlines(True),
                fromfile=relative,
                tofile="gnu_compat/" + target.name,
            )
        )
    write_changed(compatibility / "gnu_tiresia_unit_name.f90", UNIT_NAME)
    (compatibility / "portability.patch").write_text("".join(patch))
    mpi = args.mpi.resolve()
    compiler = str(mpi / "bin/mpifort")
    env = os.environ.copy()
    env["PATH"] = str(mpi / "bin") + os.pathsep + env["PATH"]
    env["LD_LIBRARY_PATH"] = str(mpi / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["OPENBLAS_NUM_THREADS"] = "1"
    libraries = f"{args.scalapack.resolve()} {args.blas.resolve()} -lpthread -lm"
    suffix = "-gprof" if args.gprof else ("-native" if args.native else "")
    cache_metadata = None
    cache_vpath = ""
    if args.gto_cache or args.gto_fast or args.gto_batch:
        from matrix_sparks.reference.tiresia.overlays.gto_cache import prepare_gto_cache

        cache_directory = compatibility / "gto_cache"
        cache_metadata = prepare_gto_cache(source, cache_directory)
        cache_vpath = "gnu_compat/gto_cache:"
        suffix = "-gto-cache" + suffix
    fast_metadata = None
    if args.gto_fast or args.gto_batch:
        from matrix_sparks.reference.tiresia.overlays.gto_fast import prepare_gto_fast

        fast_metadata = prepare_gto_fast(cache_directory, compatibility / "gto_fast")
        cache_vpath = "gnu_compat/gto_fast:" + cache_vpath
        suffix = suffix.replace("-gto-cache", "-gto-fast")
    optimization = (
        "-O2 -pg -g"
        if args.gprof
        else ("-O3 -march=native -fno-strict-aliasing" if args.native else "-O2")
    )
    sparse_metadata = None
    if args.sparse_blocks:
        from matrix_sparks.reference.tiresia.overlays.sparse_blocks import prepare_sparse_blocks

        sparse_directory = compatibility / "sparse_blocks"
        sparse_metadata = prepare_sparse_blocks(source, sparse_directory)
        cache_vpath = "gnu_compat/sparse_blocks:" + cache_vpath
        suffix = "-sparse-blocks" + suffix
    manifest = dict(
        host=host,
        changes=changes,
        commands=[],
        targets={},
        optimization=optimization,
        gto_cache=cache_metadata,
        sparse_blocks=sparse_metadata,
        gto_fast=fast_metadata,
    )
    if args.angular_cache:
        from matrix_sparks.reference.tiresia.overlays.angular_cache import prepare_angular_cache

        manifest["angular_cache"] = prepare_angular_cache(source, compatibility / "angular_cache")
        cache_vpath = "gnu_compat/angular_cache:" + cache_vpath
        suffix = "-angular-cache" + suffix
    if args.gto_batch:
        from matrix_sparks.reference.tiresia.overlays.gto_batch import prepare_gto_batch

        if args.sparks_source is None or not args.sparks_source.is_file():
            parser.error(
                "--gto-batch requires --sparks-source pointing to the general SPARKS C++ kernel"
            )
        cgto = args.cgto_library
        if cgto is None:
            import pyscf

            cgto = Path(pyscf.__file__).parent / "lib/libcgto.so"
        cgto = cgto.resolve(strict=True)
        batch_directory = compatibility / "gto_batch"
        manifest["gto_batch"] = prepare_gto_batch(
            source, cache_directory, compatibility / "gto_fast", batch_directory
        )
        source_hash = hashlib.sha256(args.sparks_source.read_bytes()).hexdigest()
        cgto_hash = hashlib.sha256(cgto.read_bytes()).hexdigest()
        blas_hash = hashlib.sha256(args.blas.resolve().read_bytes()).hexdigest()
        compiler_version = subprocess.run(
            ["g++", "--version"], capture_output=True, text=True, check=True
        ).stdout
        native_identity = hashlib.sha256(
            (source_hash + cgto_hash + blas_hash + str(cgto) + compiler_version).encode()
        ).hexdigest()[:20]
        # Immutable dependencies keep older executables and active jobs valid.
        native = batch_directory / ("libmatrix_sparks_grid-" + native_identity + ".so")
        native_command = [
            "g++",
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-shared",
            str(args.sparks_source.resolve()),
            str(cgto),
            str(args.blas.resolve()),
            "-Wl,-rpath," + str(cgto.parent),
            "-o",
            str(native),
        ]
        if not native.exists():
            subprocess.run(native_command, check=True)
        manifest["native_grid"] = dict(
            command=native_command,
            library_path=str(native),
            source_sha256=source_hash,
            blas_sha256=blas_hash,
            library_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            cgto_sha256=cgto_hash,
            compiler_version=compiler_version,
        )
        (batch_directory / "native-library.json").write_text(
            json.dumps(manifest["native_grid"], indent=2) + "\n"
        )
        libraries += " " + str(native) + " -Wl,-rpath," + str(batch_directory)
        libraries += " -Wl,-rpath-link," + str(cgto.parent)
        libraries += " -Wl,-rpath-link," + str(cgto.parent / "deps/lib")
        cache_vpath = "gnu_compat/gto_batch:" + cache_vpath
        suffix = suffix.replace("-gto-fast", "-gto-batch")
    if args.hartree_batch:
        from matrix_sparks.reference.tiresia.overlays.hartree_batch import prepare_hartree_batch

        if not args.gto_batch:
            parser.error("--hartree-batch currently requires --gto-batch")
        manifest["hartree_batch"] = prepare_hartree_batch(
            source, batch_directory, compatibility / "hartree_batch"
        )
        cache_vpath = "gnu_compat/hartree_batch:" + cache_vpath
        suffix = "-hartree-batch" + suffix
    previous_manifest = compatibility / ("manifest" + suffix + ".json")
    previous = json.loads(previous_manifest.read_text()) if previous_manifest.exists() else {}
    flags_changed = bool(
        previous and (previous.get("optimization") != optimization or not previous.get("targets"))
    )
    for target in args.targets:
        if target in ("main", "sigma"):
            upstream = "makelcaompix_english" if target == "main" else "makemake"
            objdir = ("lcaompi" if target == "main" else "sigma2") + suffix + "_obj/"
            makefile = compatibility / (
                ("Makefile" if target == "main" else "Makefile.sigma") + suffix
            )
            # Upstream compiles with $?; module prerequisites must be order-only
            # so object filenames are not passed as missing source-root inputs.
            dependency = (
                "bfact.o logfact.o: | lfactor_C.o\na3j.o: | bfact_I.o\n"
                "lfactor_C.o: | vast_kind_param_M.o\n"
                if target == "sigma"
                else ""
            )
            if args.gto_batch and target == "main":
                dependency += "mpiint2clap_1.o mpiint2cenxx_1.o: | mod_orbread_n.o\n"
                dependency += "$(PROG): " + str(native) + "\n"
            if args.hartree_batch and target == "main":
                dependency += "$(PROG): matrix_hartree_batch.o\n"
                dependency += "mpiint2cenxx_1.o vkspotg.o: | matrix_hartree_batch.o\n"
            write_changed(
                makefile,
                f"include {upstream}\nVPATH := {cache_vpath}gnu_compat:$(VPATH)\n"
                "$(PROG): gnu_tiresia_unit_name.o\n" + dependency,
            )
            command = [
                "make",
                "-f",
                str(makefile.relative_to(source)),
                f"-j{args.jobs if target == 'main' else 1}",
                f"FC={compiler}",
                f"F90={compiler}",
                f"OBJSDIR={objdir}",
                f"PROG={'lcaompi' if target == 'main' else 'sigma2'}{suffix}.exe",
                f"FFLAGS={optimization} -ffree-line-length-none -J {objdir} -I {objdir}",
                "LDFLAGS=" + optimization,
                "LIBS=" + libraries,
            ]
            if flags_changed:
                (source / objdir).mkdir(exist_ok=True)
                command.extend(["-B", "--old-file=" + objdir])
            if args.gto_batch and previous.get("native_grid", {}).get("library_path") != str(
                native
            ):
                command.extend(["-W", str(native)])
            binary = source / (("lcaompi" if target == "main" else "sigma2") + suffix + ".exe")
        else:
            binary = source / ("sdipole" + suffix + ".exe")
            command = [
                compiler,
                *optimization.split(),
                "-ffree-line-length-none",
                "-o",
                str(binary),
                str(compatibility / "sdipole.f90"),
                str(compatibility / "gnu_tiresia_unit_name.f90"),
                str(args.blas.resolve()),
            ]
        manifest["commands"].append(command)
        log = compatibility / f"build-{target}{suffix}.log"
        with log.open("wb") as stream:
            completed = subprocess.run(
                command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        text = log.read_text(errors="replace")
        # Upstream sigma makefile uses .IGNORE, so its exit status alone is insufficient.
        if (
            completed.returncode
            or re.search(r"Error:|undefined reference|fatal error:", text)
            or not binary.exists()
        ):
            (compatibility / ("manifest" + suffix + ".json")).write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            raise RuntimeError(f"{target} build failed: {log}")
        manifest["targets"][target] = dict(
            binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest()
        )
        (compatibility / ("manifest" + suffix + ".json")).write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(target, manifest["targets"][target], flush=True)


if __name__ == "__main__":
    main()
