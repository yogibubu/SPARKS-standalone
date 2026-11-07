"""Compare GTO values and Cartesian gradients with the upstream routines."""

import argparse
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

from ..resources import fortran_source

DRIVER = fortran_source("qualify_tiresia_gto_fast_driver.f90")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower() != "oracle":
        raise RuntimeError("Run numerical qualification on Oracle")
    from matrix_sparks.reference.tiresia.overlays.gto_cache import prepare_gto_cache
    from matrix_sparks.reference.tiresia.overlays.gto_fast import prepare_gto_fast

    args.output.mkdir(exist_ok=True, parents=True)
    prepare_gto_cache(args.source, args.output / "cache")
    metadata = prepare_gto_fast(args.output / "cache", args.output / "fast")
    sources = [args.output / "fast/mod_orbread_n.f90"]
    for name in ("orbden", "orbgrad"):
        filename = args.source / "monew" / (name + "_1.f90")
        reference = args.output / (name + "_reference.f90")
        reference.write_text(
            filename.read_text().replace("subroutine " + name, "subroutine " + name + "_reference")
        )
        sources.extend([reference, args.output / "fast" / filename.name])
    sources.extend(args.source / "lib" / name for name in ("plm.f90", "csphi1.f90"))
    semifat = args.source / "monew/function.f90"
    driver = args.output / "driver.f90"
    driver.write_text(DRIVER)
    binary = args.output / "qualify.exe"
    mpifort = os.environ.get("SPARKS_MPIFORT") or shutil.which("mpifort")
    if not mpifort:
        raise RuntimeError("set SPARKS_MPIFORT or install mpifort on the qualification host")
    subprocess.run(
        [
            mpifort,
            "-O2",
            "-fcheck=all",
            "-ffree-line-length-none",
            "-I",
            str(args.source / "lcaompi_obj"),
            "-o",
            str(binary),
            *map(str, sources),
            str(semifat),
            str(driver),
        ],
        cwd=args.output,
        check=True,
    )
    result = subprocess.run(
        [str(binary)],
        cwd=args.output,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    report = dict(
        host=socket.gethostname(),
        points=600,
        shells="Cartesian s through g",
        coefficients="dense, sparse, all zero",
        bounds_checking=True,
        tolerance_scaled=5e-13,
        output=result.stdout,
        metadata=metadata,
    )
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
