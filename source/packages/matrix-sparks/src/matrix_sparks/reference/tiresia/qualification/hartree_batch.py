"""Compare factored Hartree values with the literal upstream loop, including 3D cases."""

import argparse
import json
import socket
import subprocess
from pathlib import Path

from ..resources import fortran_source

DRIVER = fortran_source("qualify_tiresia_hartree_batch_driver.f90")

REFERENCE = fortran_source("qualify_tiresia_hartree_batch_reference.f90")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower() != "oracle":
        raise RuntimeError("Run on Oracle")
    args.output.mkdir(exist_ok=True, parents=True)
    upstream = (args.source / "lcao/vkspotg.f90").read_text()
    body = upstream[upstream.index("      VVALKS = 0.0") : upstream.index("      xact =")]
    driver = args.output / "driver.f90"
    driver.write_text(REFERENCE + body + "\nend subroutine\n" + DRIVER)
    binary = args.output / "qualify.exe"
    subprocess.run(
        [
            "gfortran",
            "-O2",
            "-fcheck=all",
            "-ffree-line-length-none",
            "-o",
            str(binary),
            str(args.source / "gnu_compat/hartree_batch/matrix_hartree_batch.f90"),
            str(driver),
        ],
        cwd=args.output,
        check=True,
    )
    result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
    report = dict(
        host=socket.gethostname(),
        bounds_checking=True,
        seed=26102026,
        tolerance_scaled=5e-13,
        output=result.stdout,
        reference="Literal upstream vkspotg Hartree loop",
    )
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
