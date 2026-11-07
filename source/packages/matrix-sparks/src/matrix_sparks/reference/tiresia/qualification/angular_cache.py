"""Check lazy Wigner cache growth, invalid selection rules and fallback."""

import argparse
import json
import socket
import subprocess
from pathlib import Path

from ..resources import fortran_source

DRIVER = fortran_source("qualify_tiresia_angular_cache_driver.f90")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower() != "oracle":
        raise RuntimeError("Run on Oracle")
    args.output.mkdir(exist_ok=True, parents=True)
    driver = args.output / "driver.f90"
    driver.write_text(DRIVER)
    obj = args.source / "sigma2-angular-cache_obj"
    binary = args.output / "qualify.exe"
    subprocess.run(
        [
            "gfortran",
            "-O2",
            "-fcheck=all",
            "-I",
            str(obj),
            "-o",
            str(binary),
            str(args.source / "gnu_compat/angular_cache/a3j.f90"),
            str(driver),
            *[str(obj / (name + ".o")) for name in ("bfact", "logfact", "lfactor_C")],
        ],
        cwd=args.output,
        check=True,
    )
    result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
    report = dict(
        host=socket.gethostname(),
        bounds_checking=True,
        output=result.stdout,
        maximum_cached_l=256,
        fallback_l_checked=257,
        exact_equality=True,
    )
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
