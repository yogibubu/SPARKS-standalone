"""Compare postprocessors on retained continuum checkpoints; never rerun them."""

import argparse
import json
import os
import socket
import statistics
from pathlib import Path

from ..runtime import digest, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower() != "oracle":
        raise RuntimeError("Run on Oracle")
    import numpy as np

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = env["OMP_NUM_THREADS"] = "1"
    args.output.mkdir(exist_ok=True, parents=True)
    rows = []
    for case in ("H2plus", "CO-gto"):
        checkpoint = args.checkpoints / f"{case}-baseline-mpi4-r0/fort.20"
        example = args.source / "tiresia_tests" / ("H2+" if case == "H2plus" else "CO")
        reference = example / ("reference/t70.h2+" if case == "H2plus" else "t70.co_lb94_2")
        target = np.loadtxt(reference)
        for repeat in range(3):
            for build in (
                ("baseline", "angular-cache") if repeat % 2 == 0 else ("angular-cache", "baseline")
            ):
                binary = args.source / (
                    "sigma2.exe" if build == "baseline" else "sigma2-angular-cache.exe"
                )
                directory = args.output / f"{case}-{build}-r{repeat}"
                directory.mkdir(exist_ok=True)
                result_file = directory / "result.json"
                identity = dict(
                    binary_sha256=digest(binary),
                    checkpoint_sha256=digest(checkpoint),
                    reference_sha256=digest(reference),
                )
                if result_file.exists():
                    item = json.loads(result_file.read_text())
                    if any(item[k] != v for k, v in identity.items()) or not item["passed"]:
                        raise ValueError("Checkpoint identity or qualification mismatch")
                    rows.append(item)
                    continue
                (directory / "fort.20").symlink_to(checkpoint.resolve())
                inp = directory / "fort.5"
                inp.write_text(" &PARAM iprint=0 iener=0 / &END\n")
                seconds = execute(
                    [str(binary.resolve())], directory, inp, directory / "sigma.log", env, 90
                )
                actual = np.loadtxt(directory / "fort.70")
                passed = actual.shape == target.shape and bool(
                    np.allclose(actual, target, rtol=2e-7, atol=2e-7)
                )
                item = dict(
                    case=case,
                    build=build,
                    repetition=repeat,
                    seconds=seconds,
                    passed=passed,
                    rows=len(actual),
                    maximum_column_errors=np.max(abs(actual - target), axis=0).tolist(),
                    host=socket.gethostname(),
                    **identity,
                )
                result_file.write_text(json.dumps(item, indent=2) + "\n")
                if not passed:
                    raise ArithmeticError("Spectrum mismatch")
                rows.append(item)
                print(json.dumps(item), flush=True)
    groups = [
        dict(
            case=case,
            build=build,
            median_seconds=statistics.median(
                r["seconds"] for r in rows if r["case"] == case and r["build"] == build
            ),
        )
        for case in ("H2plus", "CO-gto")
        for build in ("baseline", "angular-cache")
    ]
    (args.output / "summary.json").write_text(
        json.dumps(dict(groups=groups, runs=rows), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
