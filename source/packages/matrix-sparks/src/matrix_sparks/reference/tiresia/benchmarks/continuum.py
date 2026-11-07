"""Measure Tiresia builds on its supplied examples, retaining every calculation.

Numerical work is restricted to Oracle. Input physics, radial/angular grids and
energy lists remain those of the distributed examples. Only the GNU integer
namelist spelling zshell=N. -> zshell=N is normalized.
"""

import argparse
import hashlib
import json
import os
import re
import resource
import socket
import statistics
from pathlib import Path

from ..runtime import digest, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mpi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", choices=("H2plus", "CO-gto"), default=["H2plus", "CO-gto"]
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--builds",
        nargs="+",
        choices=(
            "baseline",
            "native",
            "gto-cache",
            "sparse-blocks",
            "sparse-blocks-gto-cache",
            "gto-fast",
            "sparse-blocks-gto-fast",
            "sparse-blocks-gto-batch",
            "hartree-batch-sparse-blocks-gto-batch",
        ),
        default=["baseline", "native"],
    )
    parser.add_argument(
        "--sigma-binary", type=Path, help="Qualified postprocessor; defaults to source/sigma2.exe"
    )
    parser.add_argument(
        "--candidate-sigma-binary", type=Path, help="Postprocessor for candidate builds only"
    )
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "oracle":
        raise RuntimeError("Oracle required; no laptop benchmarks")
    if not all(n in (1, 2, 4, 8) for n in args.ranks):
        parser.error("Supported MPI counts: 1, 2, 4, 8")
    if not 1 <= args.repetitions <= 5:
        parser.error("Bounded repeat count: 1..5")
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**3,) * 2)
    import numpy as np

    source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["LD_LIBRARY_PATH"] = (
        str(args.mpi / "lib") + ":" + environment.get("LD_LIBRARY_PATH", "")
    )
    results = []
    for case in args.cases:
        example = source / "tiresia_tests" / ("H2+" if case == "H2plus" else "CO")
        job = example / ("spln.0.job" if case == "H2plus" else "spln.2.job")
        text = (
            job.read_text()
            .split('cat <<"END OF DATA" > input\n', 1)[1]
            .split("\nEND OF DATA", 1)[0]
            + "\n"
        )
        text = re.sub(r"(zshell\s*=\s*\d+)\.(?!\d)", r"\1", text, flags=re.I)
        reference = example / ("reference/t70.h2+" if case == "H2plus" else "t70.co_lb94_2")
        target = np.loadtxt(reference)
        # Alternate build order between repeats to reduce host-load bias.
        schedule = [
            (build, ranks, repetition)
            for ranks in args.ranks
            for repetition in range(args.repetitions)
            for build in (args.builds if repetition % 2 == 0 else list(reversed(args.builds)))
        ]
        for build, ranks, repetition in schedule:
            binary = source / ("lcaompi.exe" if build == "baseline" else f"lcaompi-{build}.exe")
            binary_hash = digest(binary)
            sigma_binary = args.sigma_binary or (source / "sigma2.exe")
            if build != "baseline" and args.candidate_sigma_binary is not None:
                sigma_binary = args.candidate_sigma_binary
            directory = args.output / f"{case}-{build}-mpi{ranks}-r{repetition}"
            directory.mkdir(exist_ok=True)
            checkpoint = directory / "result.json"
            if checkpoint.exists():
                item = json.loads(checkpoint.read_text())
                if (
                    item["binary_sha256"] != binary_hash
                    or item["input_sha256"] != hashlib.sha256(text.encode()).hexdigest()
                ):
                    raise ValueError("Existing benchmark has a different source identity")
                if item["reference_sha256"] != digest(reference) or not item["passed"]:
                    raise ValueError("Existing benchmark failed or used a different reference")
                if item.get("sigma_binary_sha256", digest(source / "sigma2.exe")) != digest(
                    sigma_binary
                ):
                    raise ValueError(
                        "Postprocessor changed; main checkpoint retained. Reprocess fort.20 with benchmark_tiresia_sigma.py"
                    )
                actual = np.loadtxt(directory / "fort.70")
                if actual.shape != target.shape or not np.allclose(
                    actual, target, rtol=2e-7, atol=2e-7
                ):
                    raise ValueError("Existing spectrum no longer passes its reference check")
                results.append(item)
                continue
            input_path = directory / "input"
            input_path.write_text(text)
            if case == "CO-gto":
                link = directory / "mofile.dat"
                if not link.exists():
                    link.symlink_to(example / "co.hf.dat_1")
                elif link.resolve() != (example / "co.hf.dat_1").resolve():
                    raise ValueError("Existing GTO source identity differs")
            command = [str(args.mpi / "bin/mpirun"), "-np", str(ranks), str(binary)]
            main_seconds = execute(
                command, directory, input_path, directory / "main.log", environment, args.timeout
            )
            sigma_input = directory / "fort.5"
            sigma_input.write_text(" &PARAM iprint=0 iener=0 / &END\n")
            sigma_seconds = execute(
                [str(sigma_binary)],
                directory,
                sigma_input,
                directory / "sigma.log",
                environment,
                60,
            )
            actual = np.loadtxt(directory / "fort.70")
            if actual.shape != target.shape:
                raise ValueError(f"Spectrum shape mismatch: {directory}")
            errors = np.max(abs(actual - target), axis=0)
            passed = bool(np.allclose(actual, target, rtol=2e-7, atol=2e-7))
            item = dict(
                case=case,
                build=build,
                ranks=ranks,
                repetition=repetition,
                main_seconds=main_seconds,
                sigma_seconds=sigma_seconds,
                spectrum_rows=len(actual),
                maximum_absolute_column_errors=errors.tolist(),
                passed=passed,
                binary_sha256=binary_hash,
                input_sha256=hashlib.sha256(text.encode()).hexdigest(),
                reference_sha256=digest(reference),
                host=socket.gethostname(),
                sigma_binary_sha256=digest(sigma_binary),
                blas_threads_per_process=1,
            )
            checkpoint.write_text(json.dumps(item, indent=2) + "\n")
            results.append(item)
            print(json.dumps(item), flush=True)
            if not passed:
                raise ArithmeticError(
                    "Reference spectrum check failed; retain logs and reject candidate"
                )
    groups = []
    for case in args.cases:
        for build in args.builds:
            for ranks in args.ranks:
                rows = [
                    r
                    for r in results
                    if (r["case"], r["build"], r["ranks"]) == (case, build, ranks)
                ]
                if rows:
                    groups.append(
                        dict(
                            case=case,
                            build=build,
                            ranks=ranks,
                            repetitions=len(rows),
                            median_main_seconds=statistics.median(r["main_seconds"] for r in rows),
                            minimum_main_seconds=min(r["main_seconds"] for r in rows),
                            maximum_main_seconds=max(r["main_seconds"] for r in rows),
                            all_reference_checks_passed=all(r["passed"] for r in rows),
                        )
                    )
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                groups=groups,
                runs=results,
                scope="Supplied test systems at their fixed grids; no extrapolation to H2O cost or continuum accuracy",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
