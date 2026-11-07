"""Compare original and sparse Tiresia assembly kernels on a compute host."""

import argparse
import ctypes
import json
import socket
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower() != "oracle":
        raise RuntimeError("Run numerical qualification on Oracle")
    import numpy as np

    from matrix_sparks.reference.tiresia.overlays.sparse_blocks import prepare_sparse_blocks

    args.output.mkdir(exist_ok=True, parents=True)
    metadata = prepare_sparse_blocks(args.source, args.output / "candidate")
    sources = [args.source / "modint/vast_kind_param_M.f90"]
    for name in ("blockint", "blockintlap"):
        sources.append(args.source / "lcao" / (name + ".f90"))
        candidate = args.output / (name + "_candidate.f90")
        candidate.write_text(
            (args.output / "candidate" / (name + ".f90"))
            .read_text()
            .replace("subroutine " + name + "(", "subroutine " + name + "_candidate(")
            .replace("end subroutine " + name, "end subroutine " + name + "_candidate")
        )
        sources.append(candidate)
    library = args.output / "kernels.so"
    subprocess.run(
        [
            "gfortran",
            "-O2",
            "-fcheck=bounds",
            "-shared",
            "-fPIC",
            "-o",
            str(library),
            *map(str, sources),
        ],
        cwd=args.output,
        check=True,
    )
    lib = ctypes.CDLL(str(library))

    def invoke(name, values):
        storage = []
        for value in values:
            if isinstance(value, np.ndarray):
                storage.append(value)
            else:
                storage.append(
                    ctypes.c_int(value) if isinstance(value, int) else ctypes.c_double(value)
                )
        pointers = [
            v.ctypes.data_as(ctypes.c_void_p) if isinstance(v, np.ndarray) else ctypes.byref(v)
            for v in storage
        ]
        getattr(lib, name + "_")(*pointers)

    rng = np.random.default_rng(26012026)
    count = 0
    for lm0 in range(5):
        for lm1 in range(5):
            for kh0 in (0, 1, 3):
                for kh1 in (0, 1, 3):
                    for angular in ("dense", "diagonal", "zero"):
                        ns0, ns1, ib0, ib1 = 6, 7, 1, 2
                        nd0, nd1 = ns0 * (lm0 + 1) ** 2 + 4, ns1 * (lm1 + 1) ** 2 + 4
                        lm2 = max(lm0, lm1)

                        def random(shape):
                            return np.asfortranarray(rng.normal(size=shape))

                        f0 = random(max(1, kh0 * (lm0 + 1) ** 2))
                        ff = [random(max(1, kh1 * (lm1 + 1) ** 2)) for _ in range(4)]
                        vphi = random((2 * lm2 + 1, 2 * lm1 + 1))
                        if angular != "dense":
                            vphi[:] = 0
                        if angular == "diagonal":
                            for m in range(-lm1, lm1 + 1):
                                vphi[m + lm2, m + lm1] = rng.normal()
                        a, b, c = random((nd0, nd1)), random((nd0, nd1)), random((nd1, nd1))
                        originals = [a, b, c, vphi]
                        candidates = [x.copy(order="F") for x in originals]
                        for name, arrays in (
                            ("blockint", originals),
                            ("blockint_candidate", candidates),
                        ):
                            aa, bb, cc, vv = arrays
                            invoke(
                                name,
                                [
                                    ns1,
                                    lm1,
                                    kh1,
                                    ib1,
                                    ns0,
                                    lm0,
                                    4,
                                    ib0,
                                    kh0,
                                    f0,
                                    *ff,
                                    lm2,
                                    vv,
                                    0.31,
                                    aa,
                                    bb,
                                    cc,
                                    nd0,
                                    nd1,
                                ],
                            )
                        if not all(np.array_equal(x, y) for x, y in zip(originals, candidates)):
                            raise ArithmeticError(
                                ("blockint mismatch", lm0, lm1, kh0, kh1, angular)
                            )
                        a, b = random(nd1), random((nd0, nd1))
                        aa, bb = a.copy(), b.copy(order="F")
                        angular_vector = random(2 * lm1 + 1)
                        invoke(
                            "blockintlap",
                            [
                                ns1,
                                lm1,
                                kh1,
                                ib1,
                                ns0,
                                lm0,
                                ib0,
                                kh0,
                                f0,
                                ff[0],
                                ff[1],
                                angular_vector,
                                a,
                                b,
                                nd0,
                                nd1,
                            ],
                        )
                        invoke(
                            "blockintlap_candidate",
                            [
                                ns1,
                                lm1,
                                kh1,
                                ib1,
                                ns0,
                                lm0,
                                ib0,
                                kh0,
                                f0,
                                ff[0],
                                ff[1],
                                angular_vector,
                                aa,
                                bb,
                                nd0,
                                nd1,
                            ],
                        )
                        if not np.array_equal(a, aa) or not np.array_equal(b, bb):
                            raise ArithmeticError(("blockintlap mismatch", lm0, lm1, kh0, kh1))
                        count += 1
    report = dict(
        host=socket.gethostname(),
        parameter_cases=count,
        kernels_per_case=2,
        agreement="exact array equality",
        bounds_checking=True,
        seed=26012026,
        metadata=metadata,
    )
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
