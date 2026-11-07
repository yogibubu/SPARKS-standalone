"""Screen exported response matrices and contract phase-preserving vertices."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .continuum.response import screen_response
from .execution import require_compute_host


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="NPZ with chi, kernel, external_potential and operator_basis_id"
    )
    parser.add_argument(
        "output", type=Path, help="New NPZ output; existing files are never replaced"
    )
    args = parser.parse_args(argv)
    host = require_compute_host()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as data:
        identity = data["operator_basis_id"]
        if identity.ndim or identity.dtype.kind != "U" or not str(identity):
            raise ValueError("Require a nonempty scalar Unicode operator_basis_id")
        result = screen_response(data["chi"], data["kernel"], data["external_potential"])
        output = dict(
            effective_potential=result.effective_potential,
            induced_density=result.induced_density,
            backward_error=np.asarray(result.backward_error),
            operator_basis_id=identity.copy(),
            input_sha256=np.asarray(hashlib.sha256(args.input.read_bytes()).hexdigest()),
            host=np.asarray(str(host)),
        )
        if "potential_vertices" in data:
            output["amplitudes"] = result.amplitudes(data["potential_vertices"])
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **output)


if __name__ == "__main__":
    main()
