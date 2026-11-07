"""Reuse exact upstream dipole Wigner coefficients across continuum energies."""

import difflib
import hashlib
import re
from pathlib import Path

from ..resources import fortran_source

WRAPPER = fortran_source("tiresia_angular_cache_wrapper.f90")


def prepare_angular_cache(source: Path, destination: Path) -> dict:
    destination.mkdir(exist_ok=True)
    filename = source / "lib/a3j.f90"
    old = filename.read_text()
    new = WRAPPER + re.sub(r"\ba3j\b", "matrix_a3j_reference", old, flags=re.I)
    target = destination / "a3j.f90"
    if not target.exists() or target.read_text() != new:
        target.write_text(new)
    patch = "".join(
        difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile="lib/a3j.f90",
            tofile="angular_cache/a3j.f90",
        )
    )
    (destination / "angular-cache.patch").write_text(patch)
    return dict(
        source="lib/a3j.f90",
        original_sha256=hashlib.sha256(old.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),
        scope="Exact upstream Wigner values for li=1, l<=256; reference fallback otherwise",
    )
