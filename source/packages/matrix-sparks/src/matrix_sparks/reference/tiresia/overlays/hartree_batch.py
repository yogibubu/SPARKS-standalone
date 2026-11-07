"""Factor the existing two-centre Hartree sum before the azimuthal loop."""

import difflib
import hashlib
from pathlib import Path

from ..resources import fortran_source

MODULE = fortran_source("tiresia_hartree_batch_module.f90")


def prepare_hartree_batch(source: Path, base: Path, destination: Path) -> dict:
    destination.mkdir(exist_ok=True)
    changed = {}
    caller = base / "mpiint2cenxx_1.f90"
    old = caller.read_text()
    new = old.replace(
        "      implicit none",
        "      use matrix_hartree_batch, only: prepare_hartree\n      implicit none",
        1,
    )
    anchor = "            vphi((-lmax2 ) :lmax2,(-lmax1 ) :lmax1) = 0.d0"
    if old.count(anchor) != 1:
        raise ValueError("Unexpected potential quadrature layout")
    new = new.replace(
        anchor,
        """            if (idens/=0 .and. imol/=0) call prepare_hartree(lmax1,khigh1,f4,r1,q2,acc1, &
                lmax0,khigh0,f0,acc0,nsplmx,nspl1x,natot,iord,lmax2,npph,ibl0,iblr(ipr),iq,lmaxx)
"""
        + anchor,
    )
    changed[caller] = new
    potential = source / "lcao/vkspotg.f90"
    old = potential.read_text()
    new = old.replace(
        "      implicit none",
        "      use matrix_hartree_batch, only: hartree_phi\n      implicit none",
        1,
    )
    start = new.index("      VVALKS = 0.0")
    end = new.index("      xact =", start)
    new = new[:start] + "      vvalks = hartree_phi(iph)\n\n" + new[end:]
    changed[potential] = new
    module = destination / "matrix_hartree_batch.f90"
    if not module.exists() or module.read_text() != MODULE:
        module.write_text(MODULE)
    metadata, patch = [], []
    for filename, new in changed.items():
        old = filename.read_text()
        target = destination / filename.name
        if not target.exists() or target.read_text() != new:
            target.write_text(new)
        metadata.append(
            dict(
                source=str(filename),
                input_sha256=hashlib.sha256(old.encode()).hexdigest(),
                candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),
            )
        )
        patch.extend(
            difflib.unified_diff(
                old.splitlines(True),
                new.splitlines(True),
                fromfile=filename.name,
                tofile="hartree_batch/" + filename.name,
            )
        )
    (destination / "hartree-batch.patch").write_text("".join(patch))
    return dict(
        changes=metadata,
        module_sha256=hashlib.sha256(MODULE.encode()).hexdigest(),
        scope="Factor radial contractions from the azimuthal Hartree loop; no integral omitted",
    )
