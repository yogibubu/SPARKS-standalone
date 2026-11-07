"""Preserve exact angular sparsity in Tiresia's two-centre assembly.

No screening tolerance is introduced. Only updates with an exactly zero
multiplier and angular entries forbidden by the existing formulas are omitted.
Upstream sources remain intact; candidates use a separate VPATH overlay.
"""

import difflib
import hashlib
import re
from pathlib import Path


def prepare_sparse_blocks(source: Path, destination: Path) -> dict:
    destination.mkdir(exist_ok=True)
    changes = []
    patch = []
    for name in ("blockint.f90", "blockintlap.f90"):
        filename = source / "lcao" / name
        old = filename.read_text()
        new = old
        if name == "blockint.f90":
            # Array sections may span continuation lines. Keep all index
            # increments outside the guard, preserving the original layout.
            # Match assignments explicitly to avoid consuming intervening loops.
            pattern = r"(?im)^( +)([abc])\([^\n]+ = [^\n]+(?:&\s*\n[^\n]+)?\*(ff3|fx1)\s*\n"
            counts = {"a": 0, "b": 0, "c": 0}

            def guard(match):
                counts[match[2]] += 1
                return (
                    match[1]
                    + "if ("
                    + match[3]
                    + " /= 0.d0) then\n"
                    + match[0]
                    + match[1]
                    + "endif\n"
                )

            new = re.sub(pattern, guard, new)
            if counts != {"a": 1, "b": 1, "c": 3}:
                raise ValueError(f"Unexpected blockint assignment layout: {counts}")
        else:
            # The original visits all m0 and adds zero unless m0 == m1.
            # Address only that allowed angular entry, keeping radial order.
            start = new.index("               nlm0 = ibl0 - nspl0")
            end = new.index("!\n            end do", start)
            new = (
                new[:start]
                + """               if (khigh0 > 0 .and. ff2 /= 0.d0) then
                  do l0 = abs(m1), lmax0
                     i0 = (l0*(l0+1) + m1)*khigh0
                     nlm0 = (l0*(l0+1) + m1)*nspl0 + ibl0
                     b(nlm0+1:khigh0+nlm0,ind1) = b(nlm0+1:khigh0+nlm0,ind1) &
                         + f0(i0+1:khigh0+i0)*ff2
                  enddo
               endif
"""
                + new[end:]
            )
        target = destination / name
        if not target.exists() or target.read_text() != new:
            target.write_text(new)
        changes.append(
            dict(
                source=str(filename.relative_to(source)),
                original_sha256=hashlib.sha256(old.encode()).hexdigest(),
                candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),
            )
        )
        patch.extend(
            difflib.unified_diff(
                old.splitlines(True),
                new.splitlines(True),
                fromfile="lcao/" + name,
                tofile="sparse_blocks/" + name,
            )
        )
    (destination / "sparse-blocks.patch").write_text("".join(patch))
    return dict(
        changes=changes,
        scope="Exact-zero updates and analytic angular selection only; no threshold",
    )
