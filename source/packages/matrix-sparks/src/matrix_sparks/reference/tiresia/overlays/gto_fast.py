"""Prepare power reuse and exact sparse MO contraction over the GTO cache."""

import re
from pathlib import Path

from .files import write_overlay


def prepare_gto_fast(cache: Path, destination: Path) -> dict:
    destination.mkdir(exist_ok=True)
    changed = {}
    module = cache / "mod_orbread_n.f90"
    old = module.read_text()
    new = old.replace(
        "contains\n",
        """integer, allocatable :: mo_nonzero_count(:), mo_nonzero_index(:,:)
contains
""",
        1,
    )
    new = new.replace(
        "integer :: row, axis, power, odd, product",
        "integer :: row, axis, power, odd, product, orbital, ao, entry",
    )
    new = new.replace(
        "end subroutine initialize_static_gto",
        """  if (allocated(mo_nonzero_count)) deallocate(mo_nonzero_count, mo_nonzero_index)
  allocate(mo_nonzero_count(nstates), mo_nonzero_index(coefftot,nstates))
  do orbital = 1, nstates
    entry = 0
    do ao = 1, coefftot
      if (cfmo(ao,orbital) == 0.d0) cycle
      entry = entry + 1
      mo_nonzero_index(entry,orbital) = ao
    enddo
    mo_nonzero_count(orbital) = entry
  enddo
end subroutine initialize_static_gto""",
    )
    changed[module] = new
    for name in ("orbden_1.f90", "orbgrad_1.f90"):
        filename = cache / name
        old = filename.read_text()
        new = old.replace(
            "integer nfbtot_bas",
            """integer nfbtot_bas
integer power, entry
real*8 xp(0:4,natomx), yp(0:4,natomx), zp(0:4,natomx)""",
            1,
        )
        anchor = "  rr(iat) = x1at(iat)**2 + y1at(iat)**2"
        new = new.replace(
            anchor,
            """  if (nflag == 0) then
    xp(0,iat) = 1.d0
    yp(0,iat) = 1.d0
    zp(0,iat) = 1.d0
    do power = 1, 4
      xp(power,iat) = xp(power-1,iat)*x1at(iat)
      yp(power,iat) = yp(power-1,iat)*y1at(iat)
      zp(power,iat) = zp(power-1,iat)*z1at(iat)
    enddo
  endif
"""
            + anchor,
            1,
        )
        for axis, index in (("x", "1"), ("y", "2"), ("z", "3")):
            new = re.sub(
                axis + r"1at\(iat\)\*\*\(L" + index + r"-1\)",
                axis + "p(L" + index + "-1,iat)",
                new,
                flags=re.I,
            )
            new = re.sub(
                axis + r"1at\(iat\)\*\*L" + index, axis + "p(L" + index + ",iat)", new, flags=re.I
            )
        # Preserve increasing AO summation order. Dense MOs keep the original
        # loop, avoiding indirect-index overhead when sparsity is insufficient.
        start = new.rindex("do iorb=1,nstates")
        finish = new.index("end subroutine", start)
        original_loop = new[start:finish].rstrip()
        lines = original_loop.splitlines(True)
        body = "".join(lines[2:-2])
        if "do ifb=1,nfbtot_bas" not in original_loop:
            raise ValueError("Unknown MO contraction layout")
        new_loop = (
            """do iorb=1,nstates
  if (mo_nonzero_count(iorb)*4 < nfbtot_bas*3) then
    do entry=1,mo_nonzero_count(iorb)
      ifb=mo_nonzero_index(entry,iorb)
"""
            + body
            + """    enddo
  else
    do ifb=1,nfbtot_bas
"""
            + body
            + """    enddo
  endif
enddo

"""
        )
        changed[filename] = new[:start] + new_loop + new[finish:]
    metadata = write_overlay(changed, destination, source_root=cache, digest_field="input_sha256")
    return dict(
        changes=metadata, scope="Cartesian powers reused; only exactly zero MO coefficients omitted"
    )
