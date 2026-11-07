"""Prepare a source-level cache for Tiresia's GTO grid evaluator.

Data are refreshed after every orbital-file read. Coordinates, radial values,
orbital values, gradients and continuum matrices are never cached across points
or geometries. MPI retains its original process-local module state.
"""

import re
from pathlib import Path

from .files import write_overlay


def prepare_gto_cache(source: Path, destination: Path) -> dict:
    destination.mkdir(exist_ok=True)
    changed = {}
    module = source / "monew/mod_orbread_n.f90"
    text = module.read_text()
    declarations = """
! MATRIX: quantities independent of the spatial quadrature point.
real*8 :: cartesian_normalization(35)
real*8, allocatable :: density_orbitals(:), gradient_orbitals(:), orbital_gradients(:,:)
contains
subroutine initialize_static_gto()
  implicit none
  integer :: row, axis, power, odd, product
  cfmo = reshape(array_cfmo, shape(cfmo))
  do row = 1, size(cartesian_normalization)
    product = 1
    do axis = 1, 3
      power = Lval(row, axis)
      do odd = 1, 2*power-1, 2
        product = product * odd
      enddo
    enddo
    cartesian_normalization(row) = 1.d0 / sqrt(real(product, kind(1.d0)))
  enddo
  if (allocated(density_orbitals)) deallocate(density_orbitals)
  if (allocated(gradient_orbitals)) deallocate(gradient_orbitals)
  if (allocated(orbital_gradients)) deallocate(orbital_gradients)
  allocate(density_orbitals(nstates), gradient_orbitals(nstates), orbital_gradients(nstates,3))
end subroutine initialize_static_gto
"""
    new, count = re.subn(r"(?im)^end module\s*$", declarations + "\nend module", text)
    if count != 1:
        raise ValueError("Unexpected GTO module layout")
    changed[module] = new
    for name in ("orbread_no_1.f90", "orbread_n_1.f90"):
        filename = source / "monew" / name
        original = filename.read_text()
        start = original.index("array_cfmo(i)=num")
        end = re.search(r"(?im)^\s*enddo\s*$", original[start:])
        if end is None:
            raise ValueError("Missing orbital-file coefficient loop")
        index = start + end.end()
        changed[filename] = original[:index] + "\ncall initialize_static_gto()\n" + original[index:]
    copy_pattern = (
        r"(?im)^i=1\s*\n"
        r"do iorb=1,nstates\s*\n"
        r"\s*do ifb=1,coefftot\s*\n"
        r"\s*cfmo\(ifb,iorb\)=array_cfmo\(i\)\s*\n"
        r"\s*i=i\+1\s*\n\s*enddo\s*\nenddo\s*$"
    )
    normalization = (
        r"(?im)^\s*nl1=2\*l1-1\s*\n\s*nl2=2\*l2-1\s*\n"
        r"\s*nl3=2\*l3-1\s*\n"
        r"\s*nprod = semifat\(nl1,nsfl1\) \* semifat\(nl2,nsfl2\) \* semifat\(nl3,nsfl3\)\s*\n"
    )
    for name in ("orbden_1.f90", "orbgrad_1.f90", "orbval_n_1.f90"):
        filename = source / "monew" / name
        new, count = re.subn(
            copy_pattern,
            "! MATRIX: cfmo prepared when the orbital file is read.\n",
            filename.read_text(),
        )
        if count != 1:
            raise ValueError(f"Coefficient copy layout differs: {filename}")
        if name == "orbgrad_1.f90":
            new, count = re.subn(
                normalization + r"\s*nprod = 1 / sqrt \( nprod \)\s*\n",
                "\n      nprod = cartesian_normalization(row)\n",
                new,
            )
        else:
            new, count = re.subn(
                normalization + r"\s*fang = fang \* 1 / sqrt \( nprod \)\s*\n",
                "\n      fang = fang * cartesian_normalization(row)\n",
                new,
            )
        if count != 1:
            raise ValueError(f"Normalization layout differs: {filename}")
        changed[filename] = new
    for name, names in [
        ("gdensity_1.f90", {"orbout": "density_orbitals"}),
        ("ggradient_1.f90", {"orbout": "gradient_orbitals", "gradout": "orbital_gradients"}),
    ]:
        filename = source / "monew" / name
        lines = []
        for line in filename.read_text().splitlines(True):
            if re.search(r"(?i)allocatable\s*::\s*orbout", line):
                continue
            if re.search(r"(?i)^\s*(?:deallocate|allocate)\s*\(\s*(?:orbout|gradout)", line):
                continue
            for original, replacement in names.items():
                line = re.sub(r"\b" + original + r"\b", replacement, line)
            lines.append(line)
        changed[filename] = "".join(lines)
    metadata = write_overlay(changed, destination, source_root=source)
    return dict(
        changes=metadata,
        scope="MO layout, Cartesian constants and reusable grid-point workspace; no quadrature or physical model change",
    )
