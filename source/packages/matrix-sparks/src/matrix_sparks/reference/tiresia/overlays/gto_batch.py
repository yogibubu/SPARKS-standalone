"""Connect Tiresia quadratures to the general SPARKS/libcgto batch grid core."""

from pathlib import Path

from ..resources import fortran_source
from .files import write_overlay

DECLARATIONS = fortran_source("tiresia_gto_batch_declarations.f90")


ROUTINES = fortran_source("tiresia_gto_batch_routines.f90")


def prepare_gto_batch(source: Path, cache: Path, fast: Path, destination: Path) -> dict:
    destination.mkdir(exist_ok=True)
    changed = {}
    module = fast / "mod_orbread_n.f90"
    new = module.read_text().replace("implicit none", "use iso_c_binding\nimplicit none", 1)
    new = new.replace("contains\n", DECLARATIONS + "\ncontains\n", 1)
    new = new.replace(
        "end subroutine initialize_static_gto",
        "  call initialize_sparks_grid()\nend subroutine initialize_static_gto",
    )
    new = new.replace("\nend module", ROUTINES + "\nend module")
    changed[module] = new
    for name in ("gdensity_1.f90", "ggradient_1.f90"):
        filename = cache / name
        new = filename.read_text()
        start = new.index("      den =")
        fields = "        den=sparks_density(sparks_cursor)\n"
        if name.startswith("ggradient"):
            fields += "        gradabs=sparks_gradient(sparks_cursor)\n        wsaop=0.d0\n        wresp=0.d0\n"
        new = (
            new[:start]
            + """      if (sparks_cursor>0) then
        if (sparks_cursor>size(sparks_density)) stop 'SPARKS batch overrun'
"""
            + fields
            + """        sparks_cursor=sparks_cursor+1
        return
      endif
"""
            + new[start:]
        )
        changed[filename] = new
    for name, derivative, condition in (
        ("mpiint2clap_1.f90", 0, "idens/=0"),
        ("mpiint2cenxx_1.f90", 1, "idens/=0 .and. imol/=0"),
    ):
        filename = source / "lcao" / name
        new = filename.read_text().replace(
            "      implicit none",
            "      use modul_orbread, only: sparks_prefill_two_center, sparks_batch_clear\n      implicit none",
            1,
        )
        anchor = "         do ipr = 1, nprr"
        if anchor not in new:
            raise ValueError(f"Unknown radial loop: {filename}")
        new = new.replace(
            anchor,
            f"""         if ({condition}) call sparks_prefill_two_center(costh,sinth,nprr,xrd,nph,npph, &
             cosphi,sinphi,x1,y1,z1,ausi,{derivative})
"""
            + anchor,
            1,
        )
        # Clear exactly after the radial loop, never while its scalar consumers run.
        import re

        new, count = re.subn(
            r"(?im)^(\s*end do\s*!\s*ipr\s*\n)",
            r"\1         call sparks_batch_clear()\n",
            new,
            count=1,
        )
        if count != 1:
            raise ValueError(f"Unknown end of radial loop: {filename}")
        changed[filename] = new
    metadata = write_overlay(changed, destination, digest_field="input_sha256")
    return dict(
        changes=metadata,
        scope="Batch existing two-centre quadratures through SPARKS/libcgto and BLAS",
    )
