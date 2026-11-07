"""Export unchanged reference response matrices and continuum contractions.

The overlay is restricted to one MPI process to make array ordering explicit.
It instruments the separately installed reference, never production SPARKS.
"""

from pathlib import Path
import numpy as np
from .equivalence import _once
from .files import write_overlay


def prepare_response_export(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    changes = {}
    path = source / "tddft/mpichi_s1_M.f90"
    text = path.read_text()
    text = _once(
        text,
        "      integer ::  itestw, myfile , j, iout",
        """      integer ::  itestw, myfile , j, iout
      integer, save :: response_export_index=0
      integer :: response_export_unit
      character(len=96) :: response_export_file""",
    )
    anchor = "      call pdgetrf(nht2, nht2, ht, 1, 1, mdeschh, ind, info)"
    export = """      if (nprow /= 1 .or. npcol /= 1) stop 'Response export requires serial MPI layout'
      response_export_index=response_export_index+1
      write(response_export_file,'(a,i6.6,a)') 'response-',response_export_index,'.bin'
      open(newunit=response_export_unit,file=trim(response_export_file), &
           access='stream',form='unformatted',status='new')
      write(response_export_unit) nhtd,idip
      write(response_export_unit) dmatr(1:nhtd,1:nhtd),dmati(1:nhtd,1:nhtd),hk(1:nhtd,1:nhtd)
      write(response_export_unit) vext(1:nhtd,1)
"""
    before_fields = "            vext = zero"
    header, _ = export.split("      write(response_export_unit) vext", 1)
    text = _once(text, before_fields, header + before_fields)
    text = _once(text, anchor, "      write(response_export_unit) vext(1:nhtd,1)\n" + anchor)
    anchor = "        if (info.ne.0) write(*,*) ' in chi pdgetrs info = ', info"
    text = _once(
        text,
        anchor,
        anchor
        + """
      write(response_export_unit) info,vscf(1:nht2,1)
      close(response_export_unit)
""",
    )
    changes[path] = text
    path = source / "tddft/partial_2_M.f90"
    text = path.read_text()
    anchor = "      complex*16  dlm, bi, bjt"
    text = _once(
        text,
        anchor,
        anchor
        + """
      integer, save :: vertex_export_index=0
      integer :: vertex_export_unit
      character(len=96) :: vertex_export_file
""",
    )
    anchor = "         allocate (x1(nht))"
    text = _once(
        text,
        anchor,
        anchor
        + """
         vertex_export_index=vertex_export_index+1
         write(vertex_export_file,'(a,i6.6,a)') 'vertex-',vertex_export_index,'.bin'
         open(newunit=vertex_export_unit,file=trim(vertex_export_file), &
              access='stream',form='unformatted',status='new')
         write(vertex_export_unit) nht,nasy,idip,jb,ia,iprin,i
         write(vertex_export_unit) omega,ekin,const
         write(vertex_export_unit) csinv,vscf(1:2*nht,idip)
""",
    )
    anchor = "            read(idip0a) x1(1:nht)"
    text = _once(text, anchor, anchor + "\n            write(vertex_export_unit) x1(1:nht)")
    anchor = "          deallocate(x1)"
    text = _once(
        text,
        anchor,
        """         write(vertex_export_unit) x2
         close(vertex_export_unit)
"""
        + anchor,
    )
    changes[path] = text
    # Export raw Galerkin data, before factorization or response assembly.
    # These records permit independent Green-function/susceptibility tests.
    path = source / 'tddft/mpihomog_2_M.f90'
    text = path.read_text()
    anchor = '      real(double) :: ekin, ecorr, ee,'
    line = next(line for line in text.splitlines() if line.startswith(anchor))
    text = _once(text, line, '''      integer :: response_raw_unit, response_raw_knots
      logical :: response_raw_exists
      character(len=128) :: response_raw_file
''' + line)
    anchor = '        iprint1 = iprint(3)'
    export = '''        if (nprow /= 1 .or. npcol /= 1) stop 'Raw response export requires one rank'
        write(response_raw_file,'(a,i4.4,a)') 'response-operators-',ia,'.bin'
        inquire(file=trim(response_raw_file),exist=response_raw_exists)
        if (.not. response_raw_exists) then
          response_raw_knots=maxval(nspla(1:nvec,ia))+iord+1
          open(newunit=response_raw_unit,file=trim(response_raw_file), &
               access='stream',form='unformatted',status='new')
          write(response_raw_unit) nht,nvec,response_raw_knots,iord
          write(response_raw_unit) zinf,slb94
          write(response_raw_unit) ilf(1:nvec,isyma(ia)),nspla(1:nvec,ia)
          write(response_raw_unit) tkn(1:response_raw_knots),smat(1:nht,1:nht),hmat(1:nht,1:nht)
          close(response_raw_unit)
        endif
'''
    text = _once(text, anchor, export + anchor)
    anchor = '         call mpiinhomog( nht, omat, omat2, scr, scr(n1), scr(n2),'
    line = next(line for line in text.splitlines() if line.startswith(anchor))
    export = '''         write(response_raw_file,'(a,3(i5.5,a))') 'response-source-',ia,'-',kmpd,'-',jb,'.bin'
         inquire(file=trim(response_raw_file),exist=response_raw_exists)
         if (.not. response_raw_exists) then
           open(newunit=response_raw_unit,file=trim(response_raw_file), &
                access='stream',form='unformatted',status='new')
           write(response_raw_unit) nht,nhtd,ia,jb,kappad(kmpd),nvec,klpd,kmpd
           write(response_raw_unit) ebshift(jb),zocc(jb)
           write(response_raw_unit) reshape(scr(1:ldi*ldjd),[ldi,ldjd])
           close(response_raw_unit)
         endif
'''
    text = _once(text, line, export + line)
    changes[path] = text
    return dict(
        changes=write_overlay(changes, destination, source_root=source),
        scope="Unchanged serial raw H/S/sources, TDDFT chi, kernel, effective field and continuum vertices",
    )


def read_screening_export(path):
    """Read one checked little-endian GNU reference stream."""
    with Path(path).open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=2)
        if len(header) != 2 or header[0] <= 0 or header[1] not in (1, 2, 3):
            raise ValueError("Invalid reference response header")
        n, idip = map(int, header)
        raw = np.fromfile(stream, dtype="<f8", count=3 * n * n + n)
        if len(raw) != 3 * n * n + n:
            raise ValueError("Truncated reference response matrices")
        info = np.fromfile(stream, dtype="<i4", count=1)
        effective = np.fromfile(stream, dtype="<f8", count=2 * n)
        if len(info) != 1 or info[0] != 0 or len(effective) != 2 * n or stream.read(1):
            raise ValueError("Invalid or incomplete reference response solution")
    chi = raw[: n * n].reshape(n, n, order="F") + 1j * raw[n * n : 2 * n * n].reshape(
        n, n, order="F"
    )
    return dict(
        dipole=idip,
        chi=chi.conjugate(),
        kernel=raw[2 * n * n : 3 * n * n].reshape(n, n, order="F"),
        external_potential=raw[3 * n * n :, None],
        effective_potential=-(effective[:n] - 1j * effective[n:])[:, None],
        native_effective_potential=(effective[:n] + 1j * effective[n:])[:, None],
        convention="chi=conj(native chi), V=-conj(native V); positive Cartesian r driving",
    )


def read_vertex_export(path):
    with Path(path).open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=7)
        if len(header) != 7 or np.any(header <= 0):
            raise ValueError("Invalid reference vertex header")
        n, nc, idip, orbital, symmetry, response_kind, component = map(int, header)
        raw = np.fromfile(stream, dtype="<f8")
    if len(raw) != 3 + nc * nc + 2 * n + nc * n + 2 * nc:
        raise ValueError("Invalid reference vertex length")
    a = 3
    inverse = raw[a : a + nc * nc].reshape(nc, nc, order="F")
    a += nc * nc
    effective = raw[a : a + n] + 1j * raw[a + n : a + 2 * n]
    a += 2 * n
    vertices = raw[a : a + nc * n].reshape(nc, n)
    a += nc * n
    contracted = raw[a : a + nc] + 1j * raw[a + nc : a + 2 * nc]
    return dict(
        dipole=idip,
        orbital=orbital,
        symmetry=symmetry,
        response_kind=response_kind,
        component=component,
        photon_energy_hartree=raw[0],
        electron_energy_hartree=raw[1],
        normalization=raw[2],
        inverse_matching=inverse,
        effective_potential=effective,
        vertices=vertices,
        contracted=contracted,
    )


def read_serial_kernel_archive(path, dimensions):
    """Read the retained MPI kernel file for its explicit one-rank layout.

    The legacy work array aliases K and V; archived kernels are immutable and
    can recover complete reference K after the screening solve has reused that
    work array. Dimensions are supplied in Cartesian y,z,x dipole order.
    """
    dimensions = tuple(dimensions)
    if len(dimensions) != 3 or any(type(n) is not int or n <= 0 for n in dimensions):
        raise ValueError("Require three positive serial kernel dimensions")
    raw = np.fromfile(path, dtype="<f8")
    if len(raw) != sum(n * n for n in dimensions) or not np.all(np.isfinite(raw)):
        raise ValueError("Kernel archive size differs from the serial layout")
    output = []
    offset = 0
    for n in dimensions:
        output.append(raw[offset : offset + n * n].reshape(n, n, order="F"))
        offset += n * n
    return tuple(output)
