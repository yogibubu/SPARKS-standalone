"""Read-only numerical exports from an isolated TIRESIA reference build.

Extra files contain operators, sources and raw solutions. Original arithmetic,
inputs and checkpoints are preserved. Production SPARKS does not import this
module or execute the reference code.
"""

from pathlib import Path

from .files import write_overlay


def _once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError(f"Reference source layout changed: {anchor[:75]}")
    return text.replace(anchor, replacement, 1)


def _source_declarations():
    return """
      logical, allocatable, save :: eq_written(:)
      logical :: eq_active
      real(double), allocatable :: eq_source(:,:,:,:)
      integer :: eq_unit
      character(len=128) :: eq_filename
"""


def _source_start():
    return """
      if (.not. allocated(eq_written)) then
        allocate(eq_written(nsymmx))
        eq_written = .false.
      endif
      eq_active = .not. eq_written(ia)
      if (eq_active) then
        allocate(eq_source(nhtmxa,ndg(isyma(ia)),3,nvdtotb))
        eq_source = 0.d0
      endif
"""


def _source_end(prefix):
    return f"""
      if (eq_active) then
        write(eq_filename,'(a,i4.4,a,i4.4,a)') '{prefix}-',ia,'-r',myid,'.bin'
        open(newunit=eq_unit,file=trim(eq_filename),access='stream',form='unformatted',status='replace')
        write(eq_unit) nhtmxa,ndg(isyma(ia)),nvdtotb
        write(eq_unit) eq_source
        close(eq_unit)
        eq_written(ia) = .true.
      endif
"""


def prepare_equivalence_export(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    changes = {}
    path = source / "lcao/mpihsbuild_1.f90"
    text = path.read_text()
    anchor = "!     write(*,*) 'HSBUILD, punto 4'"
    export = """
      block
        integer, save :: eq_index=0
        integer :: eq_unit
        character(len=128) :: eq_filename
        eq_index=eq_index+1
        write(eq_filename,'(a,i4.4,a,i4.4,a)') 'eq-hs-',eq_index,'-r',myid,'.bin'
        open(newunit=eq_unit,file=trim(eq_filename),access='stream',form='unformatted',status='replace')
        write(eq_unit) nht,ldi,ldj,nprow,npcol,ipr,ipc,nbr,nbc
        write(eq_unit) at,bt
        close(eq_unit)
      end block
"""
    changes[path] = _once(text, anchor, export + "\n" + anchor)
    path = source / "lcao/mpihdiagclc_b.f90"
    text = path.read_text()
    anchor = """      ENDIF
      END"""
    export = """
      block
        integer, save :: eq_index=0
        integer :: eq_unit,eq_knots
        character(len=128) :: eq_filename
        eq_index=eq_index+1
        eq_knots=maxval(nspl)+iord+1
        write(eq_filename,'(a,i5.5,a)') 'eq-vector-',eq_index,'.bin'
        open(newunit=eq_unit,file=trim(eq_filename),access='stream',form='unformatted',status='replace')
        write(eq_unit) nht,nvec,nangf,eq_knots,iord,ninter
        write(eq_unit) e,zinf
        write(eq_unit) ilf,nspl,tkn(:eq_knots)
        write(eq_unit) vec,csinv,cs,cc
        close(eq_unit)
        write(eq_filename,'(a,i5.5,a)') 'eq-matching-',eq_index,'.bin'
        open(newunit=eq_unit,file=trim(eq_filename),access='stream',form='unformatted',status='replace')
        write(eq_unit) e,zinf,slb94
        write(eq_unit) ifit,itype
        close(eq_unit)
      end block
"""
    changes[path] = _once(text, anchor, export + anchor)
    path = source / "lcao/mpidipole_1.f90"
    text = path.read_text()
    anchor = "      real(double) :: c, xdd3 "
    text = _once(text, anchor, anchor + "\n" + _source_declarations())
    text = _once(text, "      dipl = zero ", _source_start() + "\n      dipl = zero ")
    anchor = "                        xdd3 = xdd(icomp) "
    text = _once(
        text,
        anchor,
        anchor
        + """
                        if (eq_active) eq_source(ipa+i1:ipa+i2,ka,mg,kbg) = &
                            eq_source(ipa+i1:ipa+i2,ka,mg,kbg)+xd(i1:i2)*xdd3
""",
    )
    anchor = "      IF (IMONO .NE. 0)"
    # Only the first clause is active; preserve trailing continuation spacing.
    if text.count(anchor) != 1:
        raise ValueError("Unexpected central dipole reduction layout")
    text = text.replace(anchor, _source_end("eq-source-central") + anchor, 1)
    changes[path] = text
    path = source / "lcao/mpidipol1_1.f90"
    text = path.read_text()
    anchor = "      real(double) :: dd, dd0, dd1"
    text = _once(text, anchor, anchor + "\n" + _source_declarations())
    anchor = "             IPROC = -1"
    text = _once(text, anchor, _source_start() + "\n" + anchor)
    anchor = "                   DD = DD + CA(NALC+N, NVA+IVA) * CD2(IP2 + N)"
    if text.count(anchor) != 1:
        raise ValueError("Unexpected local dipole contraction")
    line = next(line for line in text.splitlines() if line.startswith(anchor))
    text = _once(
        text,
        line,
        line
        + """
                   if (eq_active .and. iva==1) eq_source(nalc+n,mua,kappa,kvdb) = &
                       eq_source(nalc+n,mua,kappa,kvdb)+cd2(ip2+n)
""",
    )
    anchor = "                    DD = DD + CA(NA+N, NVA+IVA) * CD1(NN)"
    line = next(line for line in text.splitlines() if line.startswith(anchor))
    text = _once(
        text,
        line,
        line
        + """
                    if (eq_active .and. iva==1) eq_source(na+n,mua,kappa,kvdb) = &
                        eq_source(na+n,mua,kappa,kvdb)+cd1(nn)
""",
    )
    anchor = "      CALL MPIREDMATSUM (DIPL, SCR, NVDTOTA*3, NVDTOTB, 0, MYID)"
    changes[path] = _once(text, anchor, _source_end("eq-source-local") + anchor)
    return dict(
        changes=write_overlay(changes, destination, source_root=source),
        scope="Export unchanged reference H/S, physical sources and solutions for equivalence checks",
    )
