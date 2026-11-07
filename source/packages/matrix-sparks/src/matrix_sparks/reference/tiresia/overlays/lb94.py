"""Export unchanged CO density, XC values and angular projections for qualification."""

from pathlib import Path
from .files import write_overlay
from .equivalence import _once


def prepare_lb94_export(source: Path, native_module: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    changes = {}
    text = native_module.read_text()
    text = _once(
        text,
        "subroutine initialize_sparks_grid()\n",
        "subroutine initialize_sparks_grid()\n  use mympi, only: myid\n",
    )
    text = _once(
        text,
        "  integer, parameter :: first(0:4)",
        "  integer :: lb_unit\n  character(len=128) :: lb_name\n  integer, parameter :: first(0:4)",
    )
    anchor = "  if (ao/=coefftot) stop 'SPARKS AO dimension mismatch'"
    text = _once(
        text,
        anchor,
        anchor
        + """
  write(lb_name,'(a,i4.4,a)') 'eq-lb94-grid-r',myid,'.bin'
  open(newunit=lb_unit,file=trim(lb_name),access='stream',form='unformatted',status='replace')
  write(lb_unit) natom,ns,coefftot,nstates,ne
  write(lb_unit) at,bs,loc,ev,cf,occno
  close(lb_unit)
""",
    )
    changes[native_module] = text
    path = source / "lcao/mpiinte_1c_2_new.f90"
    text = path.read_text()
    declaration = "      real(double), allocatable  :: plg(:,:), csmphi(:)"
    text = _once(
        text,
        declaration,
        declaration + "\n      integer :: lb_points,lb_unit\n      character(len=128) :: lb_name",
    )
    anchor = "      coxa     ="
    text = _once(
        text,
        anchor,
        """      write(lb_name,'(a,i4.4,a)') 'eq-lb94-points-r',myid,'.bin'
      open(newunit=lb_points,file=trim(lb_name),access='stream',form='unformatted',status='replace')
"""
        + anchor,
    )
    # Both original density backends; all existing arithmetic is retained.
    anchor = "               call exchg ( ixc, alph, coxa, den, grad, pxc, wsaop, wresp )"
    if text.count(anchor) != 2:
        raise ValueError("Reference XC call layout changed")
    text = text.replace(anchor, anchor + "\n               write(lb_points) x,y,z,den,grad,pxc")
    anchor = "            endif \n\n           if (idens .eq. 0) then"
    text = _once(
        text,
        anchor,
        """              if (myid==0) then
                write(lb_name,'(a,i4.4,a)') 'eq-lb94-angular-',ireg,'.bin'
                open(newunit=lb_unit,file=trim(lb_name),access='stream',form='unformatted',status='replace')
                write(lb_unit) n,nangv
                write(lb_unit) xa(:n),ya(:n),za(:n),p1(:,:n),ilv(:nangv)
                close(lb_unit)
              endif
"""
        + anchor,
    )
    anchor = "       deallocate (sa)"
    text = _once(
        text,
        anchor,
        anchor
        + """
      close(lb_points)
      if (myid==0) then
        open(newunit=lb_unit,file='eq-lb94-projection.bin',access='stream',form='unformatted',status='replace')
        write(lb_unit) nangv,nint,ngauss,iord,nsplmx
        write(lb_unit) xg,wg,tkn(:nsplmx+iord+1),s,si
        close(lb_unit)
      endif
""",
    )
    changes[path] = text
    # The multicentre Poisson path uses MPIINTE_P rather than MPIINTE_1C.
    path = source / "lcao/mpiinte_p_3_new_M.f90"
    text = path.read_text()
    declaration = "      real(double), allocatable  :: plg(:,:), csmphi(:)"
    text = _once(
        text,
        declaration,
        declaration
        + "\n      integer :: lb_points,lb_unit,lb_region\n      character(len=128) :: lb_name",
    )
    anchor = "      itest = iprint(37)"
    text = _once(
        text,
        anchor,
        """      lb_region=0
      write(lb_name,'(a,i4.4,a)') 'eq-lb94-points-r',myid,'.bin'
      open(newunit=lb_points,file=trim(lb_name),access='stream',form='unformatted',status='replace')
"""
        + anchor,
    )
    anchor = "               call exchg ( ixc, alph, coxa, den, grad, pxc, wsaop, wresp )"
    if text.count(anchor) != 2:
        raise ValueError("Multicentre reference XC call layout changed")
    text = text.replace(anchor, anchor + "\n               write(lb_points) x,y,z,den,grad,pxc")
    anchor = "           if (idens .eq. 0) then"
    text = _once(
        text,
        anchor,
        """           if (myid==0 .and. lb_region/=ireg) then
             write(lb_name,'(a,i4.4,a)') 'eq-lb94-angular-',ireg,'.bin'
             open(newunit=lb_unit,file=trim(lb_name),access='stream',form='unformatted',status='replace')
             write(lb_unit) n,nangv
             write(lb_unit) xa(:n),ya(:n),za(:n),p1(:,:n),ilv(:nangv)
             close(lb_unit)
             lb_region=ireg
           endif
"""
        + anchor,
    )
    anchor = "      si = zero"
    text = _once(
        text,
        anchor,
        """      close(lb_points)
      if (myid==0) then
        open(newunit=lb_unit,file='eq-lb94-projection.bin',access='stream',form='unformatted',status='replace')
        write(lb_unit) nangv,nint,ngauss,iord,nsplmx
        write(lb_unit) xg,wg,tkn(:nsplmx+iord+1),s,s
        close(lb_unit)
      endif
"""
        + anchor,
    )
    changes[path] = text
    return dict(
        changes=write_overlay(changes, destination),
        scope="Read-only export of existing density/XC quadrature and libcint AO arrays",
    )
