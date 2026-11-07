"""Check Tiresia-to-libcgto ordering and normalization against upstream GTOs."""

import argparse
import json
import socket
import subprocess
from pathlib import Path

from matrix_sparks.reference.tiresia.qualification.gto_fast import DRIVER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower() != "oracle":
        raise RuntimeError("Run on Oracle")
    import pyscf

    args.output.mkdir(exist_ok=True, parents=True)
    driver = DRIVER.replace(
        "integer i,j,k,t,n,seed_size", "integer i,j,k,t,n,seed_size,status,axis"
    )
    driver = driver.replace(
        "natom=2",
        """real(c_double) xyz(200,3),density(200),gradient(200),refden(200),refgrad(200),gradvec(3)
natom=2""",
        1,
    )
    driver = driver.replace(
        "largest=0.d0", "occno=(/2.d0,2.d0,2.d0,2.d0,1.d0,1.d0,0.d0,0.d0/)\nlargest=0.d0", 1
    )
    driver = driver.replace(
        "  point=(point-0.5d0)*8.d0", "  point=(point-0.5d0)*8.d0\n  xyz(n,:)=point", 1
    )
    anchor = "  if(err>5.d-13) stop 3\n"
    driver = driver.replace(
        anchor,
        anchor
        + """  refden(n)=sum(occno*oldmo**2)
  do axis=1,3
    gradvec(axis)=sum(2.d0*occno*oldmo*oldgrad(:,axis))
  enddo
  refgrad(n)=sqrt(sum(gradvec**2))
""",
        1,
    )
    anchor = "enddo\nwrite(*,*) 'passed 600 points; maximum scaled error',largest"
    driver = driver.replace(
        anchor,
        """ status=sparks_evaluate(sparks_context,200,xyz,density,gradient,1)
 if(status/=0)stop 4
 err=maxval(abs(density-refden)/(1.d0+abs(refden)))
 largest=max(largest,err)
 if(err>5.d-13)stop 5
 err=maxval(abs(gradient-refgrad)/(1.d0+abs(refgrad)))
 largest=max(largest,err)
 if(err>5.d-13)stop 6
enddo
write(*,*) 'passed 600 points; maximum scaled error',largest""",
        1,
    )
    # Keep numerical checks; omit the earlier scalar performance microbenchmark.
    driver = (
        driver[: driver.index("call random_number(array_cfmo)", driver.index("'passed 600"))]
        + "end program\n"
    )
    driver_file = args.output / "driver.f90"
    driver_file.write_text(driver)
    sources = [args.source / "gnu_compat/gto_batch/mod_orbread_n.f90"]
    for name in ("orbden", "orbgrad"):
        reference = args.output / (name + "_reference.f90")
        reference.write_text(
            (args.source / "monew" / (name + "_1.f90"))
            .read_text()
            .replace("subroutine " + name, "subroutine " + name + "_reference")
        )
        sources.extend([reference, args.source / "gnu_compat/gto_fast" / (name + "_1.f90")])
    sources.extend(args.source / "lib" / name for name in ("plm.f90", "csphi1.f90"))
    sources.append(args.source / "monew/function.f90")
    cgto = Path(pyscf.__file__).parent / "lib"
    native = Path(
        json.loads((args.source / "gnu_compat/gto_batch/native-library.json").read_text())[
            "library_path"
        ]
    )
    binary = args.output / "qualify.exe"
    subprocess.run(
        [
            "/home/enzo/opt/openmpi/4.1.8/bin/mpifort",
            "-O2",
            "-fcheck=all",
            "-ffree-line-length-none",
            "-I",
            str(args.source / "lcaompi_obj"),
            "-o",
            str(binary),
            *map(str, sources),
            str(driver_file),
            str(native),
            "-Wl,-rpath," + str(native.parent),
            "-Wl,-rpath-link," + str(cgto),
            "-Wl,-rpath-link," + str(cgto / "deps/lib"),
        ],
        cwd=args.output,
        check=True,
    )
    result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
    report = dict(
        host=socket.gethostname(),
        bounds_checking=True,
        tolerance_scaled=5e-13,
        points=600,
        cartesian_shells="s–g",
        coefficients="dense, sparse and zero",
        output=result.stdout,
    )
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
