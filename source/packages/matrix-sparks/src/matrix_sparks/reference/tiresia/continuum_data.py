"""TIRESIA continuum checkpoint reader for independent equivalence checks.

This reads TIRESIA's own unformatted output, not electronic-structure inputs.
The supported checkpoint is the real static one-particle (ICALTYPE=0) layout.
References: sigma1c_2.f90, sdrive0.f90 and dxtran.f90 in TIRESIA 1.0.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import FortranEOFError, FortranFile


@dataclass(frozen=True)
class ReferenceSymmetry:
    identifier: int
    name: str
    angular_momenta: np.ndarray
    degeneracy: int
    # Real harmonic expansion of each (channel,degenerate component).
    angular_expansions: tuple


@dataclass(frozen=True)
class ReferenceFrame:
    energy: float
    k_eigenvalues: tuple[np.ndarray, ...]
    k_eigenvectors: tuple[np.ndarray, ...]
    # Each symmetry: (degenerate component, Cartesian dipole, channel, bound).
    amplitudes: tuple[np.ndarray, ...]
    scattering: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ReferenceContinuum:
    residual_charge: float
    bound_energies: np.ndarray
    symmetries: tuple[ReferenceSymmetry, ...]
    frames: tuple[ReferenceFrame, ...]
    group: str
    bound_component_parents: np.ndarray | None = None

    def cross_sections(self, *, occupations, alpha=7.29735308e-3, bohr2_mb=0.529177249**2 * 100):
        """Isotropic Mb using explicit bound-state occupations, no hidden factor2."""
        occupations = np.asarray(occupations, dtype=float)
        if occupations.shape != self.bound_energies.shape or np.any(occupations < 0):
            raise ValueError("Provide an occupation for every stored bound state")
        output = []
        for frame in self.frames:
            strength = sum(np.sum(abs(a) ** 2, axis=(0, 1, 2)) for a in frame.amplitudes)
            if self.bound_component_parents is not None:
                strength = np.bincount(
                    self.bound_component_parents,
                    weights=strength,
                    minlength=len(self.bound_energies),
                )
            output.append(
                4
                * np.pi**2
                * alpha
                / 3
                * (frame.energy - self.bound_energies)
                * strength
                * occupations
                * bohr2_mb
            )
        return np.asarray(output)


def read_continuum(path: Path | str):
    """Read native little-endian 4-byte-integer/8-byte-real TIRESIA checkpoints."""
    with FortranFile(path, "r", header_dtype=np.dtype("<u4")) as stream:

        def record():
            return stream.read_record(np.uint8).tobytes()

        first = record()
        if len(first) != 32 or np.frombuffer(first[-4:], dtype="<i4")[0] != 0:
            raise ValueError("Only the TIRESIA static one-particle fort.20 layout is supported")
        group = first[20:28].decode().strip()
        dimensions = record()
        if len(dimensions) != 56:
            raise ValueError("Unsupported integer precision/header layout")
        a = np.frombuffer(dimensions[:28], dtype="<i4")
        nvda, nvdb, nsa, nsb, nvmax, lmax, jdim = map(int, a)
        charge = float(np.frombuffer(dimensions[28:36], dtype="<f8")[0])
        nsym, nsub, idim, nvta, nvtb = map(int, np.frombuffer(dimensions[36:], dtype="<i4"))
        if min(nvda, nvdb, nsa, nsb, nvmax, nsym, nsub, idim, nvtb) <= 0:
            raise ValueError("Invalid checkpoint dimensions")
        raw = record()
        offset = 0

        def ints(count, shape=None):
            nonlocal offset
            values = np.frombuffer(raw, dtype="<i4", count=count, offset=offset).copy()
            offset += 4 * count
            return values if shape is None else values.reshape(shape, order="F")

        ilf = ints(idim * nsym, (idim, nsym))
        nmf = ints(idim * nsub, (idim, nsub))
        ipf = ints(idim * nsub, (idim, nsub))
        nlasym, isyma, isymb, nvb, ndg, ifsym = (ints(nsym) for _ in range(6))
        names = [raw[offset + 10 * i : offset + 10 * (i + 1)].decode().strip() for i in range(nsym)]
        offset += 10 * nsym
        ints(nvtb)  # initial-vector identifiers; records must still be consumed
        if offset != len(raw):
            raise ValueError("Unexpected angular header size")
        bcf = np.frombuffer(record(), dtype="<f8").copy()
        mmf = np.frombuffer(record(), dtype="<i4").copy()
        if len(bcf) < jdim or len(mmf) < jdim:
            raise ValueError("Truncated harmonic expansion tables")
        eb = np.frombuffer(record(), dtype="<f8").copy()
        if len(eb) not in (nvtb, nvdb):
            raise ValueError("Unexpected bound-energy storage length")
        eb = eb[:nvtb]
        parents = []
        parent = 0
        for ib in range(nsb):
            degeneracy = int(ndg[int(isymb[ib]) - 1])
            for _ in range(int(nvb[ib])):
                parents.extend([parent] * degeneracy)
                parent += 1
        if parent != nvtb or len(parents) != nvdb:
            raise ValueError("Bound degeneracies do not span the stored dipoles")
        symmetries = []
        for ia in range(nsa):
            sym = int(isyma[ia]) - 1
            nc = int(nlasym[ia])
            ng = int(ndg[sym])
            expansions = []
            for deg in range(ng):
                sub = int(ifsym[sym]) - 1 + deg
                component = []
                for c in range(nc):
                    start = int(ipf[c, sub])
                    length = int(nmf[c, sub])
                    component.append(
                        tuple(
                            (int(ilf[c, sym]), int(mmf[j]), float(bcf[j]))
                            for j in range(start, start + length)
                        )
                    )
                expansions.append(tuple(component))
            symmetries.append(
                ReferenceSymmetry(sym + 1, names[sym], ilf[:nc, sym].copy(), ng, tuple(expansions))
            )
        frames = []
        while True:
            try:
                energy = np.frombuffer(record(), dtype="<f8")
            except FortranEOFError:
                break
            if len(energy) != 2:
                raise ValueError("Malformed energy record")
            eigenvalues = []
            eigenvectors = []
            scattering = []
            for symmetry in symmetries:
                nc = len(symmetry.angular_momenta)
                data = np.frombuffer(record(), dtype="<f8")
                if len(data) != nc + nc * nc:
                    raise ValueError("Malformed K matrix record")
                k = data[:nc].copy()
                u = data[nc:].reshape(nc, nc, order="F").copy()
                eigenvalues.append(k)
                eigenvectors.append(u)
                scattering.append((u * ((1 + 1j * k) / (1 - 1j * k))[None, :]) @ u.T)
            dipole = np.frombuffer(record(), dtype="<f8")
            if len(dipole) != nvda * 3 * nvdb:
                raise ValueError("Malformed K-normalized dipole record")
            dipole = dipole.reshape(nvda, 3, nvdb, order="F")
            amplitudes = []
            row = 0
            for symmetry, k, u in zip(symmetries, eigenvalues, eigenvectors):
                nc = len(k)
                ng = symmetry.degeneracy
                transform = (u * (1 / (1 - 1j * k))[None, :]) @ u.T
                components = []
                for deg in range(ng):
                    d = dipole[row + deg : row + nc * ng : ng]
                    physical = np.einsum("ij,jmb->mib", transform, d, optimize=True)
                    # TIRESIA dipoles are r*realY1 in y,z,x order.
                    components.append(np.sqrt(4 * np.pi / 3) * physical[[2, 0, 1]])
                amplitudes.append(np.asarray(components))
                row += nc * ng
            if row != nvda:
                raise ValueError("Angular dimensions do not span stored continuum rows")
            frames.append(
                ReferenceFrame(
                    float(energy[0]),
                    tuple(eigenvalues),
                    tuple(eigenvectors),
                    tuple(amplitudes),
                    tuple(scattering),
                )
            )
    return ReferenceContinuum(
        charge, eb, tuple(symmetries), tuple(frames), group, np.asarray(parents, dtype=int)
    )


@dataclass(frozen=True)
class ReferenceResponseFrame:
    photon_energy: float
    response_kind: int
    amplitudes: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ReferenceResponseContinuum:
    """Physical Cartesian dipoles with state-specific outgoing electron energies."""

    static_metadata: ReferenceContinuum
    frames: tuple[ReferenceResponseFrame, ...]

    def cross_sections(self, *, occupations, alpha=7.29735308e-3, bohr2_mb=0.529177249**2*100):
        occupations=np.asarray(occupations,float)
        metadata=self.static_metadata
        if (occupations.shape!=metadata.bound_energies.shape
                or not np.all(np.isfinite(occupations)) or np.any(occupations<0)):
            raise ValueError('Require an explicit occupation for each parent orbital')
        rows=[]
        for frame in self.frames:
            strength=sum(np.sum(abs(x)**2,axis=(0,1,2)) for x in frame.amplitudes)
            strength=np.bincount(metadata.bound_component_parents,weights=strength,
                                 minlength=len(metadata.bound_energies))
            rows.append(4*np.pi**2*alpha/3*frame.photon_energy*strength*occupations*bohr2_mb)
        return np.asarray(rows)


def read_response_continuum(path, static_checkpoint):
    """Read ICALTYPE=2 dipoles, checking every angular header against a static run.

    The static checkpoint supplies the existing, independently checked angular
    decoder. All populated angular, dimension and bound-energy data must match.
    Unused Fortran allocation tails and zero padding carry no physical data. This reader therefore cannot silently apply a
    different molecular frame, bound-state ordering or symmetry expansion.
    """
    metadata=read_continuum(static_checkpoint)
    with FortranFile(path,'r',header_dtype=np.dtype('<u4')) as stream, FortranFile(
            static_checkpoint,'r',header_dtype=np.dtype('<u4')) as original:
        def raw(source):return source.read_record(np.uint8).tobytes()
        first=raw(stream);old=raw(original)
        if (len(first)!=32 or np.frombuffer(first[-4:],dtype='<i4')[0]!=2
                or first[20:28]!=old[20:28]):
            raise ValueError('Require the ICALTYPE=2 response format and matching symmetry')
        headers=[raw(stream) for _ in range(5)]
        # Reuse the static angular decoder on a header-only temporary stream.
        # Its parser already distinguishes stored dimensions from used entries.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="sparks-response-header-") as directory:
            temporary=Path(directory)/"header.bin"
            with FortranFile(temporary,"w",header_dtype=np.dtype("<u4")) as output:
                output.write_record(np.frombuffer(first[:-4]+bytes(4),dtype=np.uint8))
                for header in headers:
                    output.write_record(np.frombuffer(header,dtype=np.uint8))
            decoded=read_continuum(temporary)
        if (decoded.group!=metadata.group or decoded.residual_charge!=metadata.residual_charge
                or not np.array_equal(decoded.bound_energies,metadata.bound_energies)
                or not np.array_equal(decoded.bound_component_parents,metadata.bound_component_parents)
                or len(decoded.symmetries)!=len(metadata.symmetries)):
            raise ValueError('Response and static checkpoint physical headers differ')
        for left,right in zip(decoded.symmetries,metadata.symmetries):
            if (left.identifier!=right.identifier or left.name!=right.name
                    or left.degeneracy!=right.degeneracy
                    or not np.array_equal(left.angular_momenta,right.angular_momenta)
                    or left.angular_expansions!=right.angular_expansions):
                raise ValueError('Response and static angular expansions differ')
        nvda,nvdb=map(int,np.frombuffer(headers[0][:8],dtype='<i4'))
        frames=[]
        while True:
            try:head=raw(stream)
            except FortranEOFError:break
            if len(head)!=12:
                raise ValueError('Malformed response photon-energy/type record')
            omega=float(np.frombuffer(head[:8],dtype='<f8')[0])
            kind=int(np.frombuffer(head[8:],dtype='<i4')[0])
            data=np.frombuffer(raw(stream),dtype='<c16')
            if (not np.isfinite(omega) or omega<=0 or kind not in (1,2)
                    or len(data)!=nvda*3*nvdb or not np.all(np.isfinite(data))):
                raise ValueError('Invalid response dipole record')
            full=data.reshape(nvda,3,nvdb,order='F')
            amplitudes=[];row=0
            for symmetry in metadata.symmetries:
                nc=len(symmetry.angular_momenta);ng=symmetry.degeneracy
                components=[]
                for deg in range(ng):
                    d=full[row+deg:row+nc*ng:ng]
                    # DXTRAN already supplied the outgoing normalization.
                    components.append(np.sqrt(4*np.pi/3)*d[:,[2,0,1],:].transpose(1,0,2))
                amplitudes.append(np.asarray(components));row+=nc*ng
            if row!=nvda:
                raise ValueError('Response angular rows are inconsistent')
            frames.append(ReferenceResponseFrame(omega,kind,tuple(amplitudes)))
    return ReferenceResponseContinuum(metadata,tuple(frames))
