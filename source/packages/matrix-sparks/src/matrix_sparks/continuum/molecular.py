"""Molecular point-charge operators and electronic dipole-source projection.

Coordinates, charges and sources are explicit inputs. APOC owns electronic
state import; SPARKS does not introduce another Gaussian/Molden parser.
"""

import numpy as np
from scipy.special import eval_legendre


class NuclearPotential:
    """Exact point-charge potential projected onto a finite angular space.

    The addition theorem needs only L<=2*lmax in this finite matrix, including
    off-centre nuclei. Nuclear radii should be radial breakpoints. No softened
    Coulomb potential or numerical fit is introduced.
    """

    def __init__(self, channels, charges, positions_bohr):
        z = np.asarray(charges, dtype=float)
        positions = np.asarray(positions_bohr, dtype=float)
        if (
            z.ndim != 1
            or not len(z)
            or positions.shape != (len(z), 3)
            or not np.all(np.isfinite(z))
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("Require finite charges and matching (N,3) nuclear coordinates")
        self.charges, self.positions = z.copy(), positions.copy()
        self.channels = channels
        self.nuclear_radii = np.linalg.norm(positions, axis=1)
        self._terms = []
        weighted = channels.values * channels.weights[:, None]
        identity = np.eye(len(channels.labels))
        for charge, position, radius in zip(z, positions, self.nuclear_radii):
            if radius == 0:
                self._terms.append((charge, radius, np.array([identity])))
                continue
            cosine = np.clip(channels.directions @ (position / radius), -1, 1)
            matrices = []
            for l in range(2 * channels.lmax + 1):
                matrix = weighted.T @ (eval_legendre(l, cosine)[:, None] * channels.values)
                matrices.append(0.5 * (matrix + matrix.T))
            self._terms.append((charge, radius, np.asarray(matrices)))

    def __call__(self, radii):
        r = np.asarray(radii, dtype=float)
        if r.ndim != 1 or np.any(r <= 0) or not np.all(np.isfinite(r)):
            raise ValueError("Radial nodes must be finite and strictly positive")
        n = len(self.channels.labels)
        result = np.zeros((len(r), n, n))
        for charge, radius, matrices in self._terms:
            hi, lo = np.maximum(r, radius), np.minimum(r, radius)
            powers = (lo / hi)[:, None] ** np.arange(len(matrices))[None, :]
            result -= charge * np.einsum(
                "rl,lab->rab", powers / hi[:, None], matrices, optimize=True
            )
        return result


class AODysonOrbital:
    """Reuse APOC/PySCF AO conventions for an explicitly supplied real Dyson vector.

    The norm and phase are preserved; an MO is not silently treated as a
    correlated Dyson orbital. An AO coefficient column matrix evaluates many
    orbitals with one AO batch; columns retain their individual norms and signs.
    Batches use the existing compiled libcgto AO
    evaluator and NumPy BLAS. No electronic calculation is performed.
    """

    def __init__(self, state, coefficients_ao, *, batch_size=4096):
        from matrix_apoc.population import pyscf_molecule_from_electronic_state

        coefficients = np.asarray(coefficients_ao)
        if (
            np.iscomplexobj(coefficients)
            or coefficients.ndim not in (1, 2)
            or (coefficients.ndim == 2 and coefficients.shape[1] == 0)
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("Dyson coefficients must be a finite real AO vector or column matrix")
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        self.molecule = pyscf_molecule_from_electronic_state(state)
        if len(coefficients) != self.molecule.nao_nr():
            raise ValueError("Dyson AO dimensions do not match the imported state")
        self.coefficients = coefficients.astype(float, copy=True)
        self.batch_size = int(batch_size)
        self.norm_squared = (float(self.coefficients @ state.overlap_ao @ self.coefficients)
            if coefficients.ndim == 1 else
            np.einsum("ip,ij,jp->p",self.coefficients,state.overlap_ao,self.coefficients))

    @classmethod
    def from_apoc(cls, dyson, *, batch_size=4096):
        """Consume APOC's qualified Dyson object without backend-specific parsing."""
        from matrix_apoc.dyson import DysonOrbital

        if not isinstance(dyson, DysonOrbital):
            raise TypeError("Expected an APOC DysonOrbital")
        return cls(dyson.state, dyson.coefficients_ao, batch_size=batch_size)

    def __call__(self, points_bohr):
        points = np.asarray(points_bohr, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("Orbital coordinates must be finite (N,3) points")
        result = np.empty((len(points),) + self.coefficients.shape[1:])
        evaluator = "GTOval_cart" if self.molecule.cart else "GTOval_sph"
        for start in range(0, len(points), self.batch_size):
            end = start + self.batch_size
            result[start:end] = (
                self.molecule.eval_gto(evaluator, points[start:end]) @ self.coefficients
            )
        return result


class ProjectedDipoleSource:
    """Project a real spatial Dyson orbital onto reduced radial dipole sources.

    f_c,mu(r) = r^2 integral dOmega Y_c(Omega) n_mu psi_D(r Omega).
    The orbital callback is reusable for other electronic backends. Work is
    streamed over radial shells and AO evaluation is independently batched.
    """

    def __init__(self, channels, orbital, *, origin_bohr=(0, 0, 0), radial_batch=16):
        origin = np.asarray(origin_bohr, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("Origin must be a finite Cartesian vector")
        if int(radial_batch) != radial_batch or radial_batch < 1:
            raise ValueError("Radial batch must be a positive integer")
        self.channels, self.orbital = channels, orbital
        self.origin, self.radial_batch = origin.copy(), int(radial_batch)
        self._angular = np.einsum(
            "pc,pm,p->pcm", channels.values, channels.directions, channels.weights
        )

    def __call__(self, radii):
        r = np.asarray(radii, dtype=float)
        if r.ndim != 1 or np.any(r < 0) or not np.all(np.isfinite(r)):
            raise ValueError("Radii must be finite and nonnegative")
        result = None
        # Empty radii still preserve the explicit number of orbital columns.
        starts = range(0, len(r), self.radial_batch) if len(r) else [0]
        for start in starts:
            rb = r[start : start + self.radial_batch]
            xyz = self.origin + rb[:, None, None] * self.channels.directions[None, :, :]
            values = np.asarray(self.orbital(xyz.reshape(-1, 3)))
            if (np.iscomplexobj(values) or values.ndim not in (1,2)
                    or values.shape[0] != len(rb)*len(self.channels.weights)
                    or (values.ndim == 2 and values.shape[1] == 0)
                    or not np.all(np.isfinite(values))):
                raise ValueError("Orbital evaluator must return finite real values or columns at all points")
            n = 1 if values.ndim == 1 else values.shape[1]
            if result is None:
                result = np.empty((len(r),len(self.channels.labels),3*n))
            if result.shape[2] != 3*n:
                raise ValueError("Orbital column count must be constant")
            block = np.einsum("rpo,pcm->rcom",
                values.reshape(len(rb),len(self.channels.weights),n),self._angular,optimize=True)
            result[start:start+len(rb)] = rb[:,None,None]**2 * block.reshape(len(rb),len(self.channels.labels),3*n)
        return result


class AOHartreePotential:
    """Coulomb repulsion from the actual total APOC density, in Hartree.

    Reuses PySCF/libcint's exact int1e_grids integrals, contracted in bounded
    shell-pair blocks by the SPARKS C++ driver. ``contraction="matrix"`` retains
    the explicit AO-matrix path; ``auto`` uses it when no compiler is available.
    ``direct`` requires the native driver. The selected implementation is
    reported in ``contraction_backend``. No auxiliary
    charges, density fitting, SCF, exchange functional or polarization model is
    introduced. Spin densities are summed explicitly. The positive Hartree
    contribution must be combined with nuclear attraction and any separately
    specified physical exchange/correlation model.
    """

    def __init__(self, state, *, workspace_bytes=64 * 1024**2, contraction="auto"):
        from matrix_apoc.population import pyscf_molecule_from_electronic_state

        if contraction not in {"auto", "direct", "matrix"}:
            raise ValueError("Coulomb contraction must be auto, direct or matrix")
        self.molecule = pyscf_molecule_from_electronic_state(state)
        density = sum(state.density_ao(channel) for channel in range(len(state.mo_coefficients)))
        if np.iscomplexobj(density) or not np.all(np.isfinite(density)):
            raise ValueError("Hartree provider requires the finite real total AO density")
        self.density = np.asarray(density, dtype=float).copy()
        if not np.allclose(self.density, self.density.T, rtol=1e-12, atol=1e-13):
            raise ValueError("Total AO density must be symmetric")
        if int(workspace_bytes) != workspace_bytes or workspace_bytes < 8 * len(density) ** 2:
            raise ValueError("Workspace must hold at least one AO integral matrix")
        self.batch_size = int(workspace_bytes) // (8 * len(density) ** 2)
        self.electron_number = float(np.einsum("ij,ji", density, state.overlap_ao))
        self._direct_contraction = None
        if contraction != "matrix":
            from .coulomb_native import DirectCoulombContraction
            try:
                self._direct_contraction = DirectCoulombContraction(self.molecule, workspace_bytes=workspace_bytes)
            except FileNotFoundError:
                if contraction == "direct":
                    raise
        self.contraction_backend = "direct" if self._direct_contraction is not None else "matrix"

    def __call__(self, points_bohr):
        points = np.asarray(points_bohr, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("Hartree coordinates must be finite (N,3) points")
        if getattr(self, "_direct_contraction", None) is not None:
            return self._direct_contraction(points, self.density)
        result = np.empty(len(points))
        # libcint stores the grid index contiguously (Fortran order). Contract
        # that layout directly instead of permuting the two AO axes first.
        density_vector = self.density.T.ravel(order="F")
        for start in range(0, len(points), self.batch_size):
            end = start + self.batch_size
            integrals = self.molecule.intor("int1e_grids", grids=points[start:end], hermi=1)
            result[start:end] = integrals.reshape(len(integrals), -1, order="F") @ density_vector
        return result


class ProjectedLocalPotential:
    """Project a scalar molecular potential onto real angular channels.

    Use NuclearPotential separately for nuclear singularities. This projector
    handles a smooth electronic or other local contribution with an explicitly
    supplied callable. Its angular quadrature must be converged for that field.
    """

    def __init__(self, channels, potential, *, origin_bohr=(0, 0, 0), radial_batch=16):
        origin = np.asarray(origin_bohr, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("Origin must be a finite Cartesian vector")
        if int(radial_batch) != radial_batch or radial_batch < 1:
            raise ValueError("Radial batch must be a positive integer")
        self.channels, self.potential = channels, potential
        self.origin, self.radial_batch = origin.copy(), int(radial_batch)

    def __call__(self, radii, *, return_samples=False):
        r = np.asarray(radii, dtype=float)
        if r.ndim != 1 or np.any(r < 0) or not np.all(np.isfinite(r)):
            raise ValueError("Radii must be finite and nonnegative")
        n = len(self.channels.labels)
        output = np.empty((len(r), n, n))
        samples = np.empty((len(r),len(self.channels.weights))) if return_samples else None
        weighted_values = self.channels.weights[:, None] * self.channels.values
        for start in range(0, len(r), self.radial_batch):
            rb = r[start : start + self.radial_batch]
            xyz = self.origin + rb[:, None, None] * self.channels.directions[None, :, :]
            v = np.asarray(self.potential(xyz.reshape(-1, 3)))
            if (
                np.iscomplexobj(v)
                or v.shape != (len(rb) * len(self.channels.weights),)
                or not np.all(np.isfinite(v))
            ):
                raise ValueError("Local potential must return finite real values at all points")
            if samples is not None:
                samples[start:start+len(rb)] = v.reshape(len(rb),-1)
            block = self.channels.values.T @ (
                v.reshape(len(rb), -1)[..., None] * weighted_values
            )
            output[start : start + len(rb)] = (block + block.swapaxes(1, 2)) * 0.5
        return (output,samples) if return_samples else output
