"""Real spherical harmonics and exact finite-space angular contractions."""

import numpy as np
from scipy.special import roots_legendre, sph_harm_y


class RealSphericalChannels:
    """Explicit (l,m) channels: m>0 cosine, m<0 sine, p_x,p_y,p_z positive.

    Projection uses a Gauss-Legendre/Fourier rule that exactly integrates the
    finite harmonic products used by point-charge multipoles and dipoles.
    Spatial quadrature for a general electronic orbital still needs convergence.
    """

    def __init__(
        self,
        labels,
        *,
        polar_order=None,
        azimuthal_order=None,
        quadrature="product",
        lebedev_order=None,
    ):
        raw = np.asarray(labels)
        if (
            raw.ndim != 2
            or raw.shape[1] != 2
            or not len(raw)
            or not np.all(np.isfinite(raw))
            or np.any(raw != raw.astype(int))
        ):
            raise ValueError("Channels must be nonempty integer (l,m) pairs")
        self.labels = raw.astype(int)
        if np.any(self.labels[:, 0] < np.abs(self.labels[:, 1])) or len(
            set(map(tuple, self.labels))
        ) != len(raw):
            raise ValueError("Require unique channels with l>=|m|")
        self.angular_momenta = self.labels[:, 0]
        self.lmax = int(self.angular_momenta.max())
        self.quadrature = quadrature
        if quadrature == "lebedev":
            if polar_order is not None or azimuthal_order is not None:
                raise ValueError("Product-grid orders cannot be combined with Lebedev")
            from pyscf.dft.LebedevGrid import LEBEDEV_ORDER, MakeAngularGrid

            minimum = max(3, 4 * self.lmax)
            available = sorted(order for order in LEBEDEV_ORDER if order >= minimum)
            if not available:
                raise ValueError("No available Lebedev rule resolves this angular space")
            order = available[0] if lebedev_order is None else lebedev_order
            if order not in LEBEDEV_ORDER or order < minimum:
                raise ValueError("Lebedev algebraic order must resolve products through 4*lmax")
            grid = MakeAngularGrid(LEBEDEV_ORDER[order])
            self.directions = grid[:, :3].copy()
            self.weights = 4 * np.pi * grid[:, 3].copy()
            self.lebedev_order = order
            self.values = self.evaluate(self.directions)
            return
        if quadrature != "product" or lebedev_order is not None:
            raise ValueError("Choose product quadrature or an explicit Lebedev rule")
        nt = 2 * self.lmax + 2 if polar_order is None else int(polar_order)
        np_ = 4 * self.lmax + 3 if azimuthal_order is None else int(azimuthal_order)
        if nt < 2 * self.lmax + 2 or np_ < 4 * self.lmax + 3:
            raise ValueError("Angular rule underresolves finite harmonic products")
        z, w = roots_legendre(nt)
        phi = np.arange(np_) * (2 * np.pi / np_)
        zz, pp = np.meshgrid(z, phi, indexing="ij")
        self.directions = np.column_stack(
            (
                (np.sqrt(1 - zz**2) * np.cos(pp)).ravel(),
                (np.sqrt(1 - zz**2) * np.sin(pp)).ravel(),
                zz.ravel(),
            )
        )
        self.weights = np.repeat(w, np_) * (2 * np.pi / np_)
        self.values = self.evaluate(self.directions)

    def evaluate(self, directions):
        """Evaluate normalized real harmonics on unit direction vectors."""
        xyz = np.asarray(directions, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
            raise ValueError("Directions must be finite (N,3) vectors")
        norms = np.linalg.norm(xyz, axis=1)
        if np.any(norms == 0):
            raise ValueError("Angular direction cannot vanish")
        theta = np.arccos(np.clip(xyz[:, 2] / norms, -1, 1))
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        columns = [None] * len(self.labels)
        for (l, order), members in self._real_channel_groups().items():
            y = sph_harm_y(l, order, theta, phi)
            for index, m in members:
                columns[index] = (
                    y.real if m == 0 else np.sqrt(2) * (-1.0) ** order
                    * (y.real if m > 0 else y.imag)
                )
        return np.column_stack(columns)

    def _real_channel_groups(self):
        """Group sine/cosine partners without changing the requested ordering."""
        groups = {}
        for index, (l, m) in enumerate(self.labels):
            groups.setdefault((int(l), int(abs(m))), []).append((index, int(m)))
        return groups

    def dipole_couplings(self, initial):
        """Return <final channel|n_xyz|initial channel>, shape (final,3,initial)."""
        initial_values = initial.evaluate(self.directions)
        return np.einsum(
            "pa,pm,pb,p->amb",
            self.values,
            self.directions,
            initial_values,
            self.weights,
            optimize=True,
        )

    def transform_dipole_amplitudes(self, amplitudes, orthogonal, *, dyson_character):
        """Transform a symmetry-related geometry in a fixed origin/frame.

        ``orthogonal`` acts on Cartesian column vectors and may be improper.
        The isolated reference Dyson character fixes the relative channel
        phase. The final two axes are (Cartesian dipole, spherical channel).
        This transforms amplitudes, not squared strengths. An incomplete
        angular space that is not closed under the operation is rejected.
        """
        matrix = np.asarray(orthogonal, float)
        values = np.asarray(amplitudes, complex)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12, rtol=0)
                or dyson_character not in (-1, 1)
                or values.shape[-2:] != (3, len(self.labels))
                or not np.isfinite(values).all()):
            raise ValueError("Finite dipoles, orthogonal operation and isolated real Dyson character required")
        representation = self.values.T @ (
            self.weights[:, None] * self.evaluate(self.directions @ matrix))
        if not np.allclose(representation.T @ representation, np.eye(len(self.labels)),
                           atol=2e-11, rtol=0):
            raise ValueError("Angular channel space is not closed under the symmetry operation")
        return dyson_character * np.einsum(
            'ab,...bd,cd->...ac', matrix, values, representation, optimize=True)

    def evaluate_with_gradient(self, vectors):
        """Return Y_lm and its Cartesian gradient at nonzero Cartesian vectors.

        Pole limits are analytic. No angular finite differences are used.
        """
        xyz = np.asarray(vectors, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
            raise ValueError("Require finite Cartesian vectors")
        r = np.linalg.norm(xyz, axis=1)
        if np.any(r == 0):
            raise ValueError("Angular gradients are undefined at the origin")
        n = xyz / r[:, None]
        theta = np.arccos(np.clip(n[:, 2], -1, 1))
        phi = np.arctan2(n[:, 1], n[:, 0])
        st, ct = np.sin(theta), np.cos(theta)
        ep = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(len(r))))
        et = np.column_stack((ct * np.cos(phi), ct * np.sin(phi), -st))
        poles = abs(st) < 32 * np.finfo(float).eps
        values, gradients = [None] * len(self.labels), [None] * len(self.labels)
        for (l, order), members in self._real_channel_groups().items():
            y, derivative = sph_harm_y(l, order, theta, phi, diff_n=1)
            dy = derivative[:, 0]
            ratio = np.zeros_like(y)
            np.divide(1j * order * y, st, out=ratio, where=~poles)
            if order == 1:
                ratio[poles] = 1j * dy[poles] / ct[poles]
            grad = (dy[:, None] * et + ratio[:, None] * ep) / r[:, None]
            for index, m in members:
                if m == 0:
                    values[index], gradients[index] = y.real, grad.real
                else:
                    scale = np.sqrt(2) * (-1.0) ** order
                    values[index] = scale * (y.real if m > 0 else y.imag)
                    gradients[index] = scale * (grad.real if m > 0 else grad.imag)
        return np.column_stack(values), np.stack(gradients, axis=1)
