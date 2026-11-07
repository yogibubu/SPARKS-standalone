"""Retarded/advanced Coulomb boundary self-energy for Sternheimer solves."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

from .coulomb import coulomb_boundary, _outgoing_series
from .resolvent import _real_sparse_matrix


def coulomb_log_derivative(energy, radius, angular_momenta, charge, *, rtol=2e-10, atol=2e-11):
    """Outgoing h'/h above the real axis, decaying below threshold.

    The lower half plane is the advanced conjugate. Positive real energies
    reuse the existing energy-normalized Coulomb boundary. At negative or
    complex energy the same outgoing asymptotic series is propagated as a
    logarithmic derivative, avoiding exponentially overflowing radial waves.
    Exact threshold is excluded; no finite imaginary part is inserted.
    """
    z = complex(energy)
    raw = np.asarray(angular_momenta)
    if (
        raw.ndim != 1
        or not len(raw)
        or not np.all(np.isfinite(raw))
        or np.any(raw < 0)
        or np.any(raw != raw.astype(int))
        or not np.isfinite(z)
        or z == 0
        or not np.all(np.isfinite([radius, charge, rtol, atol]))
        or radius <= 0
        or min(rtol, atol) <= 0
    ):
        raise ValueError(
            "Require nonzero finite energy, integer angular momenta and positive radius/tolerances"
        )
    if z.imag < 0:
        return coulomb_log_derivative(
            z.conjugate(), radius, raw, charge, rtol=rtol, atol=atol
        ).conjugate()
    if z.imag == 0 and z.real > 0:
        k = np.sqrt(2 * z.real)
        eta, target = -charge / k, k * radius
        turning = eta + np.sqrt(eta**2 + raw.max() * (raw.max() + 1))
        if charge != 0 and target > turning + 2:
            # Only h'/h is needed. Propagate its smooth l=0 envelope instead
            # of phase-oscillating normalized waves, then recur on h_l/h_l-1.
            start = max(target, 64.0, 4 * (eta**2 + 1))
            series, derivative = _outgoing_series(eta, np.array([0]), start, atol)
            q = 1j * (1 - eta / start) + derivative[0] / series[0]
            if start > target:
                result = solve_ivp(
                    lambda rho, value: -(1 - 2 * eta / rho) - value**2,
                    (start, target), np.array([q]), method="DOP853",
                    rtol=rtol, atol=atol, t_eval=[target],
                )
                if not result.success:
                    raise ArithmeticError(f"Coulomb logarithmic propagation failed: {result.message}")
                q = result.y[0, -1]
            ratios = np.empty(int(raw.max()) + 1, complex)
            ratios[0] = (1 / target + eta - q) / np.sqrt(1 + eta**2)
            for order in range(1, len(ratios)):
                ratios[order] = (
                    (2 * order + 1) * (eta + order * (order + 1) / target)
                    - (order + 1) * np.sqrt(order**2 + eta**2) / ratios[order - 1]
                ) / (order * np.sqrt((order + 1)**2 + eta**2))
            orders = raw.astype(int)
            q = k * ((orders + 1) / target + eta / (orders + 1)
                     - np.sqrt(1 + (eta / (orders + 1))**2) * ratios[orders])
            if not np.all(np.isfinite(q)) or np.any(q.imag <= 0):
                raise ArithmeticError("Coulomb logarithmic derivative lost its outgoing flux")
            return q
        waves = coulomb_boundary(z.real, radius, raw, charge, rtol=rtol, atol=atol)
        return (waves.irregular_derivative + 1j * waves.regular_derivative) / (
            waves.irregular + 1j * waves.regular
        )
    # IEEE negative zero must not select the growing square-root branch.
    if z.imag == 0:
        z = complex(z.real, 0.0)
    l, inverse = np.unique(raw.astype(int), return_inverse=True)
    if z.imag == 0 and z.real < 0:
        # W_{Z/kappa,l+1/2}(2*kappa*r): exponential and power factors
        # cancel analytically in h'/h. No growing wave is constructed.
        from scipy.special import hyperu

        kappa = np.sqrt(-2 * z.real)
        a = l + 1 - charge / kappa
        argument = 2 * kappa * radius
        u = hyperu(a, 2 * l + 2, argument)
        up = hyperu(a + 1, 2 * l + 3, argument)
        if np.all(np.isfinite(u)) and np.all(u != 0) and np.all(np.isfinite(up)):
            q = -kappa + (l + 1) / radius - 2 * kappa * a * up / u
            if np.all(np.isfinite(q)):
                return q[inverse].astype(complex)
    k = np.sqrt(2 * z)
    eta = -charge / k
    start = max(radius, 64 / abs(k), 4 * (abs(eta) ** 2 + l.max() * (l.max() + 1) + 1) / abs(k))
    rho = k * start
    series, derivative = _outgoing_series(eta, l, rho, atol)
    q = k * (1j * (1 - eta / rho) + derivative / series)
    if start > radius:
        # Propagate the linear radial equations across possible radial nodes.
        # Per-channel rescaling preserves h'/h and bounds exponential growth.
        def rhs(r, state):
            u, up = state.reshape(2, len(l))
            return np.r_[up, (l * (l + 1) / r**2 - 2 * charge / r - 2 * z) * u]

        state = np.r_[np.ones(len(l), complex), q]
        segments = max(1, int(np.ceil((start - radius) * abs(k) / 16)))
        points = np.linspace(start, radius, segments + 1)
        for left, right in zip(points[:-1], points[1:]):
            result = solve_ivp(
                rhs, (left, right), state, method="DOP853", rtol=rtol, atol=atol, t_eval=[right]
            )
            if not result.success:
                raise ArithmeticError(f"Retarded Coulomb propagation failed: {result.message}")
            u, up = result.y[:, -1].reshape(2, len(l))
            scale = np.maximum(abs(u), abs(up) / abs(k))
            if np.any(scale == 0) or not np.all(np.isfinite(scale)):
                raise ArithmeticError("Retarded Coulomb wave lost its scale")
            state = np.r_[u / scale, up / scale]
        u, up = state.reshape(2, len(l))
        if np.any(u == 0):
            raise ArithmeticError("Coulomb logarithmic derivative has a boundary pole")
        q = up / u
    return q[inverse]


class CoulombResponseBoundary:
    """Supply Sigma(z)=-B*q(z)*B.T/2 in the Galerkin response convention.

    A supplied tail potential is the complete local potential excluding the
    centrifugal term, with shape (points,channels,channels). Its declared
    outer radius carries a pure Coulomb boundary. Convergence of that exterior
    approximation is the caller's responsibility; no dipole tail is removed
    or silently fitted. All channel thresholds are zero.
    """

    def __init__(
        self,
        boundary,
        radius,
        angular_momenta,
        charge,
        *,
        outer_radius=None,
        tail_potential=None,
        rtol=2e-10,
        atol=2e-11,
    ):
        self.boundary = _real_sparse_matrix(boundary, "boundary")
        self.radius = float(radius)
        self.outer_radius = self.radius if outer_radius is None else float(outer_radius)
        self.angular_momenta = np.asarray(angular_momenta)
        self.charge = float(charge)
        self.rtol, self.atol = rtol, atol
        self.tail_potential = tail_potential
        if (
            self.boundary.shape[1] != len(self.angular_momenta)
            or self.angular_momenta.ndim != 1
            or not np.isfinite(self.outer_radius)
            or self.outer_radius < self.radius
            or (self.outer_radius > self.radius and tail_potential is None)
        ):
            raise ValueError("Boundary channels and explicit tail/radii must agree")
        # Validate remaining scalar and angular parameters without propagation.
        if (
            not np.all(np.isfinite(self.angular_momenta))
            or np.any(self.angular_momenta < 0)
            or np.any(self.angular_momenta != self.angular_momenta.astype(int))
            or not np.all(np.isfinite([self.radius, self.charge, rtol, atol]))
            or self.radius <= 0
            or min(rtol, atol) <= 0
        ):
            raise ValueError("Invalid Coulomb response boundary parameters")

    def log_derivative(self, energy):
        """Return the channel-sized outgoing logarithmic derivative."""
        z = complex(energy)
        q = np.diag(
            coulomb_log_derivative(
                z,
                self.outer_radius,
                self.angular_momenta,
                self.charge,
                rtol=self.rtol,
                atol=self.atol,
            )
        )
        nc = len(q)
        if self.outer_radius > self.radius:

            def rhs(r, flat):
                potential = np.asarray(self.tail_potential(np.asarray([r])))
                if (
                    potential.shape != (1, nc, nc)
                    or np.iscomplexobj(potential)
                    or not np.all(np.isfinite(potential))
                    or not np.allclose(potential[0], potential[0].T, atol=1e-13, rtol=1e-12)
                ):
                    raise ValueError("Tail must be a finite real symmetric channel potential")
                value = flat.reshape(nc, nc)
                return (
                    2 * potential[0]
                    - 2 * z * np.eye(nc)
                    + np.diag(self.angular_momenta * (self.angular_momenta + 1) / r**2)
                    - value @ value
                ).ravel()

            result = solve_ivp(
                rhs,
                (self.outer_radius, self.radius),
                q.ravel(),
                method="DOP853",
                rtol=self.rtol,
                atol=self.atol,
                t_eval=[self.radius],
            )
            if not result.success:
                raise ArithmeticError(f"Response tail propagation failed: {result.message}")
            q = result.y[:, -1].reshape(nc, nc)
        return q

    def __call__(self, energy):
        q = self.log_derivative(energy)
        return -0.5 * self.boundary @ csc_matrix(q) @ self.boundary.T


class CoulombSampleBoundary:
    """Explicit two-surface outgoing constraint and closed-box endpoints.

    samples[j,c,:] evaluates a trial function in channel c at radii[j].
    Positive energies impose u(r2)=h(r2)/h(r1)*u(r1), with the declared shifted
    Coulomb radii. Nonpositive real part imposes zero endpoint coefficients.
    This finite-space boundary model is useful for matching legacy calculations;
    it is separate from the surface-logarithmic-derivative self-energy model.
    """

    def __init__(self, boundary, samples, radii, angular_momenta, charge, *, rtol=2e-10, atol=2e-11):
        self.boundary = _real_sparse_matrix(boundary, 'boundary')
        self.samples = np.array(samples, dtype=float, copy=True)
        self.radii = np.asarray(radii, dtype=float)
        self.orders = np.asarray(angular_momenta)
        self.charge = float(charge)
        n, nc = self.boundary.shape
        if (self.samples.shape != (2, nc, n) or self.radii.shape != (2,)
                or self.orders.shape != (nc,) or np.any(self.orders < 0)
                or np.any(self.orders != self.orders.astype(int))
                or np.any(self.radii <= 0) or np.diff(self.radii)[0] <= 0
                or not all(np.all(np.isfinite(x)) for x in
                           (self.samples, self.radii, self.orders, [charge, rtol, atol]))
                or min(rtol, atol) <= 0):
            raise ValueError('Finite ordered sampling radii, integer channels and matching trial values required')
        self.rtol, self.atol = rtol, atol

    def constraint(self, energy):
        z = complex(energy)
        if not np.isfinite(z) or z == 0:
            raise ValueError('Finite nonzero energy required')
        if z.real <= 0:
            return self.boundary.T
        r1, r2 = self.radii
        q = coulomb_log_derivative(z, r2, self.orders, self.charge,
                                   rtol=self.rtol, atol=self.atol)
        nc = len(q)

        def rhs(r, state):
            derivative = state[:nc]
            return np.r_[self.orders*(self.orders+1)/r**2 - 2*self.charge/r - 2*z - derivative**2,
                         derivative]

        result = solve_ivp(rhs, (r2, r1), np.r_[q, np.zeros(nc, complex)], method='DOP853',
                           rtol=self.rtol, atol=self.atol, t_eval=[r1])
        if not result.success:
            raise ArithmeticError('Outgoing sample-ratio propagation failed')
        ratio = np.exp(-result.y[nc:, -1])
        return self.samples[1] - ratio[:, None]*self.samples[0]
