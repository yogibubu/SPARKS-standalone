"""Outgoing waves for a Coulomb plus anisotropic inverse-square tail.

The real angular matrix L(L+1)+2D is diagonalised once. Scalar outgoing
solutions are integrated with the existing Coulomb asymptotic recurrence;
no finite outer cutoff of D/r^2 is used. Original spherical-channel phases
are restored before returning the common BoundaryWaves contract.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import loggamma

from .coulomb import coulomb_boundary
from .spectral import BoundaryWaves


class DipoleCoulombBoundary:
    """Reusable boundary provider for V(r)=-charge/r I + D/r^2.

    ``dipole_coupling`` is the real symmetric potential coefficient D in the
    caller's angular-channel order, in atomic units. Molecular dipoles must use
    the same origin and sign as that potential. Higher multipoles require the
    existing ``propagate_boundary`` between this reference and the inner box.
    Eigenvalues below -1/4 are supported without inventing a real angular
    momentum or dropping the associated channels. No origin boundary condition
    for a singular point-dipole Hamiltonian is imposed by this exterior solver.
    """

    def __init__(self, angular_momenta, dipole_coupling, charge):
        raw_l, raw_d = np.asarray(angular_momenta), np.asarray(dipole_coupling)
        if np.iscomplexobj(raw_l) or np.iscomplexobj(raw_d):
            raise ValueError("Angular labels and dipole coupling must be real")
        l, d = np.asarray(raw_l, float), np.asarray(raw_d, float)
        if (
            l.ndim != 1
            or not len(l)
            or not np.all(np.isfinite(l))
            or np.any(l < 0)
            or np.any(l != l.astype(int))
        ):
            raise ValueError("Require nonnegative integer angular momenta")
        if (
            d.shape != (len(l), len(l))
            or not np.all(np.isfinite(d))
            or not np.allclose(d, d.T, rtol=1e-12, atol=1e-13)
        ):
            raise ValueError("Dipole coefficient must be finite, symmetric and channel-sized")
        if not np.isfinite(charge):
            raise ValueError("Coulomb charge must be finite")
        self.angular_momenta = l.astype(int)
        self.dipole_coupling = d.copy()
        self.charge = float(charge)
        self.eigenvalues, self.rotation = np.linalg.eigh(np.diag(l * (l + 1)) + 2 * d)

    def __call__(self, energy, radius, *, rtol=2e-11, atol=2e-12):
        if (
            not np.all(np.isfinite([energy, radius, rtol, atol]))
            or min(energy, radius, rtol, atol) <= 0
        ):
            raise ValueError("Require positive open energy, radius and tolerances")
        if not np.any(self.dipole_coupling):
            return coulomb_boundary(
                energy, radius, self.angular_momenta, self.charge, rtol=rtol, atol=atol
            )
        k = np.sqrt(2 * energy)
        eta, target = -self.charge / k, k * radius
        lam = self.eigenvalues
        start = max(target, 64.0, 4 * (eta**2 + np.max(np.abs(lam)) + 1))
        term = np.ones(len(lam), dtype=complex)
        series, derivative = term.copy(), np.zeros(len(lam), dtype=complex)
        for n in range(64):
            term *= ((n + 1j * eta) * (n + 1 + 1j * eta) - lam) / (2j * (n + 1) * start)
            series += term
            derivative -= (n + 1) * term / start
            if np.max(np.abs(term)) < min(1e-15, atol * 1e-3):
                break
        else:
            raise ArithmeticError("Dipole-Coulomb outgoing expansion did not converge")
        # Zero-phase outgoing functions. The channel-dependent Coulomb phase
        # is supplied after rotating back, never in the changing eigenbasis.
        phase = np.exp(1j * (start - eta * np.log(2 * start)))
        h = phase * series
        hp = phase * (1j * (1 - eta / start) * series + derivative)
        if start > target:

            def rhs(rho, state):
                u, up = state.reshape(2, len(lam))
                return np.r_[up, -(1 - 2 * eta / rho - lam / rho**2) * u]

            solution = solve_ivp(
                rhs,
                (start, target),
                np.r_[h, hp],
                method="DOP853",
                rtol=rtol,
                atol=atol,
                t_eval=[target],
            )
            if not solution.success:
                raise ArithmeticError(f"Dipole-Coulomb propagation failed: {solution.message}")
            h, hp = solution.y[:, -1].reshape(2, len(lam))
        norm = np.sqrt(2 / (np.pi * k))
        h, hp = norm * h, norm * k * hp
        flux = np.imag(h.conj() * hp)
        if not np.allclose(flux, 2 / np.pi, rtol=max(2e-8, 100 * rtol), atol=100 * atol):
            raise ArithmeticError("Dipole-Coulomb normalization lost during propagation")
        l = self.angular_momenta
        physical_phase = np.exp(1j * (-np.pi * l / 2 + np.imag(loggamma(l + 1 + 1j * eta))))
        outgoing = ((self.rotation * h) @ self.rotation.T) * physical_phase[None, :]
        slope = ((self.rotation * hp) @ self.rotation.T) * physical_phase[None, :]
        return BoundaryWaves(outgoing.imag, outgoing.real, slope.imag, slope.real)
