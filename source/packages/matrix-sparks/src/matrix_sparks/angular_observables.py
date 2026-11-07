"""Angle-integrated, anisotropy and molecular-frame photoelectron observables."""

import numpy as np
from scipy.special import loggamma


def molecular_frame_photoelectron_distribution(amplitudes, channels, *,
        electron_energy_hartree, residual_charge, polarization_density,
        directions=None):
    """Polarization-resolved molecular-frame strength per steradian.

    Dipoles have shape (...,3,channel), with exactly the Coulomb and real-
    harmonic conventions of photoelectron_angular_parameters. A unit-trace,
    positive Hermitian polarization_density[a,b] = <epsilon_a epsilon_b*>
    represents linear, circular, elliptical or incoherently mixed light in
    the molecular Cartesian frame. Photon propagation/transversality is the
    caller's responsibility. This is a fixed-orientation observable, not an
    orientation-averaged circular dichroism calculation.

    Strength is sum_ab rho_ab d_a(n) d_b(n)*. Its analytic angular integral
    is returned even for caller-supplied directions. Multiplication by
    4*pi**2*alpha*omega (and any explicit channel multiplicity) converts the
    energy-normalized strength to differential cross section in bohr**2/sr.
    There is no isotropic 1/3 factor for a fixed molecular orientation.
    """
    d = np.asarray(amplitudes, complex)
    energy = float(electron_energy_hartree)
    charge = float(residual_charge)
    rho = np.asarray(polarization_density, complex)
    if (d.ndim < 2 or d.shape[-2:] != (3, len(channels.labels))
            or not np.all(np.isfinite(d)) or not np.isfinite(energy)
            or energy <= 0 or not np.isfinite(charge)):
        raise ValueError('Require finite Cartesian/channel dipoles and positive energy')
    if (rho.shape != (3, 3) or not np.all(np.isfinite(rho))
            or not np.allclose(rho, rho.conj().T, rtol=0, atol=1e-12)
            or not np.isclose(np.trace(rho), 1, rtol=0, atol=1e-12)):
        raise ValueError('Polarization density must be Hermitian with unit trace')
    if np.linalg.eigvalsh(rho).min() < -1e-12:
        raise ValueError('Polarization density must be positive semidefinite')
    xyz = channels.directions if directions is None else np.asarray(directions, float)
    values = channels.values if directions is None else channels.evaluate(xyz)
    l = channels.angular_momenta
    eta = -charge / np.sqrt(2 * energy)
    phase = np.exp(1j * (loggamma(l + 1 + 1j * eta).imag - np.pi * l / 2))
    angular = np.einsum('...ac,pc,c->...pa', d, values, phase, optimize=True)
    density = np.einsum('...pa,ab,...pb->...p', angular, rho,
                        angular.conj(), optimize=True).real
    integral = np.einsum('...ac,ab,...bc->...', d, rho, d.conj(), optimize=True).real
    return dict(angular_strength_per_steradian=density,
                integrated_dipole_strength=integral, directions=xyz.copy(),
                polarization_density=rho.copy(),
                electron_energy_hartree=energy, residual_charge=charge)


def photoelectron_angular_parameters(amplitudes, channels, *,
        electron_energy_hartree, residual_charge):
    """Return the isotropic-ensemble linear-polarization anisotropy beta.

    Input dipoles have shape (...,3,channel), in SPARKS' real harmonics and
    G+iF scattering convention, before angular Coulomb/centrifugal phases.
    Apply exp(i*sigma_l-i*pi*l/2), sigma_l=arg Gamma(l+1+i*eta), explicitly.
    Matching must have reached this Coulomb asymptote in these channels.
    A coupled dipole asymptote must be propagated/transformed accordingly
    before using this Coulomb-only reconstruction.

    Uniform orientation averaging gives beta=3*A/B-1, with
    A=integral |n dot d(n)|^2 dOmega and B=integral |d(n)|^2 dOmega.
    The existing angular quadrature exactly integrates the finite harmonic
    products. No random orientations, Wigner cutoff or fitted beta is used.
    Zero-strength channels return NaN beta, since their anisotropy is undefined.
    """
    d = np.asarray(amplitudes, complex)
    energy = float(electron_energy_hartree)
    charge = float(residual_charge)
    if (d.ndim < 2 or d.shape[-2:] != (3,len(channels.labels))
            or not np.all(np.isfinite(d)) or not np.isfinite(energy)
            or energy <= 0 or not np.isfinite(charge)):
        raise ValueError('Require finite Cartesian/channel dipoles and positive energy')
    l = channels.angular_momenta
    eta = -charge/np.sqrt(2*energy)
    phase = np.exp(1j*(loggamma(l+1+1j*eta).imag-np.pi*l/2))
    angular = np.einsum('...ac,pc,c->...pa', d, channels.values, phase, optimize=True)
    radial = np.einsum('...pa,pa->...p', angular, channels.directions)
    a = np.einsum('...p,p->...', abs(radial)**2, channels.weights)
    b = np.sum(abs(d)**2,axis=(-1,-2))
    ratio = np.full_like(b, np.nan, dtype=float)
    np.divide(a, b, out=ratio, where=b>0)
    return dict(summed_dipole_strength=b, longitudinal_angular_integral=a,
                beta=3*ratio-1, residual_charge=charge,
                electron_energy_hartree=energy,
                convention='differential cross section = sigma/(4*pi)*(1+beta*P2(cos(theta)))')
