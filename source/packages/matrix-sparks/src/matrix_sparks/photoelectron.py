"""Energy-conserving photoelectron lines from signed continuum dipoles.

The continuum is energy normalized, so no extra electron momentum factor is
inserted. Nuclear moments may come from harmonic, VPT2 or DVR calculations.
The electronic provider owns its approximation and common channel/phase gauge.
"""

import numpy as np
from scipy.constants import alpha, physical_constants


BOHR_SQUARED_TO_MB = (physical_constants["Bohr radius"][0] ** 2) / 1e-22


def gaussian_line_profiles(axis, centers, strengths, *, sigma, maximum_kernel_entries=1048576):
    """Normalized Gaussian line profiles with bounded two-dimensional workspace.

    Strengths have shape (..., lines), including signed interference signals.
    Every line and its full Gaussian tail is retained. The returned final axis
    follows the supplied grid; no deposition, interpolation or FFT is used.
    """
    x, c, s = np.asarray(axis), np.asarray(centers), np.asarray(strengths)
    if (np.iscomplexobj(x) or np.iscomplexobj(c) or np.iscomplexobj(s)
            or x.ndim != 1 or c.ndim != 1 or s.ndim < 1 or s.shape[-1] != len(c)
            or not all(np.all(np.isfinite(v)) for v in (x, c, s))
            or not np.isfinite(sigma) or sigma <= 0
            or isinstance(maximum_kernel_entries, bool)
            or not isinstance(maximum_kernel_entries, (int, np.integer))
            or maximum_kernel_entries < 1):
        raise ValueError('Finite real axes/line strengths, positive width and workspace required')
    x, c, s = x.astype(float), c.astype(float), s.astype(float)
    out = np.zeros(s.shape[:-1] + (len(x),))
    if not len(x) or not len(c):
        return out
    flat = s.reshape(-1, len(c))
    output = out.reshape(-1, len(x))
    nx = min(len(x), maximum_kernel_entries)
    nl = max(1, maximum_kernel_entries // nx)
    for i in range(0, len(x), nx):
        for j in range(0, len(c), nl):
            kernel = x[i:i+nx, None] - c[None, j:j+nl]
            kernel /= sigma
            np.square(kernel, out=kernel)
            kernel *= -0.5
            np.exp(kernel, out=kernel)
            kernel /= np.sqrt(2*np.pi)*sigma
            output[:, i:i+nx] += flat[:, j:j+nl] @ kernel.T
    return out


class ContinuumDipoleTable:
    """Explicit piecewise-linear interpolation of complex channel dipoles.

    Coefficients have axes (energy, Cartesian, channel, nuclear operator).
    Operator zero is constant; the others multiply caller-supplied nuclear
    moments. All samples must use the same phase, channel and molecular frame.
    Interpolation acts on amplitudes before squaring, never on intensities.
    Extrapolation and nonpositive electron energies are rejected. Accuracy
    must be checked against independently evaluated intermediate energies.
    """

    def __init__(self, energies_hartree, coefficients):
        e = np.asarray(energies_hartree, float)
        c = np.asarray(coefficients, complex)
        if (e.ndim != 1 or len(e) < 2 or not np.all(np.isfinite(e))
                or e[0] <= 0 or np.any(np.diff(e) <= 0)
                or c.ndim != 4 or c.shape[:2] != (len(e), 3)
                or min(c.shape[2:]) < 1 or not np.all(np.isfinite(c))):
            raise ValueError("Require increasing positive energies and finite (energy,3,channel,operator) dipoles")
        self.energies_hartree = e.copy()
        self.coefficients = c.copy()

    def __call__(self, energies_hartree):
        e = np.asarray(energies_hartree, float)
        grid = self.energies_hartree
        if (e.ndim != 1 or not np.all(np.isfinite(e))
                or np.any(e < grid[0]) or np.any(e > grid[-1])):
            raise ValueError("Requested electron energies are outside the continuum table")
        i = np.clip(np.searchsorted(grid, e, side="right") - 1, 0, len(grid)-2)
        f = ((e-grid[i])/(grid[i+1]-grid[i]))[:, None, None, None]
        return (1-f)*self.coefficients[i] + f*self.coefficients[i+1]


def fixed_photon_photoelectron_lines(photon_energies_hartree,
        transition_energies_hartree, nuclear_moments, dipole_provider, *,
        populations=None, channel_multiplicity=1.0,
        angular_channels=None, residual_charge=None):
    """Isotropic one-photon partial cross sections, in megabarns (Mb).

    Each transition has its own binding energy including electronic and
    nuclear energies. At each photon energy the provider is queried at
    E_e = E_photon - E_transition, only for strictly open transitions.
    The provider accepts a 1D vector of unique energies and returns complex
    (energy,3,channel,operator) coefficients. Nuclear moments have shape
    (transition,operator). No assumption on nuclear coordinates is made.

    Cross sections are 4*pi**2*alpha*omega/3 times summed energy-normalized
    dipole strengths. The factor 1/3 averages an isotropic molecular ensemble;
    no extra Dyson norm, electron momentum or spin degeneracy is introduced.
    Any physical channel multiplicity is explicit and defaults to one.
    Populations apply per transition and are not automatically normalized.
    Optional angular_channels and residual_charge also evaluate the FCHT
    anisotropy beta in the asymptotic convention documented by
    photoelectron_angular_parameters. They must be supplied together.
    """
    photon = np.asarray(photon_energies_hartree, float)
    binding = np.asarray(transition_energies_hartree, float)
    moments = np.asarray(nuclear_moments, complex)
    if binding.ndim != 1:
        raise ValueError("Transition energies must be a one-dimensional array")
    pop = np.ones(len(binding)) if populations is None else np.asarray(populations, float)
    if (photon.ndim != 1 or not len(photon) or np.any(photon <= 0)
            or binding.ndim != 1 or not len(binding) or np.any(binding <= 0)
            or moments.ndim != 2 or moments.shape[0] != len(binding)
            or moments.shape[1] < 1 or pop.shape != binding.shape
            or np.any(pop < 0) or not np.isfinite(channel_multiplicity)
            or channel_multiplicity <= 0
            or not all(np.all(np.isfinite(a)) for a in (photon, binding, moments, pop))):
        raise ValueError("Require positive photon/binding energies, finite nuclear moments and nonnegative populations")
    electron = photon[:, None]-binding[None, :]
    opened = electron > 0
    strengths = np.zeros((len(photon), 4, len(binding)))
    if (angular_channels is None) != (residual_charge is None):
        raise ValueError('Angular channels and residual charge must be supplied together')
    beta = np.full(electron.shape, np.nan) if angular_channels is not None else None
    if np.any(opened):
        energies, inverse = np.unique(electron[opened], return_inverse=True)
        c = np.asarray(dipole_provider(energies), complex)
        if (c.ndim != 4 or c.shape[:2] != (len(energies), 3)
                or c.shape[2] < 1 or c.shape[3] != moments.shape[1]
                or not np.all(np.isfinite(c))):
            raise ValueError("Continuum provider returned incompatible or nonfinite dipoles")
        pi, ti = np.nonzero(opened)
        coefficients = c[inverse]
        fc = coefficients[..., 0]*moments[ti, 0, None, None]
        ht = np.einsum("tsca,ta->tsc", coefficients[..., 1:], moments[ti, 1:], optimize=True)
        values = np.stack((np.sum(abs(fc)**2, axis=(1, 2)),
                           np.sum(abs(ht)**2, axis=(1, 2)),
                           2*np.einsum("tsc,tsc->t", fc, ht.conj()).real,
                           np.sum(abs(fc+ht)**2, axis=(1, 2))), axis=1)
        strengths[pi, :, ti] = values*pop[ti, None]
        if angular_channels is not None:
            from .angular_observables import photoelectron_angular_parameters
            for row, (p, t) in enumerate(zip(pi, ti, strict=True)):
                beta[p, t] = photoelectron_angular_parameters(fc[row]+ht[row],
                    angular_channels, electron_energy_hartree=electron[p, t],
                    residual_charge=residual_charge)['beta']
    factor = 4*np.pi**2*alpha/3*photon*BOHR_SQUARED_TO_MB*channel_multiplicity
    result = dict(schema="matrix.sparks.fixed_photon_photoelectron.v1",
                photon_energies_hartree=photon,
                transition_energies_hartree=binding,
                electron_energies_hartree=electron, open_channels=opened,
                strengths=strengths, cross_sections_Mb=strengths*factor[:, None, None],
                operator_components=("FC", "HT", "interference", "FCHT"),
                orientation="isotropic", continuum_normalization="delta(E-E')",
                channel_multiplicity=float(channel_multiplicity))
    if beta is not None:
        result['beta'] = beta
    return result
