"""Explicit spatial Coulomb tails and cached radial matrix interpolation.

Selecting a Coulomb tail changes the local model outside a declared interval.
It is never selected implicitly by an electronic importer or by LB94Potential.
"""

import numpy as np
from scipy.interpolate import CubicSpline


class CoulombTailPotential:
    """Join a scalar spatial potential to -charge/r with a C2 switch.

    Inside ``inner_radius`` the supplied potential is unchanged; beyond
    ``outer_radius`` only the Coulomb tail is evaluated. Radii and origin are in
    bohr, potential values in Hartree. Applying this to XC alone preserves the
    separately supplied nuclear and Hartree multipoles. The transition interval
    is a model choice requiring a sensitivity check, not a solver tolerance.
    """

    def __init__(self, potential, *, inner_radius, outer_radius, charge, origin_bohr=(0, 0, 0)):
        if (
            not np.all(np.isfinite([inner_radius, outer_radius, charge]))
            or not 0 < inner_radius < outer_radius
        ):
            raise ValueError("Require finite charge and 0 < inner < outer radius")
        origin = np.asarray(origin_bohr, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("Origin must be a finite Cartesian vector")
        self.potential = potential
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.charge = float(charge)
        self.origin = origin.copy()

    def __call__(self, points_bohr):
        raw = np.asarray(points_bohr)
        if np.iscomplexobj(raw):
            raise ValueError("Potential coordinates must be real")
        points = np.asarray(raw, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("Require finite (N,3) coordinates")
        radii = np.hypot.reduce(points - self.origin, axis=1)
        inside = radii <= self.inner_radius
        outside = radii >= self.outer_radius
        active = ~outside
        result = np.empty(len(points))
        values = np.asarray(self.potential(points[active])) if np.any(active) else np.empty(0)
        if (
            values.shape != (int(active.sum()),)
            or np.iscomplexobj(values)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Spatial potential must return finite real values")
        result[active] = values
        result[outside] = -self.charge / radii[outside]
        transition = ~inside & ~outside
        t = (radii[transition] - self.inner_radius) / (self.outer_radius - self.inner_radius)
        switch = t**3 * (10 + t * (-15 + 6 * t))
        result[transition] = (1 - switch) * result[transition] - switch * self.charge / radii[
            transition
        ]
        return result


class InterpolatedRadialPotential:
    """Cache a symmetric channel potential on an explicit radial interval.

    Uses SciPy's cubic interpolation and refuses extrapolation. All input
    matrices are retained without thresholding. Grid convergence must be checked
    by the caller against a denser sampling of the same analytic potential.
    This avoids reevaluating AO integrals inside an outer-region ODE solver.
    """

    def __init__(self, radii, matrices):
        raw_r, raw_v = np.asarray(radii), np.asarray(matrices)
        if np.iscomplexobj(raw_r) or np.iscomplexobj(raw_v):
            raise ValueError("Radial potentials must be real")
        r, v = np.asarray(raw_r, float), np.asarray(raw_v, float)
        if (
            r.ndim != 1
            or len(r) < 4
            or not np.all(np.isfinite(r))
            or r[0] <= 0
            or np.any(np.diff(r) <= 0)
        ):
            raise ValueError("Require at least four positive strictly increasing radii")
        if v.ndim != 3 or v.shape[0] != len(r) or v.shape[1] != v.shape[2] or v.shape[1] < 1:
            raise ValueError("Potential must have shape (radii, channels, channels)")
        if not np.all(np.isfinite(v)) or not np.allclose(
            v, v.swapaxes(1, 2), rtol=1e-12, atol=1e-13
        ):
            raise ValueError("Potential matrices must be finite and symmetric")
        self.inner_radius, self.outer_radius = float(r[0]), float(r[-1])
        self._spline = CubicSpline(r, v, axis=0, extrapolate=False)

    def __call__(self, radii):
        raw = np.asarray(radii)
        if np.iscomplexobj(raw):
            raise ValueError("Radial coordinates must be real")
        r = np.asarray(raw, float)
        if (
            r.ndim != 1
            or not np.all(np.isfinite(r))
            or np.any(r < self.inner_radius)
            or np.any(r > self.outer_radius)
        ):
            raise ValueError("Potential requested outside its sampled radial interval")
        return self._spline(r)
