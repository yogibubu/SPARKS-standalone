"""Total Cartesian derivatives of a geometry-dependent electronic parameter.

Electronic partial derivatives remain owned by the electronic backend. SMITH
supplies the geometric parameter derivatives. No electronic jobs are launched.
"""
import numpy as np


def rigid_fragment_interaction(energy, gradient, coordinates_angstrom, fragments, *,
                               geometry_tolerance_angstrom=1.e-6):
    """Subtract isolated-fragment E/G after rotating each saved reference.

    Each fragment supplies atom_indices_zero_based, coordinates_angstrom,
    energy_hartree and gradient_hartree_per_bohr. The atom partition must be
    complete. A deformed fragment requires new electronic data; its gradient
    cannot be subtracted as if it were a rotated frozen reference.
    """
    from matrix_chem.geometry_alignment import kabsch_rotation, rotate_cartesian_derivatives
    xyz = np.asarray(coordinates_angstrom, float)
    g = np.asarray(gradient, float).reshape(xyz.shape).copy()
    e = float(energy)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz)
            or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(g))
            or not np.isfinite(e) or not np.isfinite(geometry_tolerance_angstrom)
            or geometry_tolerance_angstrom <= 0):
        raise ValueError('finite Cartesian E/G and positive geometry tolerance required')
    covered = []
    for fragment in fragments:
        indices = np.asarray(fragment['atom_indices_zero_based'])
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= len(xyz))):
            raise ValueError('valid fragment atom indices required')
        source = np.asarray(fragment['coordinates_angstrom'], float)
        if source.shape != (len(indices), 3) or not np.all(np.isfinite(source)):
            raise ValueError('finite matching isolated-fragment reference required')
        target = xyz[indices]
        rotation = kabsch_rotation(source, target)
        mismatch = (source-source.mean(axis=0))@rotation-(target-target.mean(axis=0))
        if np.max(abs(mismatch)) > geometry_tolerance_angstrom:
            raise ValueError('fragment is deformed relative to its electronic reference')
        fg = np.asarray(fragment['gradient_hartree_per_bohr'], float).reshape(source.shape)
        fe = float(fragment['energy_hartree'])
        if not np.all(np.isfinite(fg)) or not np.isfinite(fe):
            raise ValueError('finite isolated-fragment E/G required')
        rotated, _ = rotate_cartesian_derivatives(fg, rotation)
        g[indices] -= rotated.reshape(source.shape)
        e -= fe
        covered.extend(indices.tolist())
    if sorted(covered) != list(range(len(xyz))):
        raise ValueError('isolated fragments must partition all atoms exactly once')
    return e, g.ravel()


def linear_parameter_energy_gradient(energy_zero, gradient_zero, energy_one,
                                     gradient_one, parameter, parameter_gradient):
    """Compose linear electronic endpoints with the complete first chain rule."""
    a = float(parameter)
    e0, e1 = float(energy_zero), float(energy_one)
    g0, g1, ga = (np.asarray(v, float) for v in
                  (gradient_zero, gradient_one, parameter_gradient))
    if (g0.ndim != 1 or not g0.size or g1.shape != g0.shape or ga.shape != g0.shape
            or not 0 <= a <= 1 or not np.all(np.isfinite([e0, e1, a]))
            or not all(np.all(np.isfinite(v)) for v in (g0, g1, ga))):
        raise ValueError('finite compatible endpoints and bounded parameter required')
    return (1-a)*e0+a*e1, (1-a)*g0+a*g1+(e1-e0)*ga


def geometry_dependent_parameter_jet(gradient_fixed, hessian_fixed, *,
        energy_parameter_first, energy_parameter_second,
        gradient_parameter, parameter_gradient, parameter_hessian):
    """Return total gradient/Hessian of ``E(x, a(x))`` in consistent units.

    ``gradient_parameter`` is d²E/(dx da), not the gradient along a(x).
    Both fixed-parameter electronic response and the complete chain rule are
    required. This also covers nonlinear hybrid prescriptions (E_aa != 0).
    """
    g, h, ga, ax, axx = (np.asarray(v, float) for v in
        (gradient_fixed, hessian_fixed, gradient_parameter,
         parameter_gradient, parameter_hessian))
    ea, eaa = float(energy_parameter_first), float(energy_parameter_second)
    n = g.size
    if (g.ndim != 1 or not n or h.shape != (n, n) or ga.shape != (n,)
            or ax.shape != (n,) or axx.shape != (n, n)
            or not all(np.all(np.isfinite(v)) for v in (g, h, ga, ax, axx, ea, eaa))):
        raise ValueError('finite compatible Cartesian and parameter derivatives required')
    if not all(np.allclose(v, v.T, rtol=1e-10, atol=1e-12) for v in (h, axx)):
        raise ValueError('electronic and parameter Hessians must be symmetric')
    total_g = g + ea * ax
    total_h = (0.5 * (h + h.T) + np.outer(ga, ax) + np.outer(ax, ga)
               + eaa * np.outer(ax, ax) + ea * 0.5 * (axx + axx.T))
    return total_g, total_h
