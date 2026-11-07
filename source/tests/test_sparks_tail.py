"""Outer-potential contracts and dilute-density arithmetic regression checks."""

import numpy as np
import pytest
from matrix_sparks.continuum import (
    AODensityGradient,
    CoulombTailPotential,
    InterpolatedRadialPotential,
)


def test_xc_coulomb_splice():
    calls = []

    def original(xyz):
        calls.append(xyz.copy())
        return np.full(len(xyz), -2.0)

    potential = CoulombTailPotential(original, inner_radius=2, outer_radius=4, charge=1)
    points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(potential(points), [-2.0, -2.0, -1.0 - 1.0 / 6.0, -0.25, -0.01])
    assert len(calls[0]) == 3
    calls.clear()
    np.testing.assert_allclose(potential(points[-2:]), [-0.25, -0.01])
    assert not calls
    assert potential(np.empty((0, 3))).shape == (0,)
    # At either end, deviation from the adjoining branch is O(h^3).
    for h in [1e-3, 1e-4]:
        assert abs(potential(np.array([[2 + h, 0.0, 0.0]]))[0] + 2) < 3 * h**3
        assert abs(potential(np.array([[4 - h, 0.0, 0.0]]))[0] + 1 / (4 - h)) < 3 * h**3
    with pytest.raises(ValueError):
        CoulombTailPotential(original, inner_radius=4, outer_radius=2, charge=1)


def test_radial_matrix_interpolation():
    r = np.linspace(1, 5, 9)
    v = (r**3)[:, None, None] * np.array([[[1.0, 0.2], [0.2, 2.0]]])
    potential = InterpolatedRadialPotential(r, v)
    x = np.linspace(1, 5, 35)
    np.testing.assert_allclose(
        potential(x), (x**3)[:, None, None] * np.array([[[1.0, 0.2], [0.2, 2.0]]]), rtol=1e-14
    )
    with pytest.raises(ValueError):
        potential(np.array([0.99]))
    with pytest.raises(ValueError):
        InterpolatedRadialPotential(r, v.astype(complex))
    bad = v.copy()
    bad[0, 0, 1] = 3
    with pytest.raises(ValueError):
        InterpolatedRadialPotential(r, bad)


def test_dilute_ao_gradient_does_not_underflow():
    class Molecule:
        cart = False

        def eval_gto(self, evaluator, points):
            # Density=1e-200, Cartesian gradient=(2e-200,0,0).
            ao = np.zeros((4, len(points), 1))
            ao[0, :, 0] = 1e-100
            ao[1, :, 0] = 1e-100
            return ao

    provider = AODensityGradient.__new__(AODensityGradient)
    provider.molecule, provider.density, provider.batch_size = Molecule(), np.ones((1, 1)), 4
    rho, grad = provider(np.zeros((3, 3)))
    np.testing.assert_allclose(rho / 1e-200, 1.0, rtol=1e-14)
    np.testing.assert_allclose(grad / 2e-200, 1.0, rtol=1e-14)


@pytest.mark.parametrize(
    "charge,coeff",
    [
        (0.0, [[0.1, 0.2], [0.2, -0.1]]),
        (1.0, [[0.1, 0.2], [0.2, -0.1]]),
        (1.0, [[-0.4, 0.2], [0.2, 0.1]]),
    ],
)
def test_dipole_boundary_against_whittaker(charge, coeff):
    import mpmath as mp
    from scipy.special import loggamma
    from matrix_sparks.continuum import DipoleCoulombBoundary

    l = np.array([0, 1])
    d = np.array(coeff)
    provider = DipoleCoulombBoundary(l, d, charge)
    energy, radius = 0.2, 12.0
    waves = provider(energy, radius)
    k = np.sqrt(2 * energy)
    eta = -charge / k
    h, hp = [], []
    with mp.workdps(40):
        for lam in provider.eigenvalues:
            order = mp.sqrt(mp.mpf(float(lam)) + 0.25)

            def outgoing(rho):
                return mp.exp(mp.pi * eta / 2) * mp.whitw(-1j * eta, order, -2j * rho)

            h.append(complex(outgoing(k * radius)))
            hp.append(complex(k * mp.diff(outgoing, k * radius)))
    phase = np.exp(1j * (-np.pi * l / 2 + np.imag(loggamma(l + 1 + 1j * eta))))
    norm = np.sqrt(2 / (np.pi * k))
    u = provider.rotation
    reference = ((u * (norm * np.array(h))) @ u.T) * phase[None, :]
    slope = ((u * (norm * np.array(hp))) @ u.T) * phase[None, :]
    np.testing.assert_allclose(
        waves.irregular + 1j * waves.regular, reference, rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(
        waves.irregular_derivative + 1j * waves.regular_derivative, slope, rtol=2e-8, atol=2e-9
    )


def test_dipole_tail_box_invariance():
    from matrix_sparks.continuum import (
        DipoleCoulombBoundary,
        assemble_sparse_radial,
        SparseResolventBackend,
    )

    d = np.array([[0.1, 0.2], [0.2, -0.1]])
    boundary = DipoleCoulombBoundary([0, 1], d, 1.0)

    def potential(r):
        x = np.maximum(r, 2.0)
        return -np.eye(2)[None, :, :] / x[:, None, None] + d[None, :, :] / x[:, None, None] ** 2

    def source(r):
        return (r**2 * np.maximum(1 - (r / 1.5) ** 2, 0) ** 4)[:, None, None] * np.array(
            [[[1.0], [0.4]]]
        )

    results = []
    for radius in [4.0, 10.0]:
        op = assemble_sparse_radial(
            np.linspace(0, radius, int(radius * 10) + 1), [0, 1], potential, source=source
        )
        solver = SparseResolventBackend(op.hamiltonian, op.overlap, op.boundary, op.sources)
        results.append(solver.evaluate(0.2, boundary(0.2, radius)))
    np.testing.assert_allclose(results[0].amplitudes, results[1].amplitudes, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(results[0].scattering, results[1].scattering, rtol=2e-8, atol=2e-9)
