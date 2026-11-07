"""Independent numerical checks for exact response arithmetic optimizations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from matrix_sparks.continuum.coulomb import coulomb_boundary
from matrix_sparks.continuum.response import screen_response, GalerkinSternheimerResponse
from matrix_sparks.continuum.response_boundary import coulomb_log_derivative


@pytest.mark.parametrize('energy,radius,charge', [
    (.03, 32., 1.), (.4, 8., 1.), (3., 40., 2.),
    (.01, 200., -1.), (.7, 8., -1.), (.2, 12., 0.),
])
@pytest.mark.parametrize('orders', [np.array([12, 0, 1, 3, 1, 8]), np.array([0, 1, 3])])
def test_direct_log_boundary_matches_full_waves(energy, radius, charge, orders):
    waves = coulomb_boundary(energy, radius, orders, charge)
    expected = (waves.irregular_derivative + 1j * waves.regular_derivative) / (
        waves.irregular + 1j * waves.regular)
    actual = coulomb_log_derivative(energy, radius, orders, charge)
    assert_allclose(actual, expected, rtol=3e-8, atol=3e-10)


@pytest.mark.parametrize('energy,radius,charge', [(.3, 32., 1.), (3., 40., 2.), (.01, 200., -1.)])
def test_open_log_propagation_against_independent_whittaker(energy, radius, charge):
    import mpmath as mp
    orders = [0, 3, 12]
    actual = coulomb_log_derivative(energy, radius, orders, charge)
    with mp.workdps(40):
        k = mp.sqrt(2 * energy)
        expected = []
        for order in orders:
            wave = lambda r: mp.whitw(1j * charge / k, order + .5, -2j * k * r)
            expected.append(complex(mp.diff(wave, mp.mpf(radius)) / wave(radius)))
    assert_allclose(actual, expected, rtol=3e-8, atol=3e-10)


@pytest.mark.parametrize('complex_kernel', [False, True])
def test_real_kernel_fast_product_preserves_complex_screening(complex_kernel):
    rng = np.random.default_rng(551)
    chi = (rng.normal(size=(31, 31)) + 1j * rng.normal(size=(31, 31))) / 31
    k = rng.normal(size=(31, 31)) / 31
    if complex_kernel:
        k = k + .1j * k.T
    ext = rng.normal(size=(31, 3)) + 1j * rng.normal(size=(31, 3))
    snapshots = [x.copy() for x in (chi, k, ext)]
    result = screen_response(chi, k, ext)
    expected = np.linalg.solve(np.eye(31) - k.astype(complex) @ chi, ext)
    assert_allclose(result.effective_potential, expected, rtol=2e-14, atol=2e-14)
    for x, snapshot in zip((chi, k, ext), snapshots):
        assert_allclose(x, snapshot, rtol=0, atol=0)


def test_closed_real_and_nonzero_loss_use_exact_arithmetic(monkeypatch):
    import matrix_sparks.continuum.response as owner
    native = owner.splu
    types = []

    def record(operator):
        types.append(operator.dtype.kind)
        return native(operator)

    monkeypatch.setattr(owner, 'splu', record)
    h = np.diag([-1., .3, .8])
    c = np.eye(3)[:, :1]
    source = np.array([[[0.], [1.], [.4]]])
    for loss, expected_type in ((0., 'f'), (1e-30, 'c')):
        boundary = lambda z: np.diag([0., -1j * loss, -1j * loss])
        solver = GalerkinSternheimerResponse(h, np.eye(3), c, [-1.], [2.], source,
                                             boundary_self_energy=boundary)
        value = solver.evaluate(.5)
        assert types[-2:] == [expected_type, expected_type]
        expected = sum(2 * sum(source[0, j, 0]**2 / (z - h[j, j] + 1j * loss)
                               for j in (1, 2)) for z in (-.5, -1.5))
        assert_allclose(value[0, 0], expected, atol=1e-15)


@pytest.mark.parametrize('omega', [0., .6, .7 + .02j])
def test_exact_degeneracy_batches_sources_without_merging_occupations(omega):
    h = np.diag([-1., -1., .2, .9])
    sources = np.array([[[.1, 0], [0, .3], [.2, .4], [.6, -.1]],
                        [[0, .2], [.4, 0], [.8, -.3], [-.2, .5]]])
    occupations = [2., 1.]
    solver = GalerkinSternheimerResponse(h, np.eye(4), np.eye(4)[:, :2],
                                         [-1., -1.], occupations, sources)
    result = solver.evaluate(omega)
    weights = 1 / (-1 + omega - np.diag(h)[2:]) + 1 / (-1 - omega - np.diag(h)[2:])
    expected = sum(o * (src[2:].T @ (weights[:, None] * src[2:]))
                   for o, src in zip(occupations, sources))
    assert_allclose(result, expected, rtol=2e-14, atol=2e-14)
    assert solver.last_diagnostics['factorizations'] == (1 if omega == 0 else 2)
