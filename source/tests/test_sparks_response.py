"""Independent analytic controls for retarded screening and projected response."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eigh

from matrix_sparks.continuum.response import GalerkinSternheimerResponse, screen_response


def test_screening_matches_real_block_reference_equations():
    rng = np.random.default_rng(9038)
    n = 17
    chi = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / n
    k = rng.normal(size=(n, n)) / n
    ext = rng.normal(size=(n, 3))
    result = screen_response(chi, k, ext)
    a, b = k @ chi.real - np.eye(n), k @ chi.imag
    # TIRESIA's real 2N formulation, independently solved.
    block = np.block([[a, -b], [b, a]])
    reference = np.linalg.solve(block, np.concatenate((-ext, np.zeros_like(ext))))
    assert_allclose(result.effective_potential, reference[:n] + 1j * reference[n:], atol=2e-15)
    assert_allclose(result.induced_density, chi @ result.effective_potential, atol=1e-15)
    assert result.backward_error < 1e-15


def test_amplitude_contraction_preserves_complex_phases():
    ext = np.array([[1, 2j], [1j, 2]])
    result = screen_response(np.eye(2) * 0.2j, np.zeros((2, 2)), ext)
    vertices = np.array([[1j, 2], [2, 3j], [1, 2j]])
    assert_allclose(result.amplitudes(vertices), vertices @ ext)
    assert_allclose(result.effective_potential, ext)


def model():
    rng = np.random.default_rng(122)
    n, nocc, p = 14, 3, 5
    transform = np.eye(n) + 0.03 * rng.normal(size=(n, n))
    s = transform.T @ transform
    h0 = rng.normal(size=(n, n))
    h0 = h0 + h0.T
    h = transform.T @ h0 @ transform
    energies, orbitals = eigh(h, s)
    potentials = rng.normal(size=(p, n, n))
    potentials = potentials + potentials.transpose(0, 2, 1)
    sources = np.einsum("apq,qi->ipa", potentials, orbitals[:, :nocc])
    occupation = np.array([2.0, 2.0, 1.0])
    response = GalerkinSternheimerResponse(
        h, s, orbitals[:, :nocc], energies[:nocc], occupation, sources
    )
    return response, energies, orbitals, nocc, occupation, sources


@pytest.mark.parametrize("omega", [0, 0.3, 0.7 + 0.02j, 3 + 0.1j])
def test_sternheimer_matches_complete_virtual_lehmann_sum(omega):
    response, energies, c, nocc, occ, sources = model()
    result = response.evaluate(omega)
    reference = np.zeros_like(result)
    for i in range(nocc):
        m = c[:, nocc:].T @ sources[i]
        delta = energies[nocc:] - energies[i]
        weight = 1 / (omega - delta) + 1 / (-omega - delta)
        reference += occ[i] * np.einsum("ap,a,aq->pq", m, weight, m)
    assert_allclose(result, reference, rtol=1e-11, atol=1e-11)
    assert response.last_diagnostics["maximum_backward_error"] < 1e-14
    assert response.last_diagnostics["factorizations"] == nocc * (1 if omega == 0 else 2)


def test_occupied_contamination_is_projected_out():
    response, energies, c, nocc, occ, sources = model()
    before = response.evaluate(0.6 + 0.01j)
    response.sources += (response.overlap @ c[:, :nocc]) @ np.ones((nocc, sources.shape[-1]))
    assert_allclose(response.evaluate(0.6 + 0.01j), before, rtol=1e-11, atol=1e-11)


def test_explicit_boundary_and_damping_are_not_ignored():
    h = np.diag([-1.0, 0.2, 0.5])
    s = np.eye(3)
    sources = np.array([[[0.0], [1.0], [0.4]]])

    def boundary(z):
        # Explicit constant retarded loss on the virtual subspace, advanced below.
        sign = 1 if z.imag >= 0 else -1
        return np.diag([0.0, -0.1j * sign, -0.2j * sign])

    response = GalerkinSternheimerResponse(
        h, s, np.eye(3)[:, :1], [-1.0], [2.0], sources, boundary_self_energy=boundary
    )
    omega = 1.4 + 0.01j
    expected = 2 * (
        1 / (omega - 1.2 + 0.1j)
        + 1 / (-omega - 1.2 - 0.1j)
        + 0.16 * (1 / (omega - 1.5 + 0.2j) + 1 / (-omega - 1.5 - 0.2j))
    )
    assert_allclose(response.evaluate(omega), [[expected]])
    assert response.last_diagnostics["boundary"] == "explicit self-energy"


def test_reject_nonstationary_occupied_states():
    with pytest.raises(ValueError, match="stationary"):
        GalerkinSternheimerResponse(
            np.diag([-1.0, 1.0]),
            np.eye(2),
            np.array([[1.0], [0.0]]),
            [-0.9],
            [2.0],
            np.zeros((1, 2, 1)),
        )


@pytest.mark.parametrize("omega", [-1, -1j, np.nan])
def test_reject_advanced_or_nonfinite_frequency(omega):
    response, *_ = model()
    with pytest.raises(ValueError, match="retarded"):
        response.evaluate(omega)


def test_reject_invalid_dimensions_and_vertices():
    with pytest.raises(ValueError):
        screen_response(np.eye(2), np.eye(3), np.ones((2, 3)))
    result = screen_response(np.eye(2), np.zeros((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        result.amplitudes(np.ones((3, 4)))


def test_auxiliary_metric_covariance():
    from matrix_sparks.continuum.response import project_adiabatic_kernel

    rng = np.random.default_rng(5)
    values = rng.normal(size=(81, 7))
    w = rng.uniform(0.1, 1, size=81)
    fxc = -rng.uniform(0, 2, size=81)
    j = rng.normal(size=(7, 7))
    j = j @ j.T
    k = project_adiabatic_kernel(j, values, w, fxc)
    metric = values.T @ (w[:, None] * values)
    assert_allclose(metric @ k @ metric, j + values.T @ ((w * fxc)[:, None] * values), atol=2e-14)
    # Same physical potential/density in a changed auxiliary representation.
    transform = np.eye(7) + 0.05 * rng.normal(size=(7, 7))
    changed = project_adiabatic_kernel(transform.T @ j @ transform, values @ transform, w, fxc)
    inverse = np.linalg.inv(transform)
    assert_allclose(changed, inverse @ k @ inverse.T, atol=1e-15)


def test_streamed_orbital_sources():
    from matrix_sparks.continuum.response import orbital_potential_sources

    rng = np.random.default_rng(6)
    b = rng.normal(size=(35, 8))
    c = rng.normal(size=(35, 3))
    p = rng.normal(size=(35, 7))
    w = rng.uniform(0.1, 1, size=35)
    actual = orbital_potential_sources(b, c, p, w, block_points=4)
    expected = np.einsum("gb,gi,gp,g->ibp", b, c, p, w)
    assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("energy", [-0.7, -0.11, -0.5 + 0.03j, 0.8 + 0.02j])
def test_closed_and_complex_coulomb_boundary_against_whittaker(energy):
    import mpmath as mp
    from matrix_sparks.continuum.response_boundary import coulomb_log_derivative

    radius = 7.0
    charge = 1.0
    orders = np.array([0, 1, 3])
    result = coulomb_log_derivative(energy, radius, orders, charge)
    with mp.workdps(35):
        k = mp.sqrt(2 * energy)
        reference = []
        for l in orders:

            def h(r):
                return mp.whitw(1j * charge / k, int(l) + 0.5, -2j * k * r)

            reference.append(complex(mp.diff(h, mp.mpf(radius)) / h(radius)))
    assert_allclose(result, reference, rtol=3e-9, atol=3e-10)
    assert_allclose(
        coulomb_log_derivative(complex(energy).conjugate(), radius, orders, charge),
        result.conjugate(),
        rtol=3e-9,
        atol=3e-10,
    )


def test_retarded_self_energy_sign_and_tail():
    from matrix_sparks.continuum.response_boundary import (
        CoulombResponseBoundary,
        coulomb_log_derivative,
    )

    b = np.array([[1.0, 0.0], [0.0, 2.0], [0.1, 0.2]])
    boundary = CoulombResponseBoundary(b, 8.0, [0, 1], 1.0)
    q = coulomb_log_derivative(0.4, 8.0, [0, 1], 1.0)
    assert_allclose(boundary(0.4).toarray(), -0.5 * (b * q) @ b.T)
    tail = lambda r: np.asarray([-np.eye(2) / x for x in r])
    propagated = CoulombResponseBoundary(
        b, 8.0, [0, 1], 1.0, outer_radius=12.0, tail_potential=tail
    )
    for z in (0.4, -0.3, 0.4 + 0.01j):
        assert_allclose(propagated(z).toarray(), boundary(z).toarray(), rtol=3e-9, atol=3e-10)


@pytest.mark.parametrize("omega", [0, 0.5, 0.7 + 0.01j])
def test_complete_spectral_response_matches_sparse_projected_resolvent(omega):
    from matrix_sparks.continuum.response import SpectralSternheimerResponse

    sparse, energies, c, nocc, occ, sources = model()
    fast = SpectralSternheimerResponse(
        sparse.hamiltonian, sparse.overlap, c[:, :nocc], energies[:nocc], occ, sources
    )
    assert_allclose(fast.evaluate(omega), sparse.evaluate(omega), rtol=1e-10, atol=1e-10)


def test_complete_spectral_response_open_boundary_and_inner_pole():
    from matrix_sparks.continuum.response import SpectralSternheimerResponse
    from matrix_sparks.continuum.response_boundary import CoulombResponseBoundary

    h = np.diag([-1.0, 0.2, 0.5, 1.0])
    s = np.eye(4)
    c = np.eye(4)[:, :1]
    sources = np.array([[[0.0, 0.1], [0.1, 1.0], [0.3, -0.4], [0.1, -0.1]]])
    boundary = CoulombResponseBoundary(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.3]]), 7.0, [0, 1], 1.0
    )
    sparse = GalerkinSternheimerResponse(
        h, s, c, [-1.0], [2.0], sources, boundary_self_energy=boundary
    )
    fast = SpectralSternheimerResponse(h, s, c, [-1.0], [2.0], sources, boundary=boundary)
    for omega in (0.5, 1.3, 1.4 + 0.02j, 1.2):
        assert_allclose(fast.evaluate(omega), sparse.evaluate(omega), rtol=1e-10, atol=1e-10)
    assert fast.last_diagnostics["method"] == "sparse inner-pole fallback"


def test_lda_kernel_against_independent_potential_derivative():
    pytest.importorskip("pyscf")
    from matrix_sparks.continuum.lb94 import lda_xc_density_derivative, lb94_xc

    rho = np.geomspace(1e-6, 10.0, 31)
    step = 1e-4
    derivative = (
        lb94_xc(rho * (1 - 2 * step), np.zeros_like(rho))
        - 8 * lb94_xc(rho * (1 - step), np.zeros_like(rho))
        + 8 * lb94_xc(rho * (1 + step), np.zeros_like(rho))
        - lb94_xc(rho * (1 + 2 * step), np.zeros_like(rho))
    ) / (12 * step * rho)
    assert_allclose(lda_xc_density_derivative(rho), derivative, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("energy", [-20.0, -1.0, -0.01, -0.003])
def test_closed_coulomb_hyperu_across_core_and_near_threshold(energy):
    import mpmath as mp
    from matrix_sparks.continuum.response_boundary import coulomb_log_derivative

    radius = 32.0
    orders = [0, 1, 3, 12]
    actual = coulomb_log_derivative(energy, radius, orders, 1.0)
    with mp.workdps(40):
        kappa = mp.sqrt(-2 * energy)
        expected = []
        for l in orders:

            def h(r):
                return mp.whitw(1 / kappa, l + 0.5, 2 * kappa * r)

            expected.append(complex(mp.diff(h, mp.mpf(radius)) / h(radius)))
    assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)


def test_reference_screening_stream_sign_and_fourier_conversion(tmp_path):
    from matrix_sparks.reference.tiresia.overlays.response_export import read_screening_export

    chi = np.array([[0.2 + 0.3j, 0.1j], [0.1j, 0.4 + 0.2j]])
    kernel = np.array([[0.3, 0.02], [0.02, 0.5]])
    rhs = np.array([0.5, 1.0])
    native = np.linalg.solve(kernel @ chi - np.eye(2), rhs)
    target = tmp_path / "response.bin"
    with target.open("wb") as stream:
        np.array([2, 1], dtype="<i4").tofile(stream)
        for x in (chi.real, chi.imag, kernel):
            np.asarray(x, dtype="<f8").ravel(order="F").tofile(stream)
        rhs.astype("<f8").tofile(stream)
        np.array([0], dtype="<i4").tofile(stream)
        np.r_[native.real, native.imag].astype("<f8").tofile(stream)
    data = read_screening_export(target)
    answer = screen_response(data["chi"], data["kernel"], data["external_potential"])
    assert_allclose(answer.effective_potential, data["effective_potential"], atol=1e-15)
    assert_allclose(data["effective_potential"][:, 0], -native.conjugate())


def test_response_continuum_header_and_normalization(tmp_path):
    import struct
    from scipy.io import FortranFile
    from matrix_sparks.reference.tiresia.continuum_data import read_response_continuum

    static = tmp_path / "static.bin"
    response = tmp_path / "response.bin"
    for path, kind in ((static, 0), (response, 2)):
        with FortranFile(path, "w", header_dtype=np.dtype("<u4")) as stream:
            header = b"01 JAN2026" + b"12:00:00  " + b"C1      " + struct.pack("<i", kind)
            assert len(header) == 32
            stream.write_record(np.frombuffer(header, dtype=np.uint8))
            dimensions = struct.pack("<7id5i", 1, 1, 1, 1, 1, 0, 1, 1.0, 1, 1, 1, 1, 1)
            stream.write_record(np.frombuffer(dimensions, dtype=np.uint8))
            angular = (
                struct.pack("<9i", 0, 1, 0, 1, 1, 1, 1, 1, 1) + b"A         " + struct.pack("<i", 1)
            )
            stream.write_record(np.frombuffer(angular, dtype=np.uint8))
            stream.write_record(np.array([1.0], dtype="<f8"))
            stream.write_record(np.array([0], dtype="<i4"))
            stream.write_record(np.array([-0.5], dtype="<f8"))
            if kind == 2:
                stream.write_record(np.frombuffer(struct.pack("<di", 1.3, 2), dtype=np.uint8))
                stream.write_record(np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype="<c16"))
    result = read_response_continuum(response, static)
    frame = result.frames[0]
    assert frame.response_kind == 2
    expected = np.sqrt(4 * np.pi / 3) * np.array([5 + 6j, 1 + 2j, 3 + 4j])
    assert_allclose(frame.amplitudes[0][0, :, 0, 0], expected)
    sigma = result.cross_sections(occupations=[1.0])
    assert_allclose(
        sigma[0, 0],
        4 * np.pi**2 * 7.29735308e-3 / 3 * 1.3 * np.sum(abs(expected) ** 2) * 0.529177249**2 * 100,
    )
