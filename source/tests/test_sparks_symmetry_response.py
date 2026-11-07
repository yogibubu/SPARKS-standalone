"""Initial-state poles and target symmetry projectors are distinct objects."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from matrix_sparks.continuum.response import GalerkinSternheimerResponse, SpectralSternheimerResponse


@pytest.mark.parametrize('nprojected', [0, 2])
@pytest.mark.parametrize('omega', [.6, .7+.01j, 0.])
def test_distinct_target_projector_matches_explicit_virtual_sum(nprojected, omega):
    energies = np.array([-2., -1., .3, .8, 1.4])
    poles = np.array([-3., -1.5, -.9])
    occupations = np.array([2., 2., 1.])
    sources = np.random.default_rng(91).normal(size=(3, 5, 4))
    args = (np.diag(energies), np.eye(5), np.eye(5)[:, :nprojected], poles,
            occupations, sources)
    expected = np.zeros((4, 4), complex)
    for e, occ, source in zip(poles, occupations, sources):
        m = source[nprojected:]
        weights = (1/(e+omega-energies[nprojected:]) +
                   1/(e-omega-energies[nprojected:]))
        expected += occ * (m.T @ (weights[:, None]*m))
    for owner in (GalerkinSternheimerResponse, SpectralSternheimerResponse):
        result = owner(*args, projection_energies_hartree=energies[:nprojected])
        assert_allclose(result.evaluate(omega), expected, rtol=3e-13, atol=3e-13)


def test_missing_or_nonstationary_target_projector_energies_rejected():
    args = (np.diag([-2., -.5, 1.]), np.eye(3), np.eye(3)[:, :1],
            [-3., -1.], [2., 2.], np.ones((2, 3, 1)))
    with pytest.raises(ValueError, match='projection energies'):
        GalerkinSternheimerResponse(*args)
    with pytest.raises(ValueError, match='stationary'):
        GalerkinSternheimerResponse(*args, projection_energies_hartree=[-1.])


def test_sample_boundary_ratio_matches_coulomb_waves():
    from matrix_sparks.continuum.response_boundary import CoulombSampleBoundary
    from matrix_sparks.continuum.coulomb import coulomb_boundary
    radii = [11., 12.]
    orders = np.array([0, 2])
    samples = np.zeros((2, 2, 4))
    samples[0, :, :2] = np.eye(2)
    samples[1, :, 2:] = np.eye(2)
    owner = CoulombSampleBoundary(np.eye(4)[:, -2:], samples, radii, orders, 1.)
    waves = [coulomb_boundary(.7, r, orders, 1.) for r in radii]
    ratio = ((waves[1].irregular+1j*waves[1].regular)/
             (waves[0].irregular+1j*waves[0].regular))
    assert_allclose(owner.constraint(.7)[:, :2], -np.diag(ratio), atol=2e-9)
    assert_allclose(owner.constraint(.7-.01j), owner.constraint(.7+.01j).conj(), atol=2e-9)
    assert_allclose(owner.constraint(-.3).toarray(), np.eye(4)[-2:])


@pytest.mark.parametrize('nprojected', [0, 1])
@pytest.mark.parametrize('omega', [.7, 1.4, 1.2, 1.4+.01j])
def test_sample_constraint_matches_dense_augmented_resolvent(nprojected, omega):
    from matrix_sparks.continuum.response_boundary import CoulombSampleBoundary
    energies = np.array([-2., -.4, .3, .8, 1.4])
    h = np.diag(energies)
    c = np.eye(5)[:, :nprojected]
    boundary = np.eye(5)[:, -1:]
    samples = np.array([[[0., .1, .3, .8, .2]], [[0., .2, .4, .6, .9]]])
    provider = CoulombSampleBoundary(boundary, samples, [11., 12.], [0], 1.)
    source = np.random.default_rng(18).normal(size=(1, 5, 2))
    expected = np.zeros((2, 2), complex)
    for z in [-.9+omega, -.9-omega]:
        constraint = provider.constraint(z)
        constraint = constraint.toarray() if hasattr(constraint, 'toarray') else constraint
        augmented = np.block([[z*np.eye(5)-h, c, boundary],
                              [c.T, np.zeros((nprojected, nprojected+1))],
                              [constraint, np.zeros((1, nprojected+1))]])
        rhs = np.zeros((6+nprojected, 2), complex)
        rhs[:5] = source[0]
        expected += 2*source[0].T @ np.linalg.solve(augmented, rhs)[:5]
    for owner, keyword in [(GalerkinSternheimerResponse, 'boundary_self_energy'),
                           (SpectralSternheimerResponse, 'boundary')]:
        result = owner(h, np.eye(5), c, [-.9], [2.], source,
                       projection_energies_hartree=energies[:nprojected], **{keyword: provider})
        assert_allclose(result.evaluate(omega), expected, rtol=2e-12, atol=2e-12)
