import numpy as np
from matrix_trinity import HarmonicVibronicModel
from matrix_trinity.vibronic_variational_observables import (
    project_sine_grid_states,
    finite_basis_operator_spectrum,
)


def test_grid_projection_and_untruncated_coordinate_action():
    axis = np.linspace(-7, 7, 141)
    phi = np.pi ** (-0.25) * np.exp(-(axis**2) / 2)
    u = np.column_stack([phi, np.sqrt(2) * axis * phi]) * np.sqrt(axis[1] - axis[0])
    p = project_sine_grid_states(axis[:, None], u, maximum_total_quanta=4)
    np.testing.assert_allclose(p["vectors"][:2], np.eye(2), atol=2e-14)
    assert p["diagnostics"]["maximum_gram_defect"] < 1e-13
    model = HarmonicVibronicModel([1000], [1000], [[1]], [0], 10000, [0, 0, 0], np.zeros((3, 1)))
    out = finite_basis_operator_spectrum(
        model,
        np.array([[0]]),
        np.ones((1, 1)),
        np.array([[0], [1]]),
        np.eye(2),
        [1, 0],
        np.array([np.eye(2)]),
        gradients=[[0], [1]],
    )
    np.testing.assert_allclose(out["moments"][:, 0], [[1, 0], [0, 1 / np.sqrt(2)]], atol=1e-14)
    np.testing.assert_allclose(out["strengths"][0, :, 0], [1, 0.5], atol=1e-14)


def test_source_norm_exposes_projection_loss_without_renormalization():
    model = HarmonicVibronicModel([1000], [1000], [[1]], [0], 10000, [0,0,0], np.zeros((3,1)))
    result = finite_basis_operator_spectrum(model,np.array([[0]]),np.ones((1,1)),
        np.array([[0]]),np.ones((1,1)),[1,0],np.array([np.eye(2)]),gradients=[[0],[1]])
    np.testing.assert_allclose(result['source_strengths'],[[1.5]],atol=1e-14)
    np.testing.assert_allclose(result['retained_source_strengths'],[[1.]],atol=1e-14)
    np.testing.assert_allclose(result['strengths'],[[[1.]]],atol=1e-14)
