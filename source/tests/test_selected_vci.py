import numpy as np
from matrix_trinity.selected_vci import SelectedVCIWorkspace
from matrix_trinity.curvilinear_vpt2 import SonicVPT2Input, _compiled_operator_matrix_element
from matrix_gf.harmonic import HESSIAN_EIGENVALUE_TO_CM


def workspace():
    w = np.array([120., 180.])
    f3 = np.zeros((2,)*3); f3[0,0,0] = 8.
    f4 = np.zeros((2,)*4); f4[0,0,0,0] = 24.; f4[1,1,1,1] = 12.
    field = SonicVPT2Input(np.diag((w/HESSIAN_EIGENVALUE_TO_CM)**2),
        f3, f4, np.eye(2), np.zeros((2,)*3), np.zeros((2,)*4), 0.)
    return SelectedVCIWorkspace(field, modal_linear_cm=[2., 0.])


def test_full_matrix_and_diagonal_match_compiled_operator():
    owner = workspace()
    states = [(i,j) for i in range(5) for j in range(4)]
    reference = np.diag(np.asarray(states) @ owner.frequencies).astype(complex)
    for i, source in enumerate(states):
        for j, target in enumerate(states):
            reference[j,i] += _compiled_operator_matrix_element(source, target, owner.terms)
    np.testing.assert_allclose(owner.matrix(states), reference, atol=1e-12)
    np.testing.assert_allclose(owner.diagonal(states), reference.diagonal().real, atol=1e-12)


def test_external_quartic_residual_is_not_omitted_or_dressed_into_matrix():
    owner = workspace()
    states = [(0,0)]
    energy = owner.matrix(states).diagonal().real
    corrections = owner.external_corrections(states, np.ones((1,1)), energy)
    assert abs(corrections[(4,0)][0]) > 0
    for q, correction in corrections.items():
        residual = owner.couplings(states[0])[q]
        np.testing.assert_allclose(correction[0] * (energy[0]-owner.diagonal([q])[0]), residual)
    assert owner.last_residual_diagnostics['full_external_quartics']
    np.testing.assert_allclose(owner.matrix(states).diagonal().real, energy)


def test_selection_floor_keeps_degenerate_partner_visible():
    owner = workspace()
    q = (4,0)
    energy = owner.diagonal([q])
    correction = owner.external_corrections([(0,0)], np.ones((1,1)), energy)
    assert np.isfinite(correction[q]).all() and abs(correction[q][0]) > 0
    assert owner.last_residual_diagnostics['near_denominator_entries'] > 0


def test_compiled_action_preserves_ordered_momentum_products_and_large_space_fallback():
    from matrix_trinity.vci_operator_action import CompiledModalAction
    from matrix_trinity.curvilinear_vpt2 import _compiled_operator_couplings
    terms = ((( (0,('p','q','p')), (16,('q',)) ), 2.5),
             (( (2,('q','q')), (19,('q','q')) ), -.7),
             (( (1,('q','q','q','q')), ), .3))
    action=CompiledModalAction(terms)
    for size,occupation in ((21,0),(21,3),(21,12),(31,1)):
        state=tuple([occupation]*size)
        reference=_compiled_operator_couplings(state,terms)
        actual=action(state)
        assert set(reference)==set(actual)
        np.testing.assert_allclose([actual[q] for q in reference],list(reference.values()),atol=1e-12)
