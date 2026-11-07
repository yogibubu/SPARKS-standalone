import numpy as np
import pytest
from matrix_core.protocol_library import load_keymaker_protocol_library
from matrix_trinity.reference_consistent_composite import (
    assemble_reference_consistent_field_full_hessian, tensor_fingerprint,
    require_stationary_vpt2_qualification)


def request_and_jet():
    jet=(np.array(-17.),np.array([3.,-2.]),np.array([[20.,5.],[5.,40.]]),
         np.ones((2,)*3),np.ones((2,)*4)*2)
    ref=dict(geometry_sha256='a'*64,coordinate_sha256='b'*64,masses_sha256='c'*64,
             atom_order=[6,1],isotopologue='test',method='analytic',basis='test',
             triples_scaling='none',normalization='dimensionless_q_G0_diagonal_omega_ref_cm')
    request=dict(schema='matrix.trinity.reference_consistent_composite_request.v2',
        protocol_sha256=load_keymaker_protocol_library().resolve('trinity-reference-consistent-composite-field.v2').sha256,
        parent_tensor_sha256=tensor_fingerprint(jet),update_id='test-complete',applied_update_ids=[],
        reference=ref,experimental_data_used=False,reference_policy='retain_origin_and_all_linear_forces',
        acquired_jet=dict(gradient_cm=[.3,-.2],hessian_cm=[[21.,-7.],[-7.,39.]],
            geometry_sha256='a'*64,coordinate_sha256='b'*64,electronic_evidence_sha256='d'*64,
            coverage='complete_analytic_gradient_and_hessian',derivative_estimation='analytic'))
    return request,jet


def test_full_hessian_replaces_acquired_matrix_and_preserves_forces_higher_orders():
    request,jet=request_and_jet()
    result,receipt=assemble_reference_consistent_field_full_hessian(request,jet)
    np.testing.assert_array_equal(result[1],[.3,-.2])
    np.testing.assert_array_equal(result[2],[[21.,-7.],[-7.,39.]])
    for index in (0,3,4):
        np.testing.assert_array_equal(result[index],jet[index])
    assert receipt['maximum_offdiagonal_increment_cm']==12.
    assert receipt['protocol_id'].endswith('.v2')
    with pytest.raises(ValueError,match='not qualified'):
        require_stationary_vpt2_qualification(receipt)
    request['applied_update_ids']=receipt['applied_update_ids']
    with pytest.raises(ValueError,match='already applied'):
        assemble_reference_consistent_field_full_hessian(request,jet)


@pytest.mark.parametrize('change',["frame","coverage","asymmetric"])
def test_incomplete_or_mismatched_full_hessian_rejected(change):
    request,jet=request_and_jet()
    if change=='frame':request['acquired_jet']['coordinate_sha256']='e'*64
    elif change=='coverage':request['acquired_jet']['coverage']='diagonal_only'
    else:request['acquired_jet']['hessian_cm'][0][1]=4.
    with pytest.raises(ValueError):
        assemble_reference_consistent_field_full_hessian(request,jet)
