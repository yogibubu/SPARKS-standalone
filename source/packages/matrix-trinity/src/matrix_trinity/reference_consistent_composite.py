"""Reference-consistent composite potential assembly, without electronic jobs.

All jets are ordinary derivatives (not Taylor coefficients) in the same
dimensionless frozen normal coordinates, with energies expressed in cm-1.
Assembly preserves nonzero forces. It does not certify a stationary VPT2
reference, global potential, nuclear Hamiltonian, or spectrum.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np

PROTOCOL_ID = "trinity-reference-consistent-composite-field.v1"
REQUEST_SCHEMA = "matrix.trinity.reference_consistent_composite_request.v1"
FULL_HESSIAN_PROTOCOL_ID = "trinity-reference-consistent-composite-field.v2"
FULL_HESSIAN_REQUEST_SCHEMA = "matrix.trinity.reference_consistent_composite_request.v2"


def tensor_fingerprint(jet):
    """Content identity independent of npz container timestamps/compression."""
    digest = hashlib.sha256()
    for value in jet:
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _jet(values):
    result = tuple(np.asarray(x, dtype=float).copy() for x in values)
    if len(result) != 5 or result[1].ndim != 1:
        raise ValueError("Complete F0..F4 ordinary-derivative jet required")
    n = result[1].size
    for degree, array in enumerate(result):
        if array.shape != (n,) * degree or not np.isfinite(array).all():
            raise ValueError("Invalid potential derivative shape or finite values")
        for axis in range(1, degree):
            if not np.allclose(array, array.swapaxes(0, axis), rtol=1e-10, atol=1e-9):
                raise ValueError("Potential derivatives must be symmetric")
    return result


def assemble_reference_consistent_field(request: Mapping, source_jet):
    """Replace acquired F1/diagonal F2 exactly once; retain other derivatives.

    Slopes estimated from a single +/- pair are retained as estimates, never
    silently promoted to exact gradients. Their numerical qualification is
    separate from assembly. Stationary perturbative consumers must reject this
    output until stationarity and joint operator transport have been qualified.
    """
    from matrix_core.protocol_library import load_keymaker_protocol_library

    protocol = load_keymaker_protocol_library().resolve(PROTOCOL_ID)
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("Unsupported composite field request")
    if request.get("protocol_sha256") != protocol.sha256:
        raise ValueError("Composite protocol fingerprint differs")
    source = _jet(source_jet)
    parent = tensor_fingerprint(source)
    if request.get("parent_tensor_sha256") != parent:
        raise ValueError("Source potential content differs from frozen parent")
    identifier = request.get("update_id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("A nonempty update identity is required")
    if identifier in request["applied_update_ids"]:
        raise ValueError("Composite update already applied")
    reference = request["reference"]
    for key in ("geometry_sha256", "coordinate_sha256", "masses_sha256"):
        value = reference.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Missing frozen reference fingerprint: " + key)
    for key in ("atom_order", "isotopologue", "method", "basis", "triples_scaling"):
        if not reference.get(key):
            raise ValueError("Missing frozen reference specification: " + key)
    if reference.get("normalization") != "dimensionless_q_G0_diagonal_omega_ref_cm":
        raise ValueError("Explicit frozen normal-mode normalization required")
    if request.get("experimental_data_used") is not False:
        raise ValueError("Experimental data cannot define this electronic update")
    if request.get("reference_policy") != "retain_origin_and_all_linear_forces":
        raise ValueError("This protocol forbids discarding forces or silent recentering")
    target = list(source)
    seen = set()
    for row in request["directions"]:
        index = row["mode_index_zero_based"]
        if type(index) is not int or index in seen or not 0 <= index < len(source[1]):
            raise ValueError("Invalid or duplicate acquired normal direction")
        seen.add(index)
        if row.get("coordinate_sha256") != reference["coordinate_sha256"]:
            raise ValueError("Acquisition normal frame differs")
        if row.get("geometry_sha256") != reference["geometry_sha256"]:
            raise ValueError("Acquisition reference geometry differs")
        if not row.get("electronic_evidence_sha256") or not row.get("derivative_estimation"):
            raise ValueError("Electronic evidence and derivative estimation required")
        slope = float(row["target_gradient_cm"])
        curvature = float(row["target_curvature_cm"])
        if not np.isfinite([slope, curvature]).all():
            raise ValueError("Missing finite electronic slope or curvature")
        target[1][index] = slope
        target[2][index, index] = curvature
    if not seen:
        raise ValueError("No acquired directions supplied")
    target = _jet(target)
    receipt = dict(
        schema="matrix.trinity.reference_consistent_composite_result.v1",
        protocol_id=PROTOCOL_ID, protocol_sha256=protocol.sha256,
        update_id=identifier, parent_tensor_sha256=parent,
        result_tensor_sha256=tensor_fingerprint(target),
        reference=dict(reference), replacement_rule="target_minus_current_exactly_once",
        applied_update_ids=[*request["applied_update_ids"], identifier],
        updated_modes_zero_based=sorted(seen),
        gradient_norm_cm=float(np.linalg.norm(target[1])),
        linear_forces_retained=True, unacquired_terms_retained=True,
        source_higher_derivatives_retained=True,
        assembly_completed=True, stationary_vpt2_ready=False,
        spectral_qualification=False, electronic_jobs_launched=0,
        next_gate="Qualify the nonstationary variational Hamiltonian, or recenter with joint transport of potential, metric/scalar and properties before stationary VPT2.",
        derivative_estimation=[dict(row) for row in request["directions"]],
    )
    return target, receipt


def _partitions(items):
    if not items:
        yield ()
        return
    first, *rest = items
    for partition in _partitions(rest):
        yield ((first,),) + partition
        for i in range(len(partition)):
            yield partition[:i] + ((first,) + partition[i],) + partition[i+1:]


def assemble_reference_consistent_field_full_hessian(request: Mapping, source_jet):
    """Install a complete acquired gradient/Hessian in the exact same frame.

    All v1 reference and replay checks remain active. No matrix is inferred
    from a list of frequencies, and no unacquired cross derivative is zeroed.
    Potential assembly retains forces and cannot authorize stationary VPT2.
    """
    from matrix_core.protocol_library import load_keymaker_protocol_library
    library = load_keymaker_protocol_library()
    protocol = library.resolve(FULL_HESSIAN_PROTOCOL_ID)
    if (request.get('schema') != FULL_HESSIAN_REQUEST_SCHEMA
            or request.get('protocol_sha256') != protocol.sha256):
        raise ValueError('Full-Hessian composite schema or protocol fingerprint differs')
    source = _jet(source_jet)
    n = len(source[1])
    acquired = request['acquired_jet']
    gradient = np.asarray(acquired['gradient_cm'],float)
    hessian = np.asarray(acquired['hessian_cm'],float)
    if (gradient.shape != (n,) or hessian.shape != (n,n)
            or not np.isfinite(gradient).all() or not np.isfinite(hessian).all()
            or not np.allclose(hessian,hessian.T,atol=1e-9,rtol=1e-10)):
        raise ValueError('Complete finite gradient and symmetric acquired Hessian required')
    reference = request['reference']
    for key in ('geometry_sha256','coordinate_sha256'):
        if acquired.get(key) != reference.get(key):
            raise ValueError('Full-Hessian acquisition reference differs: '+key)
    if acquired.get('coverage') != 'complete_analytic_gradient_and_hessian':
        raise ValueError('Explicit complete acquired matrix coverage required')
    evidence = acquired.get('electronic_evidence_sha256','')
    if len(evidence)!=64 or any(c not in '0123456789abcdef' for c in evidence):
        raise ValueError('Acquired gradient/Hessian evidence fingerprint required')
    directional = dict(request, schema=REQUEST_SCHEMA,
        protocol_sha256=library.resolve(PROTOCOL_ID).sha256,
        directions=[dict(mode_index_zero_based=i,
            geometry_sha256=reference['geometry_sha256'],coordinate_sha256=reference['coordinate_sha256'],
            electronic_evidence_sha256=evidence,derivative_estimation=acquired['derivative_estimation'],
            target_gradient_cm=float(gradient[i]),target_curvature_cm=float(hessian[i,i]))
            for i in range(n)])
    values, receipt = assemble_reference_consistent_field(directional,source)
    values = _jet((values[0],gradient,hessian,values[3],values[4]))
    receipt.update(schema='matrix.trinity.reference_consistent_composite_result.v2',
        protocol_id=FULL_HESSIAN_PROTOCOL_ID,protocol_sha256=protocol.sha256,
        result_tensor_sha256=tensor_fingerprint(values),complete_hessian_replaced=True,
        acquired_jet_fingerprint=tensor_fingerprint((gradient,hessian)),
        electronic_evidence_sha256=evidence,
        maximum_offdiagonal_increment_cm=float(np.max(abs(
            (hessian-source[2])-np.diag(np.diag(hessian-source[2]))))))
    return values, receipt


def pullback_potential_jet(source_jet, coordinate_derivatives):
    """Exact chain rule through fourth order, including F1 times map derivative 4.

    coordinate_derivatives[r-1] = d^r q_old / d q_new^r at the SAME
    reference. This function does not recenter and does not transform a kinetic
    operator. A recentering must use the complete coordinate-field owner.
    """
    source = _jet(source_jet)
    maps = tuple(np.asarray(x, dtype=float) for x in coordinate_derivatives)
    if len(maps) != 4 or maps[0].ndim != 2:
        raise ValueError("Coordinate derivatives 1..4 are required, even for nonzero F1")
    old, new = maps[0].shape
    for order, value in enumerate(maps, 1):
        if value.shape != (old,) + (new,) * order or not np.isfinite(value).all():
            raise ValueError("Invalid coordinate-map derivative")
    if old != len(source[1]):
        raise ValueError("Coordinate-map input dimension differs")
    result = [source[0].copy()]
    for order in range(1, 5):
        total = np.zeros((new,) * order)
        for partition in _partitions(tuple(range(order))):
            labels = "ABCD"[:len(partition)]
            operands = [source[len(partition)]]
            subscripts = [labels]
            for label, block in zip(labels, partition):
                operands.append(maps[len(block)-1])
                subscripts.append(label + "".join("ijkl"[axis] for axis in block))
            expression = ",".join(subscripts) + "->" + "ijkl"[:order]
            total += np.einsum(expression, *operands, optimize=True)
        result.append(total)
    return _jet(result)


def require_stationary_vpt2_qualification(receipt):
    """Consumer gate: assembly alone cannot authorize stationary perturbation."""
    if receipt.get("stationary_vpt2_ready") is not True:
        raise ValueError("Composite field is not qualified for stationary VPT2; retain forces and qualify the reference/complete Hamiltonian first")
