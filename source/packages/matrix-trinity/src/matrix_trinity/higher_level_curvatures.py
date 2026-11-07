"""Hash-bound higher-level curvature retrieval; never schedules electronic jobs.

The returned ledger describes diagonal replacements in the declared frozen
frame. It neither mutates a force field nor certifies a molecular spectrum.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

PROTOCOL_ID = "trinity-higher-level-curvatures.v1"
REQUEST_SCHEMA = "matrix.trinity.higher_level_curvature_request.v1"


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def prepare_higher_level_curvature_update(
    request: Mapping[str, Any], *, artifact_root: str | Path = "."
) -> dict[str, Any]:
    """Validate existing sources and prepare target-minus-current replacements.

    Requests contain frozen ``reference`` identity, ``electronic_level``, hashed
    ``artifacts``, a unique ``update_id``, ``applied_update_ids`` and one record
    per selected mode. Input attestations document scientific assignments;
    hashing proves artifact identity, not the truth of those assignments.
    """
    from matrix_core.protocol_library import load_keymaker_protocol_library

    protocol = load_keymaker_protocol_library().resolve(PROTOCOL_ID)
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("Unsupported higher-level curvature request")
    if request.get("protocol_sha256") != protocol.sha256:
        raise ValueError("Frozen protocol fingerprint differs")
    update_id = str(request["update_id"]).strip()
    if not update_id or update_id in request["applied_update_ids"]:
        raise ValueError("Empty or previously applied curvature update")
    root = Path(artifact_root)
    hashes = {}
    for item in request["artifacts"]:
        path = root / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Artifact fingerprint differs: {path}")
        if item["id"] in hashes:
            raise ValueError("Duplicate source artifact identifier")
        hashes[item["id"]] = digest
    ref = request["reference"]
    for key in ["geometry_sha256", "coordinate_sha256", "masses_sha256",
                "parent_field_sha256", "atom_order", "isotopologue", "conformer"]:
        if not ref.get(key):
            raise ValueError(f"Missing frozen reference identity: {key}")
    for key in ["geometry_sha256", "coordinate_sha256", "masses_sha256", "parent_field_sha256"]:
        digest = ref[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid reference fingerprint: {key}")
    if ref["parent_field_sha256"] not in hashes.values():
        raise ValueError("Current parent force field must be a verified artifact")
    if ref.get("normalization") != "dimensionless_q_G0_diagonal_omega_ref_cm":
        raise ValueError("Unsupported frozen-coordinate normalization")
    level = request["electronic_level"]
    for key in ["method", "basis", "core_treatment", "triples_scaling", "relativistic_scope"]:
        if key not in level or level[key] is None:
            raise ValueError(f"Missing electronic level specification: {key}")
    if request.get("experimental_fundamentals_used") is not False:
        raise ValueError("Experimental fundamentals cannot define curvature updates")
    rows = []
    targets, source_modes = set(), set()
    for item in request["modes"]:
        mode = item["mode_id"]
        if mode in targets:
            raise ValueError("Target modes must be unique")
        targets.add(mode)
        if item["source_artifact_id"] not in hashes:
            raise ValueError("Mode source artifact was not verified")
        if item["source_isotopologue"] != ref["isotopologue"] or item["source_conformer"] != ref["conformer"]:
            raise ValueError("Source isotopologue or conformer differs")
        if item["symmetry"] != item["source_symmetry"]:
            raise ValueError("Mode symmetry differs")
        if item.get("near_degenerate") is not False:
            raise ValueError("Near-degenerate subspaces need a separate matrix treatment")
        if not item.get("assignment_evidence"):
            raise ValueError("Mode assignment evidence is required")
        current = _positive(item["current_curvature_cm"], "current curvature")
        omega = _positive(item["reference_frequency_cm"], "reference frequency")
        route = item["route"]
        source_key = (item["source_artifact_id"], str(item["source_mode_id"]))
        if source_key in source_modes:
            raise ValueError("Source-to-target assignment must be one-to-one")
        source_modes.add(source_key)
        gradient_delta = 0.0
        assumption = None
        if route == "literature_harmonic":
            if not all(item.get(k) for k in ["primary_source", "table", "row"]):
                raise ValueError("Primary literature table and row are required")
            if item.get("frequency_kind") != "unscaled_harmonic":
                raise ValueError("Only unscaled harmonic frequencies are admissible")
            high = _positive(item["frequency_cm"], "high-level frequency")
            uncertainty = _finite(item["frequency_uncertainty_cm"], "frequency uncertainty")
            if uncertainty < 0 or uncertainty > _positive(item["maximum_frequency_uncertainty_cm"], "uncertainty bound"):
                raise ValueError("Unqualified harmonic frequency uncertainty")
            assumption = item["mode_shape_policy"]
            if assumption not in ["explicitly_assumed_same_direction", "certified_eigenvector_correspondence"]:
                raise ValueError("Mode-shape correspondence must be explicit")
            if assumption == "certified_eigenvector_correspondence" and not item.get("eigenvector_evidence"):
                raise ValueError("Eigenvector correspondence evidence missing")
            target = high * high / omega
            gradient_policy = "retain_source_gradient_literature_supplies_none"
        elif route == "directional_electronic":
            for key in ["geometry_sha256", "coordinate_sha256"]:
                if item["source_" + key] != ref[key]:
                    raise ValueError("Electronic displacement frame differs")
            steps = [_positive(x, "central step") for x in item["central_steps"]]
            if len(set(steps)) < 2 or not item.get("central_difference_evidence"):
                raise ValueError("At least two central steps and their evidence are required")
            error = _finite(item["curvature_uncertainty_cm"], "curvature uncertainty")
            if error < 0 or error > _positive(item["maximum_curvature_uncertainty_cm"], "curvature uncertainty bound"):
                raise ValueError("Directional curvature is not step-qualified")
            target = _positive(item["curvature_cm"], "directional curvature")
            gradient_delta = _finite(item["gradient_cm"], "high-level gradient") - _finite(item["current_gradient_cm"], "current gradient")
            gradient_policy = "install_gradient_then_recenter_or_bound_residual"
        else:
            raise ValueError("Unknown curvature source route")
        rows.append(dict(mode_id=mode, route=route, target_curvature_cm=target,
                         current_curvature_cm=current, curvature_increment_cm=target-current,
                         gradient_increment_cm=gradient_delta, gradient_policy=gradient_policy,
                         mode_shape_policy=assumption, source_artifact_id=item["source_artifact_id"]))
    if not rows:
        raise ValueError("No selected curvatures")
    payload = dict(request)
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(schema="matrix.trinity.higher_level_curvature_update.v1",
                protocol_id=PROTOCOL_ID, protocol_sha256=protocol.sha256,
                update_id=update_id, request_sha256=signature,
                parent_field_sha256=ref["parent_field_sha256"], artifacts=hashes,
                rows=rows, replacement_rule="target_minus_current_exactly_once",
                application_requires="verify_parent_and_unique_update_then_transport_all_operators_if_recentering",
                qualified_for_replacement_ledger=True, field_application_completed=False,
                variational_profile_qualified=False, spectral_qualification=False,
                electronic_jobs_launched=0)
