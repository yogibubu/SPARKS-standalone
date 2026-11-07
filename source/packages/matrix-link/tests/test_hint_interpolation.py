from __future__ import annotations

import numpy as np

from matrix_link import (
    HINT_SCHEMA,
    build_hint_seed,
    build_hint_seed_or_fallback,
)


def test_hint_seed_from_two_quadratic_endpoint_models() -> None:
    start = np.array([-1.0, 0.25, 0.0])
    end = np.array([1.0, -0.25, 0.0])
    h_start = np.diag([2.0, 1.5, 0.8])
    h_end = np.diag([1.5, 2.0, 0.8])
    seed = build_hint_seed(
        start,
        end,
        h_start,
        h_end,
        coordinate_units="bohr",
        hessian_units="hartree/bohr^2",
        coordinate_mapping=np.arange(3),
        endpoint_labels=("reactant", "product"),
    )

    assert seed.schema == HINT_SCHEMA
    assert seed.mex.converged
    np.testing.assert_allclose(np.linalg.norm(seed.seam_normal), 1.0, atol=1.0e-10)
    eigenvalues = np.linalg.eigvalsh(seed.approximate_hessian)
    assert eigenvalues[0] < -0.09
    assert np.all(np.isfinite(seed.coordinates))
    assert np.all(np.isfinite(seed.approximate_hessian))
    assert seed.provenance["model"] == "endpoint_quadratic_MEX"
    assert seed.provenance["endpoint_labels"] == ("reactant", "product")
    assert seed.provenance["coordinate_mapping"] == [0, 1, 2]
    assert len(seed.provenance["start_hessian_sha256"]) == 64


def test_hint_route_falls_back_without_fabricating_a_seed() -> None:
    fallback = object()
    route = build_hint_seed_or_fallback(
        np.zeros(3),
        np.ones(3),
        np.eye(3),
        np.eye(3),
        coordinate_units="angstrom",
        hessian_units="hartree/bohr^2",
        fallback=lambda error: (fallback, str(error)),
    )

    assert not route.used_hint
    assert route.seed is None
    assert route.fallback_result[0] is fallback
    assert "incompatible" in route.failure_reason
