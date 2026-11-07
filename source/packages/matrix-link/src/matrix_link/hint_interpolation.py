"""Endpoint-Hessian interpolation seed for LINK.

This module is a small, backend-neutral bridge inspired by HINT (Tesselli and
Jensen, JCTC 2026). It stops at seed construction: the returned geometry and
approximate Hessian are consumed by the existing LINK optimizer.

The endpoint model is the quadratic local model

    E_i(q) = E_i^0 + 1/2 (q-q_i)^T H_i (q-q_i),

with one mapped Cartesian coordinate ordering at both endpoints. The existing
MEX solver minimizes the average model on the equal-energy seam. The seam
normal is then assigned a controlled negative curvature while the orthogonal
curvature from the MEX Lagrangian is retained.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping, Sequence

import numpy as np

from .mex import MEXResult, MEXSurfaceEvaluation, optimize_mex


HINT_SCHEMA = "matrix.link.hint_interpolation_seed.v1"


@dataclass(frozen=True)
class HINTSeed:
    """A geometry/Hessian pair ready for ``optimize_geometry``."""

    coordinates: np.ndarray
    approximate_hessian: np.ndarray
    mex: MEXResult
    seam_normal: np.ndarray
    negative_curvature: float
    provenance: Mapping[str, object]
    schema: str = HINT_SCHEMA

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float).reshape(-1)
        hessian = np.asarray(self.approximate_hessian, dtype=float)
        normal = np.asarray(self.seam_normal, dtype=float).reshape(-1)
        if coordinates.size == 0 or hessian.shape != (coordinates.size, coordinates.size):
            raise ValueError("HINT seed geometry and Hessian have incompatible shapes")
        if normal.shape != coordinates.shape or not np.all(np.isfinite(coordinates)):
            raise ValueError("HINT seed contains an invalid seam normal or geometry")
        if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(normal)):
            raise ValueError("HINT seed contains non-finite data")
        if not np.isfinite(float(self.negative_curvature)) or float(self.negative_curvature) >= 0.0:
            raise ValueError("HINT negative_curvature must be finite and negative")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("HINT provenance must be a mapping")
        object.__setattr__(self, "coordinates", coordinates.copy())
        object.__setattr__(self, "approximate_hessian", 0.5 * (hessian + hessian.T))
        object.__setattr__(self, "seam_normal", normal.copy())
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class HINTRouteResult:
    """Result of HINT with an explicit fail-closed fallback route."""

    used_hint: bool
    seed: HINTSeed | None
    fallback_result: object | None
    failure_reason: str | None = None


def build_hint_seed(
    start_coordinates: Sequence[float] | np.ndarray,
    end_coordinates: Sequence[float] | np.ndarray,
    start_hessian: Sequence[Sequence[float]] | np.ndarray,
    end_hessian: Sequence[Sequence[float]] | np.ndarray,
    *,
    start_energy_hartree: float = 0.0,
    end_energy_hartree: float = 0.0,
    negative_curvature: float = -0.1,
    max_iterations: int = 100,
    gradient_tolerance: float = 1.0e-7,
    gap_tolerance: float = 1.0e-8,
    step_tolerance: float = 1.0e-8,
    coordinate_units: str = "bohr",
    hessian_units: str = "hartree/bohr^2",
    coordinate_mapping: Sequence[int] | None = None,
    endpoint_labels: Sequence[str] = (),
) -> HINTSeed:
    """Construct a HINT-style endpoint seed without electronic-structure calls.

    Coordinates and Hessians must use one mapped Cartesian ordering and
    consistent units. Callers can pass analytic Gaussian endpoint Hessians,
    then hand ``coordinates`` and ``approximate_hessian`` to LINK. The
    explicit mapping, units, and array hashes are retained in provenance.
    """

    q_start = np.asarray(start_coordinates, dtype=float).reshape(-1)
    q_end = np.asarray(end_coordinates, dtype=float).reshape(-1)
    h_start = np.asarray(start_hessian, dtype=float)
    h_end = np.asarray(end_hessian, dtype=float)
    if q_start.size == 0 or q_start.shape != q_end.shape:
        raise ValueError("HINT endpoints must have the same non-empty shape")
    if h_start.shape != (q_start.size, q_start.size) or h_end.shape != h_start.shape:
        raise ValueError("HINT endpoint Hessians must match the endpoint dimension")
    if not np.all(np.isfinite(q_start)) or not np.all(np.isfinite(q_end)):
        raise ValueError("HINT endpoint coordinates must be finite")
    if not np.all(np.isfinite(h_start)) or not np.all(np.isfinite(h_end)):
        raise ValueError("HINT endpoint Hessians must be finite")
    if not np.isfinite(start_energy_hartree) or not np.isfinite(end_energy_hartree):
        raise ValueError("HINT endpoint energies must be finite")
    if negative_curvature >= 0.0 or not np.isfinite(negative_curvature):
        raise ValueError("HINT negative_curvature must be finite and negative")
    coordinate_units = str(coordinate_units).strip().lower()
    hessian_units = str(hessian_units).strip().lower()
    expected_hessian_units = {
        "bohr": "hartree/bohr^2",
        "angstrom": "hartree/angstrom^2",
    }
    if coordinate_units not in expected_hessian_units:
        raise ValueError("HINT coordinate_units must be 'bohr' or 'angstrom'")
    if hessian_units != expected_hessian_units[coordinate_units]:
        raise ValueError("HINT coordinate and Hessian units are incompatible")
    mapping = (
        np.arange(q_start.size, dtype=int)
        if coordinate_mapping is None
        else np.asarray(coordinate_mapping, dtype=int).reshape(-1)
    )
    if mapping.shape != q_start.shape or not np.array_equal(np.sort(mapping), np.arange(q_start.size)):
        raise ValueError("HINT coordinate_mapping must be a complete Cartesian permutation")
    labels = tuple(str(label) for label in endpoint_labels)
    if labels and len(labels) != 2:
        raise ValueError("HINT endpoint_labels must contain start and end labels")

    h_start = 0.5 * (h_start + h_start.T)
    h_end = 0.5 * (h_end + h_end.T)

    def surface(q: np.ndarray, center: np.ndarray, hessian: np.ndarray, energy: float):
        displacement = np.asarray(q, dtype=float) - center
        gradient = hessian @ displacement
        value = float(energy + 0.5 * displacement @ gradient)
        return MEXSurfaceEvaluation(value, gradient, hessian)

    def surface_a(q: np.ndarray) -> MEXSurfaceEvaluation:
        return surface(q, q_start, h_start, float(start_energy_hartree))

    def surface_b(q: np.ndarray) -> MEXSurfaceEvaluation:
        return surface(q, q_end, h_end, float(end_energy_hartree))

    mex = optimize_mex(
        surface_a,
        surface_b,
        0.5 * (q_start + q_end),
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
        gap_tolerance=gap_tolerance,
        step_tolerance=step_tolerance,
    )
    first = surface_a(mex.coordinates)
    second = surface_b(mex.coordinates)
    seam = np.asarray(first.gradient - second.gradient, dtype=float)
    seam_norm = float(np.linalg.norm(seam))
    if seam_norm <= 1.0e-14:
        raise RuntimeError("HINT endpoint models do not define a seam normal")
    seam /= seam_norm
    projector = np.eye(q_start.size) - np.outer(seam, seam)
    hessian = projector @ mex.approximate_hessian @ projector
    hessian = 0.5 * (hessian + hessian.T)
    hessian += float(negative_curvature) * np.outer(seam, seam)
    provenance = {
        "schema": HINT_SCHEMA,
        "model": "endpoint_quadratic_MEX",
        "coordinate_units": coordinate_units,
        "hessian_units": hessian_units,
        "coordinate_dimension": int(q_start.size),
        "coordinate_mapping": mapping.tolist(),
        "endpoint_labels": labels,
        "start_coordinates_sha256": _array_sha256(q_start),
        "end_coordinates_sha256": _array_sha256(q_end),
        "start_hessian_sha256": _array_sha256(h_start),
        "end_hessian_sha256": _array_sha256(h_end),
        "mex_iterations": len(mex.iterations),
        "mex_converged": bool(mex.converged),
    }
    return HINTSeed(
        coordinates=mex.coordinates,
        approximate_hessian=hessian,
        mex=mex,
        seam_normal=seam,
        negative_curvature=float(negative_curvature),
        provenance=provenance,
    )


def build_hint_seed_or_fallback(
    *args: object,
    fallback: Callable[[Exception], object],
    **kwargs: object,
) -> HINTRouteResult:
    """Use HINT when admissible and otherwise invoke the existing hierarchy.

    The fallback receives the original exception, so callers can persist the
    rejection reason and route the same endpoint pair into LINK's established
    exploration hierarchy without manufacturing a HINT result.
    """

    try:
        seed = build_hint_seed(*args, **kwargs)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return HINTRouteResult(
            used_hint=False,
            seed=None,
            fallback_result=fallback(error),
            failure_reason=f"{type(error).__name__}: {error}",
        )
    return HINTRouteResult(used_hint=True, seed=seed, fallback_result=None)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=float))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


__all__ = [
    "HINT_SCHEMA",
    "HINTRouteResult",
    "HINTSeed",
    "build_hint_seed",
    "build_hint_seed_or_fallback",
]
