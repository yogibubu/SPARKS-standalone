"""Resident analytic potentials in the existing LINK geometry optimizer.

Only point evaluation is adapted. Coordinate realization, frozen inactive
coordinates, symmetry, trust-region steps and restart remain owned by LINK.
"""
import numpy as np

from .optimizer import GeometryEvaluationService
from .scan import PointEvaluationResult


class AnalyticPotentialGeometryService(GeometryEvaluationService):
    def __init__(self, *, potential, potential_id, **settings):
        if not callable(potential) or not isinstance(potential_id, str) or not potential_id:
            raise ValueError('resident analytic potential and immutable identity required')
        self.analytic_potential = potential
        self.analytic_potential_id = potential_id
        if settings.get('backend') is not None or settings.get('engine_command'):
            raise ValueError('resident analytic service does not launch a QM backend')
        settings.pop('backend', None); settings.pop('engine_command', None)
        super().__init__(backend=None, engine_command='resident-analytic:'+potential_id, **settings)

    def _evaluate_external_command(self, point, *, tag, requested_properties=()):
        requested = tuple(requested_properties or ('energy', 'gradient'))
        xyz = np.asarray(point.coordinates_angstrom, float)
        hessian = None
        if 'hessian' not in requested and hasattr(self.analytic_potential, 'energy_gradient'):
            energy, gradient = self.analytic_potential.energy_gradient(xyz)
        else:
            result = self.analytic_potential(xyz)
            energy = result.energy_hartree
            gradient = result.gradient_hartree_per_bohr
            if 'hessian' in requested:
                hessian = np.asarray(result.hessian_hartree_per_bohr2, float)
        gradient = np.asarray(gradient, float).reshape(-1)
        if (not np.isfinite(energy) or gradient.shape != (xyz.size,)
                or not np.all(np.isfinite(gradient))):
            raise ValueError('analytic owner returned invalid Cartesian E/G')
        if hessian is not None:
            if hessian.shape != (xyz.size, xyz.size) or not np.all(np.isfinite(hessian)):
                raise ValueError('analytic owner returned invalid Cartesian Hessian')
            hessian = (hessian+hessian.T)/2
        return PointEvaluationResult(point_index=point.index, displacement=point.displacement,
            energy_hartree=float(energy), gradient_hartree_per_bohr=gradient,
            hessian_hartree_per_bohr2=hessian, backend_coordinates_angstrom=xyz,
            source='resident analytic potential', execution=dict(
                potential_id=self.analytic_potential_id, link_evaluation_tag=tag,
                electronic_jobs=0, energy_evaluations=1, gradient_evaluations=1,
                hessian_evaluations=int(hessian is not None)))
