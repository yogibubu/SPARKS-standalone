"""Check a saved normal-coordinate electron-propagator stencil before scattering.

Uses actual signed Dyson vectors and cross-geometry AO overlaps. Reflection-
related scalar poles/norms are even; continuum amplitudes require their full
Cartesian/channel transformation in compute_sparks_normal_derivatives.py.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from matrix_apoc import (electronic_state_from_gaussian_fchk, read_gdv_ept_dysons,
                         reorient_dyson_orbital, follow_dyson_orbital)
from matrix_sparks.execution import require_compute_host


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--frozen-core', type=int, required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    host = require_compute_host()
    plan = json.loads(a.plan.read_text())

    def read(relative, coordinates):
        folder = (a.root/relative).parent
        state = electronic_state_from_gaussian_fchk(folder/'job.fchk')
        candidates = read_gdv_ept_dysons(folder/'dyson.log', state,
                                       method=a.method, frozen_core=a.frozen_core)
        return [reorient_dyson_orbital(d, coordinates)[0] for d in candidates]

    (reference,) = read(plan['center_inputs']['dyson'], plan['center_coordinates_bohr'])
    rows = {}
    for point in plan['points']:
        if 'symmetry_source' in point:
            continue
        candidates = read(point['inputs']['dyson'], point['coordinates_bohr'])
        dyson, match = follow_dyson_orbital(reference, candidates)
        rows[point['name']] = dict(pole_ev=dyson.ionization_energy_ev,
            pole_strength=dyson.pole_strength, ao_norm_error=abs(dyson.norm_squared-dyson.pole_strength),
            phase=match.phase, normalized_overlap=match.normalized_overlap)
    for point in plan['points']:
        if 'symmetry_source' in point:
            rows[point['name']] = dict(rows[point['symmetry_source']],
                                      scalar_symmetry_source=point['symmetry_source'])
    modes = []
    for mode in range(1,len(plan['frequencies_cm'])+1):
        points = [row for row in plan['points'] if row['mode']==mode and abs(row['q'])==plan['step']]
        negative, positive = sorted(points,key=lambda row:row['q'])
        minus, plus = rows[negative['name']], rows[positive['name']]
        modes.append(dict(mode=mode, frequency_cm=plan['frequencies_cm'][mode-1],
            dIP_ev_dq=(plus['pole_ev']-minus['pole_ev'])/(2*plan['step']),
            dP_dq=(plus['pole_strength']-minus['pole_strength'])/(2*plan['step']),
            reflection_odd=mode in plan.get('reflection_odd_modes',[])))
    result=dict(host=host, reference_pole_ev=reference.ionization_energy_ev,
        reference_pole_strength=reference.pole_strength,
        minimum_normalized_dyson_overlap=min(row['normalized_overlap'] for row in rows.values()),
        maximum_ao_norm_error=max(row['ao_norm_error'] for row in rows.values()),
        independent_points=len(rows)-sum('scalar_symmetry_source' in row for row in rows.values())+1,
        symmetry_reconstructed_points=sum('scalar_symmetry_source' in row for row in rows.values()),
        coordinate_units='dimensionless', step=plan['step'],
        finite_step_convergence_tested=False, electronic_jobs_launched=0,
        modes=modes, points=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('modes','points')},indent=2))


if __name__=='__main__':
    main()
