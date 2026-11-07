"""Independent TI/TD check of a nuclear FC progression from saved Hessians.

The unit constant operator isolates nuclear overlaps. It is not an electronic
photoionization dipole and supplies no absolute cross section.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from matrix_sparks.execution import require_compute_host
from matrix_trinity.vibronic_gaussian import read_gaussian_vibronic_data, gaussian_cartesian_vibronic_model
from matrix_trinity.vibronic_observables import harmonic_operator_spectrum
from matrix_trinity.vibronic_harmonic import PHASE_PER_CM_FS


def fcht_progression(model, initial, args, host, start):
    """Use resident harmonic TI/TD with the complete linear continuum bank."""
    from matrix_sparks import harmonic_photoionization_spectrum
    from matrix_qm import cartesian_normal_modes_from_hessian, HESSIAN_EIGENVALUE_TO_CM
    from matrix_gf.harmonic import CM_PER_HARTREE
    plan=json.loads(args.derivative_plan.read_text())
    modes=cartesian_normal_modes_from_hessian(initial['hessian_hartree_per_bohr2'],
        initial['masses_amu'],initial['coordinates_bohr'])
    n=len(model.initial_frequencies_cm)
    displacement=(modes.mass_weighted_modes.reshape(n,-1).T /
        np.sqrt(np.repeat(initial['masses_amu'],3))[:,None] *
        (HESSIAN_EIGENVALUE_TO_CM/np.sqrt(CM_PER_HARTREE*model.initial_frequencies_cm))[None,:])
    directions=np.asarray(plan['directions_bohr'])
    transport=np.linalg.lstsq(directions,displacement,rcond=None)[0]
    closure=float(np.linalg.norm(directions@transport-displacement)/np.linalg.norm(displacement))
    if closure>1e-7 or not np.allclose(plan['center_coordinates_bohr'],initial['coordinates_bohr'],atol=1e-7,rtol=0):
        raise ValueError('Continuum derivative geometry or normal-coordinate transport differs')
    with np.load(args.amplitudes/'center.npz') as source:
        constants=source['amplitudes'];energies=source['energies']
    if args.equilibrium_amplitudes:
        with np.load(args.equilibrium_amplitudes/'center.npz') as source:
            replacement=source['amplitudes']
            if replacement.shape!=constants.shape or not np.array_equal(source['energies'],energies):
                raise ValueError('Equilibrium and derivative channel/energy spaces differ')
            constants=replacement
    gradients=np.stack([np.load(args.amplitudes/f'derivative-mode{i+1:02d}.npz')['first'] for i in range(n)],axis=-1)
    gradients=np.einsum('escm,mn->escn',gradients,transport)
    common=dict(model=model,amplitudes=constants,derivatives=gradients,
                electron_energies_hartree=energies,workers=1)
    ti=harmonic_photoionization_spectrum(**common,method='TI',selection='adaptive',
        target_fraction=args.target_fraction,maximum_states=args.maximum_transitions,
        maximum_transitions=args.maximum_transitions,maximum_integrals=4000000,
        maximum_final_quanta=32,checkpoint_path=args.output/'ti-checkpoint.npz')
    td=harmonic_photoionization_spectrum(**common,method='TD',time_points=4097,
        time_step_fs=.25,gaussian_sigma_cm=50.)
    lines=np.asarray(ti['wavenumbers_cm']);strengths=np.asarray(ti['strengths'])
    rules=np.asarray(ti['sum_rules'])[:,3];times=np.asarray(td['times_fs'])
    if np.any(rules<=0):
        raise ValueError('Nonzero total continuum strength required at each requested electron energy')
    reconstructed=np.zeros((len(energies),len(times)),complex)
    for first in range(0,len(lines),1024):
        reconstructed+=strengths[:,3,first:first+1024]@np.exp(
            -1j*PHASE_PER_CM_FS*lines[first:first+1024,None]*times[None,:])
    differences=np.max(abs(reconstructed-np.asarray(td['correlations'])[:,3]),axis=1)
    captured=strengths[:,3].sum(axis=1)/rules
    np.savez_compressed(args.output/'ti.npz',wavenumbers_cm=lines,strengths=strengths,
                        final_quanta=ti['final_quanta'],energies_hartree=energies,sum_rules=ti['sum_rules'])
    np.savez_compressed(args.output/'td.npz',times_fs=times,correlations=td['correlations'],
                        wavenumbers_cm=td['wavenumbers_cm'],densities_per_cm=td['densities_per_cm'])
    result=dict(host=host,seconds=perf_counter()-start,modes=n,origin_cm=model.origin_cm,
        electron_energies_hartree=energies.tolist(),fcht_captured_fractions=captured.tolist(),
        transitions=len(lines),normal_transport_relative_error=closure,
        relative_maximum_TI_TD_correlation_differences=(differences/rules).tolist(),
        positive_fcht_missing_strength_fractions=(1-captured).tolist(),
        ht_to_fc_sum_rule_ratios=[float(row[1]/row[0]) if row[0]>0 else None
                                 for row in np.asarray(ti['sum_rules'])],
        operator='complete supplied complex continuum FC+HT bank, conditional on electron energy',
        coordinates='Cartesian adiabatic Hessian',anharmonic=False,
        equilibrium_electronic_source=str((args.equilibrium_amplitudes or args.amplitudes).resolve()),
        energy_alignment=model.provenance.get('energy_alignment','electronic minimum energy difference'),
        electronic_source=str(args.amplitudes.resolve()))
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--final', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-fraction', type=float, default=.99)
    p.add_argument('--maximum-transitions', type=int, default=200000)
    p.add_argument('--amplitudes',type=Path)
    p.add_argument('--derivative-plan',type=Path)
    p.add_argument('--equilibrium-amplitudes',type=Path,
                   help='Higher-level vertex already phase-aligned to the derivative reference')
    p.add_argument('--vertical-ip-ev',type=float,
                   help='Electronic removal pole at the initial geometry; subtract final harmonic reorganization')
    a = p.parse_args()
    if a.equilibrium_amplitudes and not a.amplitudes:
        raise ValueError('A separate equilibrium vertex requires the continuum derivative bank')
    host = require_compute_host()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    start = perf_counter()
    left, right = [read_gaussian_vibronic_data(path) for path in (a.initial, a.final)]
    unit = dict(left, electronic=dict(transition_dipole_au=[1.,0.,0.],
                                      electric_gradient_cartesian=None))
    model = gaussian_cartesian_vibronic_model(left, right, property_data=unit)
    if a.vertical_ip_ev is not None:
        from dataclasses import replace
        from matrix_gaussian.normal_qff import CM_PER_HARTREE
        if not np.isfinite(a.vertical_ip_ev) or a.vertical_ip_ev<=0:
            raise ValueError('Positive finite vertical removal energy required')
        relaxation=.5*np.dot(model.final_frequencies_cm,model.final_shift_dimensionless**2)
        model=replace(model,electronic_gap_cm=a.vertical_ip_ev/27.211386245988*CM_PER_HARTREE-relaxation,
            provenance=dict(model.provenance,energy_alignment='calculated vertical ionization pole minus harmonic reorganization',
                            vertical_ip_ev=a.vertical_ip_ev,reorganization_energy_cm=float(relaxation)))
    if bool(a.amplitudes) != bool(a.derivative_plan):
        raise ValueError('Continuum amplitudes and their normal-coordinate plan are required together')
    if a.amplitudes:
        return fcht_progression(model,left,a,host,start)
    ti = harmonic_operator_spectrum(model, [1.], np.ones((1,1,1)), method='TI',
        selection='adaptive', target_fraction=a.target_fraction,
        maximum_states=a.maximum_transitions, maximum_transitions=a.maximum_transitions,
        maximum_integrals=2000000, maximum_final_quanta=32, workers=1,
        checkpoint_path=a.output/'ti-checkpoint.npz')
    td = harmonic_operator_spectrum(model, [1.], np.ones((1,1,1)), method='TD',
        time_points=4097, time_step_fs=.25, gaussian_sigma_cm=50., workers=1)
    lines = np.asarray(ti['wavenumbers_cm'])
    strengths = np.asarray(ti['strengths'])[0]
    times = np.asarray(td['times_fs'])
    correlation = np.zeros(len(times), complex)
    for first in range(0, len(lines), 1024):
        correlation += strengths[first:first+1024] @ np.exp(
            -1j*PHASE_PER_CM_FS*lines[first:first+1024,None]*times[None,:])
    difference = float(np.max(abs(correlation-np.asarray(td['correlations'])[0])))
    np.savez_compressed(a.output/'ti.npz', wavenumbers_cm=lines, strengths=strengths,
                        final_quanta=ti['final_quanta'])
    np.savez_compressed(a.output/'td.npz', times_fs=times, correlations=td['correlations'],
                        wavenumbers_cm=td['wavenumbers_cm'], densities_per_cm=td['densities_per_cm'])
    report = dict(host=host, seconds=perf_counter()-start, modes=len(model.initial_frequencies_cm),
        initial_sha256=left['sha256'], final_sha256=right['sha256'],
        origin_cm=model.origin_cm, captured_fraction=float(strengths.sum()),
        transitions=len(lines), maximum_TI_TD_correlation_difference=difference,
        missing_strength_bound=float(1-strengths.sum()),
        operator='unit constant nuclear FC; no electronic intensity or cross section',
        coordinates='Cartesian adiabatic Hessian', anharmonic=False)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
