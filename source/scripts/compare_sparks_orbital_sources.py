"""Compare canonical and correlated removal sources in one fixed continuum.

The JSON specification declares existing Gaussian inputs, continuum settings,
energies and spin multiplicity. No electronic calculation is launched.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from matrix_apoc import (electronic_state_from_gaussian_fchk,read_gdv_ept_dysons,
                         reorient_electronic_state)
from matrix_sparks.execution import require_compute_host
from matrix_sparks.continuum import (RealSphericalChannels,LocalPhotoionizationCalculation,
    AODysonOrbital,AODensityGradient,LB94Potential,CoulombTailPotential,
    AtomicSplineBlock,SparseResolventBackend)
from matrix_sparks.photoelectron import fixed_photon_photoelectron_lines


def atomic_blocks_from_settings(settings,coordinates_bohr,atomic_numbers):
    """Shared atomic basis/integration settings for solves and overlap checks."""
    atomic=settings.get('atomic_blocks')
    if atomic is None:
        atoms=()
    else:
        atoms=[]
        for number,point in zip(atomic_numbers,coordinates_bohr):
            block=dict(atomic)
            block.update(atomic.get('element_overrides',{}).get(str(int(number)),{}))
            local=RealSphericalChannels([(l,m) for l in range(block['lmax']+1)
                                        for m in range(-l,l+1)],
                                        **block.get('angular_quadrature',{}))
            atoms.append(AtomicSplineBlock(point,np.linspace(0,block['radius'],block['intervals']+1),local))
        atoms=tuple(atoms)
    return atoms


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();host=require_compute_host()
    if args.output.exists():raise FileExistsError(args.output)
    spec=json.loads(args.specification.read_text());start=perf_counter()
    if spec.get('overlap_qualification'):
        check=json.loads(Path(spec['overlap_qualification']).read_text())
        if check['continuum']!=spec['continuum']:
            raise ValueError('Overlap qualification and continuum settings differ')
        minimum=check['minimum_normalized_schur_eigenvalue']
        if check.get('positive') is False or (minimum is not None and minimum<=0):
            raise ValueError('Overlap qualification failed; continuum propagation is blocked')

    state=electronic_state_from_gaussian_fchk(spec['dyson_checkpoint'])
    candidates=sorted(read_gdv_ept_dysons(spec['dyson_log'],state,method=spec['method'],
                                        frozen_core=spec['frozen_core']),
                      key=lambda item:item.ionization_energy_ev)
    chosen=spec.get('channel_indices',[spec.get('channel_index',0)])
    if not chosen or any(i not in range(len(candidates)) for i in chosen):
        raise ValueError('Removal channel index outside saved poles')
    dysons=[candidates[i] for i in chosen]
    dyson=dysons[0]
    state=dyson.state
    density=electronic_state_from_gaussian_fchk(spec['density_checkpoint'])
    density,_=reorient_electronic_state(density,state.coordinates_bohr)
    references=[]
    for dyson in dysons:
        spin=dyson.spin_channel
        occupied=np.flatnonzero(state.mo_occupations[spin]>0)
        coefficients=state.mo_coefficients[spin][:,occupied]
        overlaps=coefficients.T@state.overlap_ao@dyson.coefficients_ao
        norms=np.einsum('ip,ij,jp->p',coefficients,state.overlap_ao,coefficients)
        selected=int(spec.get('orbital_index',occupied[np.argmax(abs(overlaps)/np.sqrt(norms))]))
        if selected not in occupied:raise ValueError('Canonical reference must be occupied')
        canonical=state.mo_coefficients[spin][:,selected].copy()
        canonical/=np.sqrt(canonical@state.overlap_ao@canonical)
        if canonical@state.overlap_ao@dyson.coefficients_ao<0:canonical*=-1
        references.append((selected,canonical))
    dyson=dysons[0]
    settings=spec['continuum'];lmax=settings['lmax']
    channels=RealSphericalChannels([(l,m) for l in range(lmax+1) for m in range(-l,l+1)],
                                   **settings.get('angular_quadrature',{}))
    cuts=np.linalg.norm(density.coordinates_bohr,axis=1)
    edges=np.unique(np.r_[np.linspace(0,settings['radius'],settings['intervals']+1),cuts])
    xc=CoulombTailPotential(LB94Potential(AODensityGradient(density)),
        inner_radius=settings['xc_tail'][0],outer_radius=settings['xc_tail'][1],charge=1.)
    atoms=atomic_blocks_from_settings(settings,density.coordinates_bohr,density.atomic_numbers)
    calc=LocalPhotoionizationCalculation(density,dyson,channels,edges,xc,
        asymptotic_charge=spec['residual_charge'],outer_radius=settings['outer_radius'],
        dipole_boundary=True,quadrature_breakpoints=cuts,quadrature=settings['quadrature'],atomic_blocks=atoms)
    prepared=perf_counter()
    mode=spec.get('source_mode','batched')
    transform=None
    if mode=='batched':
        coefficients=np.column_stack([v for d,(_,c) in zip(dysons,references)
                                      for v in (d.coefficients_ao,c)])
        source=calc.dipole_sources(AODysonOrbital(state,coefficients))
        transform=np.zeros((6*len(dysons),9*len(dysons)))
        for i,d in enumerate(dysons):
            transform[6*i:6*i+3,9*i:9*i+3]=np.eye(3)
            transform[6*i+3:6*i+6,9*i+3:9*i+6]=np.sqrt(d.pole_strength)*np.eye(3)
            transform[6*i+3:6*i+6,9*i+6:9*i+9]=np.eye(3)
    elif mode=='serial':
        source_blocks=[]
        for dyson,(_,canonical) in zip(dysons,references):
            correlated=calc.dipole_sources(AODysonOrbital.from_apoc(dyson))
            unit_source=calc.dipole_sources(AODysonOrbital(state,canonical))
            source_blocks.extend((correlated,np.sqrt(dyson.pole_strength)*unit_source,unit_source))
        source=np.column_stack(source_blocks)
    else:
        raise ValueError('source_mode must be batched or serial')
    projected=perf_counter()
    solver=SparseResolventBackend(calc.operators.hamiltonian,calc.operators.overlap,calc.operators.boundary,source,
                                 source_transform=transform)
    energies=np.asarray(spec['electron_energies_hartree'],float)
    amplitudes=[];backward=[];diagnostics=[];boundary_seconds=[]
    for energy in energies:
        tick=perf_counter();waves=calc.boundary_waves(energy)
        boundary_seconds.append(perf_counter()-tick)
        amplitudes.append(solver.evaluate_amplitudes(energy,waves))
        backward.append(solver.last_diagnostics['backward_error'])
        diagnostics.append(dict(solver.last_diagnostics))
    amplitudes=np.asarray(amplitudes).reshape(len(energies),len(dysons),3,3,len(channels.labels))
    computed=perf_counter()
    source_benchmark=[]
    for count in spec.get('benchmark_source_counts',[]):
        if mode!='batched' or not 1<=count<=len(dysons):
            raise ValueError('Source benchmark requires batched mode and existing channels')
        subset=SparseResolventBackend(calc.operators.hamiltonian,calc.operators.overlap,
            calc.operators.boundary,source[:,:6*count],source_transform=transform[:6*count,:9*count])
        for repeat in range(spec.get('benchmark_repetitions',3)):
            tick=perf_counter();actual=subset.evaluate_amplitudes(energies[0],waves if len(energies)==1 else calc.boundary_waves(energies[0]))
            elapsed=perf_counter()-tick
            np.testing.assert_allclose(actual,amplitudes[0,:count].reshape(9*count,-1),rtol=1e-9,atol=1e-11)
            source_benchmark.append(dict(channels=count,repeat=repeat,seconds=elapsed,**subset.last_diagnostics))
    mode_benchmark=None
    repeats=spec.get('benchmark_source_modes_repetitions',0)
    if repeats:
        if mode!='batched' or repeats<1:
            raise ValueError('Mode benchmark requires batched mode and positive repetitions')
        tick=perf_counter();serial_blocks=[]
        for d,(_,c) in zip(dysons,references):
            ds=calc.dipole_sources(AODysonOrbital.from_apoc(d))
            cs=calc.dipole_sources(AODysonOrbital(state,c))
            serial_blocks.extend((ds,np.sqrt(d.pole_strength)*cs,cs))
        serial_source=np.column_stack(serial_blocks)
        serial_projection=perf_counter()-tick
        np.testing.assert_allclose(source@transform,serial_source,rtol=1e-10,atol=1e-12)
        old=SparseResolventBackend(calc.operators.hamiltonian,calc.operators.overlap,
                                  calc.operators.boundary,serial_source)
        waves=calc.boundary_waves(energies[0]);measured=[]
        for repeat in range(repeats):
            order=[('serial',old),('batched',solver)]
            if repeat%2:order.reverse()
            for label,owner in order:
                tick=perf_counter();value=owner.evaluate_amplitudes(energies[0],waves)
                seconds=perf_counter()-tick
                expected=amplitudes[0].reshape(9*len(dysons),-1)
                relative=float(np.linalg.norm(value-expected)/max(np.linalg.norm(expected),1e-300))
                np.testing.assert_allclose(value,expected,rtol=1e-9,atol=1e-11)
                measured.append(dict(mode=label,repeat=repeat,seconds=seconds,
                    relative_amplitude_error=relative,**owner.last_diagnostics))
        mode_benchmark=dict(serial_projection_seconds=serial_projection,
            batched_projection_seconds=projected-prepared,rows=measured)
    args.output.mkdir(parents=True)
    reports=[]
    for ion,(dyson,(selected,canonical)) in enumerate(zip(dysons,references)):
        z=dyson.pole_strength

        photon=energies+dyson.ionization_energy_ev/27.211386245988
        results={}
        for index,label in enumerate(['Dyson','canonical_same_norm','canonical_unit_norm']):
            def provider(e,index=index):
                indices=np.array([np.argmin(abs(energies-x)) for x in e])
                if not np.allclose(energies[indices],e,rtol=0,atol=1e-13):raise ValueError('Unexpected energy')
                return amplitudes[indices,ion,index,...,None]
            result=fixed_photon_photoelectron_lines(photon,[dyson.ionization_energy_ev/27.211386245988],
                np.ones((1,1)),provider,channel_multiplicity=spec['channel_multiplicity'],
                angular_channels=channels,residual_charge=spec['residual_charge'])
            results[label]=dict(cross_sections_Mb=result['cross_sections_Mb'][:,3,0].tolist(),beta=result['beta'][:,0].tolist())
        np.testing.assert_allclose(np.asarray(results['canonical_same_norm']['cross_sections_Mb']),
            z*np.asarray(results['canonical_unit_norm']['cross_sections_Mb']),rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(results['canonical_same_norm']['beta'],results['canonical_unit_norm']['beta'],atol=1e-12)
        report=dict(host=host,electronic_jobs=0,elapsed_seconds=perf_counter()-start,
            electron_energies_hartree=energies.tolist(),photon_energies_ev=(photon*27.211386245988).tolist(),
            pole_strength=z,ionization_energy_ev=dyson.ionization_energy_ev,canonical_orbital_one_based=selected+1,channel_index=chosen[ion],
            normalized_dyson_canonical_overlap=float(canonical@state.overlap_ao@dyson.coefficients_ao/np.sqrt(z)),
            hartree_backend=calc.hartree.contraction_backend,results=results,backward_errors=backward,
            overlap_diagnostics=calc.overlap_diagnostics,source_mode=mode,timing_seconds=dict(preparation=prepared-start,source_projection=projected-prepared,
                continuum_and_setup=computed-projected,production_total=computed-start),
            solve_diagnostics=diagnostics,boundary_seconds=boundary_seconds,source_benchmark=source_benchmark,source_mode_benchmark=mode_benchmark,
            continuum=settings,channel_multiplicity=spec['channel_multiplicity'],
            source_sha256={key:hashlib.sha256(Path(spec[key]).read_bytes()).hexdigest()
                           for key in ['dyson_checkpoint','dyson_log','density_checkpoint']},
            scope='Identical local field, boundary, energy and trial space; norm-matched source isolates shape; unit canonical source is a separate diagnostic')
        reports.append(report)
        destination=args.output if len(dysons)==1 else args.output/f"channel-{ion+1:02d}"
        destination.mkdir(exist_ok=True)
        np.savez_compressed(destination/'amplitudes.npz',energies=energies,amplitudes=amplitudes[:,ion],channels=channels.labels)
        (destination/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if len(reports)>1:
        (args.output/'result.json').write_text(json.dumps(dict(host=host,channels=reports,
            elapsed_seconds=perf_counter()-start,electronic_jobs=0),indent=2)+'\n')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
