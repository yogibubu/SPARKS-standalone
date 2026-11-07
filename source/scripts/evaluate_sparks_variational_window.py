"""FC+HT spectra and observable convergence of saved selected VCI spaces.

All supplied initial/final modes enter the exact Gaussian contraction. Every
final eigenstate of the saved retained space within six linewidths of the
requested window is included. No state-norm repair, energy fit, or physical
completeness claim is made from a comparison of two finite spaces.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from matrix_sparks.execution import require_compute_host
from matrix_sparks.vibronic import continuum_fc_ht_operator_bank
from matrix_sparks.photoelectron import gaussian_line_profiles
from matrix_trinity.targeted_polyads import _decode
from matrix_trinity.selected_vci import SelectedVCIWorkspace
from matrix_trinity.vibronic_vpt2 import SAMVPT2Field
from matrix_trinity.vibronic_gaussian_fields import normal_field_as_sonic
from matrix_trinity.vibronic_gaussian import read_gaussian_vibronic_data, align_gaussian_vibronic_data
from matrix_trinity.vibronic_harmonic import HarmonicVibronicModel
from matrix_trinity.vibronic_variational_observables import finite_basis_operator_spectrum
from matrix_trinity.reference_consistent_composite import tensor_fingerprint
from matrix_trinity.spectral_metrics import spectral_window_convergence
from matrix_gf.harmonic import HESSIAN_EIGENVALUE_TO_CM, CM_PER_HARTREE


def saved_space(checkpoint, field):
    saved=json.loads(checkpoint.read_text(),object_hook=_decode)
    identity=saved['signature']['identity']
    if not identity.get('selected_vci') or identity['field_sha256']!=tensor_fingerprint(tuple(field['F'+str(k)] for k in range(5))):
        raise ValueError('Selected VCI checkpoint and exact composite field identity required')
    tracked=saved['result']['tracked_states'] if saved['finished'] else saved['previous']
    ground=next(row for row in tracked if not any(row['target']))
    basis=np.asarray(ground['states'],int);vector=np.asarray(ground['column'],complex)
    if abs(np.vdot(vector,vector)-1)>1e-8:
        raise ValueError('Normalized retained VCI ground vector required')
    return basis,vector,float(ground['energy_cm']),saved['history'][-1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('initial-field','final-field','initial-checkpoint','final-checkpoint',
                 'initial-gaussian','final-gaussian','amplitudes','equilibrium-amplitudes','derivative-plan','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous',type=Path)
    p.add_argument('--window-upper-cm',type=float,default=1800.)
    p.add_argument('--sigma-cm',type=float,default=50.)
    p.add_argument('--maximum-kernel-entries',type=int,default=1048576)
    p.add_argument('--vertical-ip-ev',type=float)
    p.add_argument('--check-td',action='store_true')
    a=p.parse_args();host=require_compute_host();start=perf_counter()
    if a.output.exists():raise FileExistsError(a.output)
    if not np.isfinite(a.window_upper_cm+a.sigma_cm) or min(a.window_upper_cm,a.sigma_cm)<=0:
        raise ValueError('Positive finite window and linewidth required')
    left,right=np.load(a.initial_field),np.load(a.final_field)
    ib,ui,ei,ih=saved_space(a.initial_checkpoint,left)
    fb,_,ef0,fh=saved_space(a.final_checkpoint,right)
    wi,wf=left['frequencies_cm'],right['frequencies_cm'];n=len(wi)
    di,df=map(read_gaussian_vibronic_data,(a.initial_gaussian,a.final_gaussian))
    for data,field in ((di,left),(df,right)):
        if not np.allclose(data['coordinates_bohr'],field['coordinates_bohr'],atol=1e-8,rtol=0):
            raise ValueError('Composite reference and Gaussian geometry differ')
        if not np.array_equal(data['masses_amu'],field['masses_amu']):
            raise ValueError('Composite isotope masses differ')
    aligned=align_gaussian_vibronic_data(df,di)
    rotation=aligned['alignment']['rotation']
    transform=np.kron(np.eye(len(di['masses_amu'])),rotation.T)
    mass=np.sqrt(np.repeat(di['masses_amu'],3));scale=HESSIAN_EIGENVALUE_TO_CM/np.sqrt(CM_PER_HARTREE)
    initial_modes=mass[:,None]*left['embedding_bohr']*np.sqrt(wi)[None,:]/scale
    final_modes=mass[:,None]*(transform@right['embedding_bohr'])*np.sqrt(wf)[None,:]/scale
    if max(np.max(abs(u.T@u-np.eye(n))) for u in (initial_modes,final_modes))>1e-7:
        raise ValueError('Composite Cartesian modal embeddings must be mass orthonormal')
    j=final_modes.T@initial_modes
    delta=(di['coordinates_bohr']-aligned['coordinates_bohr']).ravel()
    shift=np.sqrt(wf)*(final_modes.T@(mass*delta))/scale
    gap=(df['energy_hartree']-di['energy_hartree'])*CM_PER_HARTREE
    if a.vertical_ip_ev is not None:
        if not np.isfinite(a.vertical_ip_ev) or a.vertical_ip_ev<=0:raise ValueError('Positive calculated vertical pole required')
        gap=a.vertical_ip_ev/27.211386245988*CM_PER_HARTREE-.5*np.dot(wf,shift**2)
    model=HarmonicVibronicModel(wi,wf,j,shift,gap,np.zeros(3),np.zeros((3,n)))
    plan=json.loads(a.derivative_plan.read_text())
    if not np.allclose(plan['center_coordinates_bohr'],di['coordinates_bohr'],atol=1e-7,rtol=0):
        raise ValueError('Continuum derivative reference geometry differs')
    directions=np.asarray(plan['directions_bohr']);transport=np.linalg.lstsq(directions,left['embedding_bohr'],rcond=None)[0]
    closure=np.linalg.norm(directions@transport-left['embedding_bohr'])/np.linalg.norm(left['embedding_bohr'])
    if closure>1e-7:raise ValueError('Continuum derivatives do not span the declared initial modal frame')
    with np.load(a.equilibrium_amplitudes/'center.npz') as source:
        constants=source['amplitudes'];energies=source['energies']
    with np.load(a.amplitudes/'center.npz') as source:
        if not np.array_equal(energies,source['energies']) or constants.shape!=source['amplitudes'].shape:
            raise ValueError('Equilibrium and derivative continuum spaces differ')
    gradients=np.stack([np.load(a.amplitudes/f'derivative-mode{i+1:02d}.npz')['first'] for i in range(n)],axis=-1)
    gradients=np.einsum('escm,mn->escn',gradients,transport)
    banks=[continuum_fc_ht_operator_bank(c,h) for c,h in zip(constants,gradients)]
    field=normal_field_as_sonic(SAMVPT2Field(wf,right['F3'],right['F4'],quartic_coverage='semidiagonal_model'))
    workspace=SelectedVCIWorkspace(field,allow_reduced_quartic_model=True,modal_linear_cm=right['F1'])
    final_h=workspace.matrix(fb)
    eigenvalues,vectors=np.linalg.eigh(final_h)
    if abs(eigenvalues[0]-ef0)>1e-5:
        raise ValueError('Tracked final ground is not the lowest retained eigenstate')
    keep=eigenvalues-eigenvalues[0]<=a.window_upper_cm+6*a.sigma_cm
    output=finite_basis_operator_spectrum(model,ib,ui[:,None],fb,vectors[:,keep],banks[0].constants,
        np.concatenate([b.forms for b in banks]),gradients=banks[0].gradients,
        maximum_grid_entries=128000000,maximum_harmonic_pairs=32000000)
    strengths=output['strengths'][:,:,0].reshape(len(energies),4,-1)
    lines=eigenvalues[keep]-ei
    axis=np.arange(-3*a.sigma_cm,a.window_upper_cm+1.,1.)
    profiles=gaussian_line_profiles(axis,lines,strengths,sigma=a.sigma_cm,
        maximum_kernel_entries=a.maximum_kernel_entries)
    a.output.mkdir(parents=True)
    physical_inputs={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (a.initial_field,a.final_field,a.initial_gaussian,a.final_gaussian,a.derivative_plan,
         a.equilibrium_amplitudes/'center.npz',a.amplitudes/'center.npz',
         *[a.amplitudes/f'derivative-mode{i+1:02d}.npz' for i in range(n)])}
    np.savez_compressed(a.output/'spectrum.npz',initial_basis=ib,axis_relative_harmonic_origin_cm=axis,
        profiles=profiles,lines_relative_harmonic_origin_cm=lines,strengths=strengths,
        nuclear_moments=output['moments'][:,0,:],binding_energies_hartree=(model.origin_cm+lines)/CM_PER_HARTREE,
        electron_energies_hartree=energies,harmonic_origin_cm=model.origin_cm,
        initial_frequencies_cm=wi,final_frequencies_cm=wf,mode_map=j,
        final_shift_dimensionless=shift,electronic_gap_cm=gap,normal_transport=transport,
        final_eigenvalues_cm=eigenvalues[keep],final_basis=fb,final_vectors=vectors[:,keep])
    td_check=None
    if a.check_td:
        from matrix_trinity.quantum_dynamics import QuantumDynamics, PHASE_PER_CM_FS
        sources=output['projected_operator_sources'][:,0,:]
        packets=[]
        for c,h in zip(constants,gradients):
            coefficients=np.column_stack((c.reshape(-1),h.reshape(-1,n)))
            _,r=np.linalg.qr(coefficients.conj(),mode='reduced')
            packets.append(sources@r.conj().T)
        widths=[packet.shape[1] for packet in packets]
        times=np.arange(2049)*.25
        dynamics=QuantumDynamics(final_h,energy_bounds_cm=[float(eigenvalues[0]-1),float(eigenvalues[-1]+1)],
            coordinate_id=tensor_fingerprint(tuple(right['F'+str(k)] for k in range(5))))
        propagated=dynamics.correlate(np.column_stack(packets),times,tolerance=1e-6,initial_energy_cm=ei)
        cuts=np.r_[0,np.cumsum(widths)]
        td=np.array([propagated['values'][:,cuts[k]:cuts[k+1]].sum(axis=1) for k in range(len(packets))])
        all_moments=np.einsum('fi,ia->fa',vectors.conj().T,sources)
        all_strengths=np.array([np.einsum('fa,ab,fb->f',all_moments,b.forms[3],all_moments.conj()).real for b in banks])
        ti=all_strengths@np.exp(-1j*PHASE_PER_CM_FS*(eigenvalues-ei)[:,None]*times[None,:])
        norms=output['retained_source_strengths'][:,0].reshape(len(energies),4)[:,3]
        errors=np.max(abs(td-ti),axis=1)/norms
        td_check=dict(relative_maximum_correlation_error=errors.tolist(),passed=bool(np.all(errors<1e-5)),
            scope='Independent Chebyshev action versus eigenstate sum of the identical complete retained Hamiltonian',
            diagnostics=propagated['diagnostics'])
        np.savez_compressed(a.output/'ti-td.npz',times_fs=times,TI=ti,TD=td)
    comparisons=[]
    if a.previous:
        old=np.load(a.previous/'spectrum.npz')
        old_report=json.loads((a.previous/'result.json').read_text())
        if old_report['physical_inputs_sha256']!=physical_inputs:
            raise ValueError('Observable convergence requires the same physical fields and electronic operator')
        refined=[]
        for name,basis in (('initial',ib),('final',fb)):
            previous_basis=set(map(tuple,old[name+'_basis']))
            current_basis=set(map(tuple,basis))
            if not previous_basis.issubset(current_basis):
                raise ValueError('Observable convergence requires nested retained spaces')
            refined.append(len(current_basis)>len(previous_basis))
        if not any(refined):
            raise ValueError('An unchanged retained space cannot certify convergence')
        if not np.array_equal(old['axis_relative_harmonic_origin_cm'],axis) or not np.array_equal(old['electron_energies_hartree'],energies):
            raise ValueError('Comparison requires identical physical axis and continuum energy bank')
        if abs(float(old['harmonic_origin_cm'])-model.origin_cm)>1e-6:
            raise ValueError('Comparison energy origins differ')
        comparisons=[spectral_window_convergence(axis,ref[3],cur[3]) for ref,cur in zip(old['profiles'],profiles)]
    report=dict(host=host,seconds=perf_counter()-start,initial_states=len(ib),final_states=len(fb),
        initial_iteration=ih['iteration'],final_iteration=fh['iteration'],final_levels=int(keep.sum()),
        modes=n,normal_transport_error=float(closure),sigma_cm=a.sigma_cm,
        relative_window_cm=[float(axis[0]),float(axis[-1])],harmonic_origin_cm=model.origin_cm,
        anharmonic_origin_cm=float(model.origin_cm+eigenvalues[0]-ei),
        ground_corrections_cm=[ei,float(eigenvalues[0])],
        physical_inputs_sha256=physical_inputs,
        refined_surfaces=(dict(zip(('initial','final'),refined)) if a.previous else None),
        ti_td_check=td_check,
        comparisons=comparisons,observable_converged=bool(comparisons) and all(row['passed'] for row in comparisons),
        input_sha256={name:hashlib.sha256(getattr(a,name).read_bytes()).hexdigest() for name in
                      ('initial_field','final_field','initial_checkpoint','final_checkpoint')},
        source_strengths=output['source_strengths'][:,0].reshape(len(energies),4).tolist(),
        retained_source_strengths=output['retained_source_strengths'][:,0].reshape(len(energies),4).tolist(),
        contraction=output['diagnostics'],normalization_changed=False,
        scope='Full retained-space low-energy FC+HT profile; no omitted-space or total molecular spectrum certificate')
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
