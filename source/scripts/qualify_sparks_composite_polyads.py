"""Refine low-energy dressed polyads of an assembled nonstationary composite.

Reuses the resident target-directed variational/PT2 workspace. This finite
low-energy check retains the complete linear force; it is not a global DVR
potential or a complete anharmonic photoelectron spectrum.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from matrix_sparks.execution import require_compute_host
from matrix_trinity.reference_consistent_composite import tensor_fingerprint
from matrix_trinity.vibronic_vpt2 import SAMVPT2Field
from matrix_trinity.vibronic_gaussian_fields import normal_field_as_sonic
from matrix_trinity.targeted_polyads import CubicDressedWorkspace, refine_targeted_polyads, _encode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--field',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--maximum-states',type=int,default=256)
    p.add_argument('--maximum-iterations',type=int,default=12)
    p.add_argument('--energy-tolerance-cm',type=float,default=.5)
    p.add_argument('--amplitude-tolerance',type=float,default=.03)
    p.add_argument('--selected-vci',action='store_true',help='Undressed VCI with complete cubic/quartic external residuals')
    p.add_argument('--target-window-cm',type=float,default=None)
    p.add_argument('--initial-support-checkpoint',type=Path)
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();host=require_compute_host();start=perf_counter()
    if a.target_window_cm is not None and (not np.isfinite(a.target_window_cm) or a.target_window_cm <= 0):
        raise ValueError('positive finite target window required')
    source=np.load(a.field);report=json.loads(a.receipt.read_text())
    jet=tuple(source['F'+str(k)] for k in range(5))
    if (not report.get('assembly_applied') or report['assembled_tensor_sha256']!=tensor_fingerprint(jet)
            or report['assembly_receipt']['protocol_id']!='trinity-reference-consistent-composite-field.v2'):
        raise ValueError('Qualified complete-Hessian assembly identity required')
    w=source['frequencies_cm'];n=len(w)
    if not np.allclose(jet[2],np.diag(w),atol=1e-5,rtol=1e-9):
        raise ValueError('Composite quadratic must use its declared dimensionless harmonic modes')
    field=normal_field_as_sonic(SAMVPT2Field(w,jet[3],jet[4],quartic_coverage='semidiagonal_model'))
    if a.selected_vci:
        from matrix_trinity.selected_vci import SelectedVCIWorkspace
        workspace=SelectedVCIWorkspace(field,allow_reduced_quartic_model=True,modal_linear_cm=jet[1])
    else:
        workspace=CubicDressedWorkspace(field,allow_reduced_quartic_model=True,modal_linear_cm=jet[1])
    if not np.allclose(workspace.frequencies,w,atol=1e-6,rtol=0):
        raise ValueError('Modal force frame changed during field ingestion')
    target_modes=[i for i in range(n) if a.target_window_cm is None or w[i]<=a.target_window_cm]
    targets=[(0,)*n,*[tuple(np.eye(n,dtype=int)[i]) for i in target_modes]]
    seed=set(targets)
    for state in targets:
        for partner,value in workspace.cubic(state).items():
            gap=float(np.dot(w,np.subtract(state,partner)))
            if abs(gap)<200 and (abs(gap)<1e-6 or abs(value/gap)>.05):
                seed.add(partner)
    if a.initial_support_checkpoint:
        saved=json.loads(a.initial_support_checkpoint.read_text())
        if saved['signature']['identity']['field_sha256'] != report['assembled_tensor_sha256']:
            raise ValueError('Saved support must belong to precisely the same composite field')
        for model in saved['result']['models']:
            seed.update(map(tuple,model['states']))
    if len(seed)>a.maximum_states:
        raise MemoryError('Required resonant seeds exceed the declared variational budget')
    identity=dict(field_sha256=report['assembled_tensor_sha256'],
        receipt_sha256=hashlib.sha256(a.receipt.read_bytes()).hexdigest(),
        linear_forces_retained=True, source_quartic_model='omitted native source terms zero before transport')
    if a.selected_vci or a.target_window_cm is not None:
        identity.update(selected_vci=a.selected_vci,target_window_cm=a.target_window_cm)
    a.output.mkdir(parents=True,exist_ok=True)
    result=refine_targeted_polyads([seed],targets,workspace.matrix,
        workspace.external_corrections if a.selected_vci else workspace.fermi_corrections,
        energy_tolerance_cm=a.energy_tolerance_cm,wavefunction_tolerance=.01,
        external_amplitude_tolerance=a.amplitude_tolerance,
        maximum_states=a.maximum_states,maximum_quanta=6,maximum_iterations=a.maximum_iterations,
        batch_size=a.batch_size,checkpoint_path=a.output/'checkpoint.json',identity=identity,
        dressed_wavefunctions=workspace.lifted_wavefunctions)
    tracked={tuple(row['target']):row for row in result['tracked_states']}
    ground=tracked[(0,)*n]['energy_cm']
    bands=[dict(mode=i+1,harmonic_cm=float(w[i]),
                excitation_cm=float(tracked[tuple(np.eye(n,dtype=int)[i])]['energy_cm']-ground))
           for i in target_modes]
    summary=dict(host=host,elapsed_seconds=perf_counter()-start,converged=result['converged'],
        reason=result['reason'],bands=bands,history=result['history'],
        initial_resonant_seed_states=len(seed),linear_forces_retained=True,
        ground_correction_cm=float(ground),vibrational_ground_energy_cm=float(ground+.5*w.sum()),
        selection_residual=getattr(workspace,'last_residual_diagnostics',None),
        scope=('selected variational states with complete supplied Hamiltonian external residuals; finite window and source-polynomial model'
               if a.selected_vci else 'low-energy dressed variational polyads with external odd-operator PT2; external quartic effects and complete spectrum not certified'))
    (a.output/'result.json').write_text(json.dumps(summary,default=_encode,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('bands','history')},indent=2),flush=True)


if __name__=='__main__':
    main()
