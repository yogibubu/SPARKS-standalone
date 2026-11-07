"""Prepare, without applying, a same-reference harmonic/anharmonic field update.

The low-level model is a Cartesian quadratic plus its declared normal cubic
and quartic derivatives. A fixed proper rotation aligns its reference once;
the subsequent Cartesian embedding is affine. No geometry-dependent Eckart
rotation, stationarity assumption or electronic acquisition is introduced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from matrix_sparks.execution import require_compute_host
from matrix_trinity.vibronic_gaussian import read_gaussian_vibronic_data, align_gaussian_vibronic_data
from matrix_trinity.reference_consistent_composite import pullback_potential_jet, tensor_fingerprint
from matrix_rovib.vibin import modes_from_hessian
from matrix_gaussian.fchk import BOHR_TO_ANGSTROM
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM
from matrix_gaussian.normal_qff import CM_PER_HARTREE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--harmonic-checkpoint', type=Path, required=True)
    p.add_argument('--source-checkpoint', type=Path, required=True)
    p.add_argument('--source-field', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--assemble', action='store_true', help='Apply the approved complete-Hessian v2 protocol')
    p.add_argument('--method')
    p.add_argument('--basis')
    a = p.parse_args()
    host = require_compute_host()
    if a.output.exists() or a.output.with_suffix('.npz').exists():
        raise FileExistsError(a.output)
    high = read_gaussian_vibronic_data(a.harmonic_checkpoint)
    low = align_gaussian_vibronic_data(read_gaussian_vibronic_data(a.source_checkpoint),high)
    field = np.load(a.source_field)
    original = read_gaussian_vibronic_data(a.source_checkpoint)
    if (not np.array_equal(field['masses_amu'],high['masses_amu'])
            or not np.allclose(field['coordinates_angstrom'],
                original['coordinates_bohr']*BOHR_TO_ANGSTROM,atol=1e-7,rtol=0)):
        raise ValueError('Source field geometry or isotopic masses differ from its checkpoint')
    masses = high['masses_amu']; root_mass = np.sqrt(np.repeat(masses,3))
    w, modes = modes_from_hessian(masses,
        # The normal-mode owner accepts the lower-triangular Cartesian array.
        high['hessian_hartree_per_bohr2'][np.tril_indices(3*len(masses))],
        high['coordinates_bohr']*BOHR_TO_ANGSTROM)
    if np.any(w<=0):
        raise ValueError('Positive high-level harmonic reference required')
    rotation = np.asarray(low['alignment']['rotation'])
    old = (field['modes_mw'].reshape(len(w),len(masses),3) @ rotation).reshape(len(w),-1)
    displacement = (high['coordinates_bohr']-low['coordinates_bohr']).ravel()
    length = HESSIAN_EIGENVALUE_TO_CM/np.sqrt(w*CM_PER_HARTREE)
    embedding = modes.reshape(len(w),-1).T*length[None,:]/root_mass[:,None]
    shift = old @ (root_mass*displacement)
    jacobian = (old*root_mass[None,:]) @ embedding
    f3, f4 = field['cubic_qmw'], field['quartic_qmw']
    # Translate the complete declared finite polynomial, including all slopes.
    nonlinear = (
        np.einsum('ijk,i,j,k',f3,shift,shift,shift)/6 +
        np.einsum('ijkl,i,j,k,l',f4,shift,shift,shift,shift)/24,
        np.einsum('ijk,j,k->i',f3,shift,shift)/2 +
        np.einsum('ijkl,j,k,l->i',f4,shift,shift,shift)/6,
        np.einsum('ijk,k->ij',f3,shift)+np.einsum('ijkl,k,l->ij',f4,shift,shift)/2,
        f3+np.einsum('ijkl,l->ijk',f4,shift), f4)
    n=len(w)
    transported=list(pullback_potential_jet(nonlinear,
        (jacobian, np.zeros((n,n,n)),np.zeros((n,n,n,n)),np.zeros((n,)*5))))
    gradient=low['gradient_hartree_per_bohr']; hessian=low['hessian_hartree_per_bohr2']
    transported[0] += low['energy_hartree']+gradient@displacement+displacement@hessian@displacement/2
    transported[1] += embedding.T@(gradient+hessian@displacement)
    transported[2] += embedding.T@hessian@embedding
    source=tuple(CM_PER_HARTREE*x for x in transported)
    target_gradient=CM_PER_HARTREE*embedding.T@high['gradient_hartree_per_bohr']
    target_hessian=CM_PER_HARTREE*embedding.T@high['hessian_hartree_per_bohr2']@embedding
    metric=embedding.T@(root_mass[:,None]**2*embedding)
    metric_error=float(np.max(abs(metric-np.diag(length**2))))
    if metric_error>1e-10 or not np.allclose(target_hessian,np.diag(w),atol=1e-5,rtol=1e-9):
        raise ValueError('Fixed Cartesian metric or harmonic curvature covariance failed')
    def polynomial(jet, q):
        from math import factorial
        return sum(float(value) if k==0 else np.einsum(
            'ijkl'[:k]+','+','.join('ijkl'[:k])+'->',value,*([q]*k))/factorial(k)
            for k,value in enumerate(jet))
    transport_error=0.
    for q in np.random.default_rng(47).normal(size=(8,n))*.1:
        delta=displacement+embedding@q
        old_q=old@(root_mass*delta)
        direct=low['energy_hartree']+gradient@delta+delta@hessian@delta/2
        direct+=np.einsum('ijk,i,j,k',f3,old_q,old_q,old_q)/6
        direct+=np.einsum('ijkl,i,j,k,l',f4,old_q,old_q,old_q,old_q)/24
        transport_error=max(transport_error,abs(CM_PER_HARTREE*direct-polynomial(source,q)))
    if transport_error>1e-6:
        raise ValueError('Independent finite-geometry polynomial transport failed')
    report=dict(host=host, modes=n, source_reference='B3LYP Cartesian quadratic plus phased normal cubic/quartic polynomial',
        target_reference='high-level checkpoint geometry and dimensionless normal embedding',
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       (a.harmonic_checkpoint,a.source_checkpoint,a.source_field)},
        parent_tensor_sha256=tensor_fingerprint(source),
        geometry_sha256=hashlib.sha256(high['coordinates_bohr'].tobytes()).hexdigest(),
        coordinate_sha256=hashlib.sha256(embedding.tobytes()).hexdigest(),
        masses_sha256=hashlib.sha256(masses.tobytes()).hexdigest(),
        source_geometry_rmsd_bohr=low['alignment']['mass_weighted_rmsd_bohr'],
        source_gradient_norm_cm=float(np.linalg.norm(source[1])),
        target_gradient_norm_cm=float(np.linalg.norm(target_gradient)),
        maximum_offdiagonal_update_cm=float(np.max(abs((target_hessian-source[2])-np.diag(np.diag(target_hessian-source[2]))))),
        constant_mass_metric_max_error=metric_error, scalar_kinetic_correction_cm=0.,
        independent_finite_geometry_energy_max_error_cm=transport_error,
        higher_coordinate_derivatives='exactly zero for this fixed affine Cartesian model',
        linear_forces_retained=True, joint_transport='constant metric; initial-state operator retained in its own frozen Cartesian frame',
        assembly_applied=False, stationary_vpt2_ready=False,
        required_assembly='full acquired Hessian and gradient; current directional-only owner cannot apply the offdiagonal update',
        electronic_jobs_launched=0)
    result=source
    if a.assemble:
        from matrix_core.protocol_library import load_keymaker_protocol_library
        from matrix_trinity.reference_consistent_composite import assemble_reference_consistent_field_full_hessian
        if not a.method or not a.basis:
            raise ValueError('Explicit acquired electronic method and basis required for assembly')
        reference={key:report[key] for key in ('geometry_sha256','coordinate_sha256','masses_sha256')}
        reference.update(atom_order=high['atomic_numbers'].tolist(),isotopologue=masses.tolist(),
            method=a.method,basis=a.basis,triples_scaling='not applicable',
            normalization='dimensionless_q_G0_diagonal_omega_ref_cm')
        request=dict(schema='matrix.trinity.reference_consistent_composite_request.v2',
            protocol_sha256=load_keymaker_protocol_library().resolve('trinity-reference-consistent-composite-field.v2').sha256,
            parent_tensor_sha256=tensor_fingerprint(source),update_id='complete-hessian-'+high['sha256'],
            applied_update_ids=[],reference=reference,experimental_data_used=False,
            reference_policy='retain_origin_and_all_linear_forces',acquired_jet=dict(
                gradient_cm=target_gradient.tolist(),hessian_cm=target_hessian.tolist(),
                geometry_sha256=reference['geometry_sha256'],coordinate_sha256=reference['coordinate_sha256'],
                electronic_evidence_sha256=high['sha256'],coverage='complete_analytic_gradient_and_hessian',
                derivative_estimation='complete analytic checkpoint gradient and Hessian projected in the fixed Cartesian embedding'))
        result,receipt=assemble_reference_consistent_field_full_hessian(request,source)
        report.update(assembly_applied=True,required_assembly=None,assembly_receipt=receipt,
                      assembled_tensor_sha256=tensor_fingerprint(result))
    np.savez_compressed(a.output.with_suffix('.npz'), **{'F'+str(k):v for k,v in enumerate(result)},
        target_gradient_cm=target_gradient,target_hessian_cm=target_hessian,
        frequencies_cm=w, embedding_bohr=embedding, mass_metric=metric,
        high_reference_energy_cm=CM_PER_HARTREE*high['energy_hartree'],
        coordinates_bohr=high['coordinates_bohr'], masses_amu=masses)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
