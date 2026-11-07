"""Two-coordinate physical valley fitted to a finite frozen sample tensor.

The longitudinal dependence reuses the shared asymptotic matrix fitter on
potential collocation slices. A bounded Morse coordinate resolves the other
stretch. Fragment functions are fitted independently and remain the exact
limits of the resulting analytic function. No electronic callback exists.
"""
from dataclasses import dataclass
import numpy as np
from .analytic_surface import validate_sample_plan,POLICY_ID,POLICY_SHA256
from .asymptotic_channels import fit_asymptotic_channels


@dataclass(frozen=True)
class AsymptoticValleySurface:
    coordinate_map: object
    coefficients_cm: np.ndarray
    left_coefficients_cm: np.ndarray
    right_coefficients_cm: np.ndarray
    transverse_domain: tuple
    morse_exponent: float

    def transverse_basis(self,v):
        v=np.asarray(v,float);lo,hi=self.transverse_domain
        if not np.all(np.isfinite(v)) or np.any(v<lo-1e-10) or np.any(v>hi+1e-10):
            raise ValueError('transverse DVR nodes must remain inside the frozen sampled domain')
        upper=np.exp(-self.morse_exponent*(hi-lo))
        w=2*(np.exp(-self.morse_exponent*(v-lo))-upper)/(1-upper)-1
        return np.polynomial.chebyshev.chebvander(w,len(self.left_coefficients_cm)-1)

    def evaluate(self,q,v):
        q,v=np.broadcast_arrays(np.asarray(q,float),np.asarray(v,float));shape=q.shape
        switch,basis=self.coordinate_map.design(q.ravel());trans=self.transverse_basis(v.ravel())
        left=trans@self.left_coefficients_cm;right=trans@self.right_coefficients_cm
        result=left+switch*(right-left)+np.einsum('ni,ij,nj->n',basis,self.coefficients_cm,trans)
        return result.reshape(shape)


def fit_asymptotic_valley(q,v,potential_cm,*,left_cm,right_cm,q_validation_indices,
        v_validation_indices,longitudinal_degree,transverse_degree,sample_budget,
        max_validation_error_cm=10.,q_derivatives=None,morse_exponent=1.,gradient_length_scale=.1):
    q=np.asarray(q,float);v=np.asarray(v,float);potential=np.asarray(potential_cm,float)
    validate_sample_plan(sample_count=potential.size+2*len(v),sample_budget=sample_budget)
    if (q.ndim!=1 or v.ndim!=1 or potential.shape!=(len(q),len(v)) or np.any(np.diff(v)<=0)
            or not np.all(np.isfinite(potential)) or not np.isfinite(morse_exponent) or morse_exponent<=0):
        raise ValueError('finite ordered tensor samples and positive Morse exponent required')
    iv=np.asarray(v_validation_indices)
    if iv.ndim!=1 or not len(iv) or iv.dtype.kind not in 'iu' or np.any(iv<=0) or np.any(iv>=len(v)-1) or len(np.unique(iv))!=len(iv):
        raise ValueError('distinct interior transverse held-outs required')
    tv=np.ones(len(v),bool);tv[iv]=False
    if transverse_degree<1 or tv.sum()<=transverse_degree:raise ValueError('insufficient transverse training samples')
    def diagonal(values):
        values=np.asarray(values);return values[...,None]*np.eye(tv.sum())
    matrix,control=fit_asymptotic_channels(q,diagonal(potential[:,tv]),
        left_matrix_cm=np.diag(np.asarray(left_cm)[tv]),right_matrix_cm=np.diag(np.asarray(right_cm)[tv]),
        basis_id='transverse potential collocation; no transverse kinetic operator',degree=longitudinal_degree,
        sample_budget=len(q),validation_indices=q_validation_indices,max_validation_error_cm=max_validation_error_cm,
        slopes=None if q_derivatives is None else diagonal(np.asarray(q_derivatives)[:,tv]),gradient_length_scale=gradient_length_scale)
    dummy=AsymptoticValleySurface(matrix.coordinate_map,np.zeros((longitudinal_degree+1,transverse_degree+1)),
        np.zeros(transverse_degree+1),np.zeros(transverse_degree+1),tuple(v[[0,-1]]),morse_exponent)
    table=dummy.transverse_basis(v)
    coefficients=np.linalg.lstsq(table[tv],np.diagonal(matrix.coefficients_cm,axis1=1,axis2=2).T,rcond=1e-12)[0].T
    left=np.linalg.lstsq(table[tv],np.asarray(left_cm)[tv],rcond=1e-12)[0]
    right=np.linalg.lstsq(table[tv],np.asarray(right_cm)[tv],rcond=1e-12)[0]
    model=AsymptoticValleySurface(matrix.coordinate_map,coefficients,left,right,tuple(v[[0,-1]]),morse_exponent)
    errors=model.evaluate(q[:,None],v[None,:])-potential
    transverse_error=float(np.max(abs(errors[:,iv])))
    asym_error=max(float(np.max(abs(table[iv]@left-np.asarray(left_cm)[iv]))),
                   float(np.max(abs(table[iv]@right-np.asarray(right_cm)[iv]))))
    if max(transverse_error,asym_error)>max_validation_error_cm:
        raise ValueError(f'valley transverse/fragment hold-out error {transverse_error:.6g}/{asym_error:.6g} exceeds {max_validation_error_cm:g}; no acquisition launched')
    return model,dict(longitudinal_control=control,transverse_validation_error_cm=transverse_error,
        fragment_validation_error_cm=asym_error,maximum_final_sample_error_cm=float(np.max(abs(errors))),
        transverse_degree=transverse_degree,longitudinal_degree=longitudinal_degree,
        sample_count=int(potential.size+2*len(v)),sample_budget=sample_budget,
        validation='independent longitudinal geometry and transverse-coordinate gates',
        policy_id=POLICY_ID,policy_sha256=POLICY_SHA256,generated_grid_QM_evaluations=0,fit_validated=True)


def run_asymptotic_valley_request(request_path):
    from pathlib import Path
    import json,hashlib
    path=Path(request_path).resolve();raw=path.read_bytes();r=json.loads(raw)
    if (r.get('schema')!='matrix.dvr.analytic_surface_request.v1' or r.get('family')!='asymptotic_valley'
            or not r.get('state_identity') or r.get('coordinate_units')!='angstrom'
            or r.get('adaptive_qm',False)):
        raise ValueError('typed frozen physical valley request with explicit state and coordinate units required')
    source=Path(r['source_npz']);data=np.load(source)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=r['source_sha256']:raise ValueError('frozen valley source fingerprint differs')
    model,control=fit_asymptotic_valley(data['q_A'],data['v_A'],data['potential_cm'],left_cm=data['left_cm'],right_cm=data['right_cm'],
        q_derivatives=data['q_derivative_cm_per_A'],sample_budget=r['sample_budget'],
        q_validation_indices=r['q_validation_indices'],v_validation_indices=r['v_validation_indices'],
        longitudinal_degree=r['longitudinal_degree'],transverse_degree=r['transverse_degree'],
        max_validation_error_cm=r['max_validation_error_cm'],morse_exponent=r.get('morse_exponent',1.))
    out=(path.parent/r.get('output_directory','valley_fit')).resolve();out.mkdir(parents=True,exist_ok=True)
    artifact=out/'analytic_valley.npz';m=model.coordinate_map
    np.savez_compressed(artifact,coefficients_cm=model.coefficients_cm,left_coefficients_cm=model.left_coefficients_cm,
        right_coefficients_cm=model.right_coefficients_cm,center=m.center,inverse_length=m.inverse_length,asymmetry=m.asymmetry,
        sampled_domain=m.sampled_domain,transverse_domain=model.transverse_domain,morse_exponent=model.morse_exponent)
    result=dict(schema='matrix.dvr.analytic_surface_result.v1',family=r['family'],state_identity=r['state_identity'],**control,
        request_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=r['source_sha256'],artifact=str(artifact),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),status='fit_validated')
    target=out/'analytic_surface_result.json';target.write_text(json.dumps(result,indent=2)+'\n');return target
