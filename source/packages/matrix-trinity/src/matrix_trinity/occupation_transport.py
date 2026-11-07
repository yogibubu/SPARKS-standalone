"""Transport a diagonal quadratic occupation Hamiltonian between mode frames.

The TS anharmonic normal-form operator is frozen. A canonical Bogoliubov
transformation uses the supplied harmonic frequencies and orthogonal mode
rotation. Its diagonal Fock expectations define a new quadratic occupation
model. This retains induced kinetic terms of that operator; it is neither an
off-TS VPT2 recalculation nor a treatment of off-diagonal resonant coherence.
"""
import numpy as np
from .hdc_polynomial import HDCPolynomialEnergy


def orthogonal_mode_transport(reference_vectors, local_vectors):
    """Closest orthogonal map from local to reference transverse coordinates.

    Frames must use a common mass-weighted Cartesian/Eckart gauge. Singular
    values quantify loss from changing transverse subspaces; the orthogonal
    polar factor is a declared closure, not a proof that leakage vanishes.
    """
    a,b=np.asarray(reference_vectors,float),np.asarray(local_vectors,float)
    if (a.ndim!=2 or a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
            or not np.allclose(a.T@a,np.eye(a.shape[1]),atol=1e-6,rtol=0)
            or not np.allclose(b.T@b,np.eye(b.shape[1]),atol=1e-6,rtol=0)):
        raise ValueError('matching orthonormal frames in a common Cartesian gauge required')
    u,s,vt=np.linalg.svd(a.T@b);r=u@vt
    return r,dict(singular_values=s,maximum_subspace_leakage=float(np.sqrt(max(0.,1-min(s)**2))),
        closure='orthogonal polar transport of the retained transverse subspaces')


def transport_quadratic_occupation_model(reference_model, reference_frequencies_cm,
        local_frequencies_cm, rotation, *, reference_electronic_cm, local_electronic_cm):
    """Freeze the TS anharmonic operator, rotate/squeeze, retain its diagonal.

    HDC falling-factorial X conventions are preserved. Nonzero cubic or
    higher occupation terms require a higher-order transport implementation
    and are explicitly rejected. Absolute ground energies are included once.
    """
    if not isinstance(reference_model,HDCPolynomialEnergy):raise TypeError('native HDC polynomial required')
    if reference_model.y_tensor_cm is not None and np.any(reference_model.y_tensor_cm!=0):
        raise ValueError('nonzero cubic occupations need higher-order operator transport')
    inputs=[reference_frequencies_cm,local_frequencies_cm,rotation]
    if any(np.iscomplexobj(x) for x in inputs):raise ValueError('real mode transport required')
    w0,w,r=[np.asarray(x,float) for x in inputs];m=len(reference_model.fundamentals_cm)
    if (w0.shape!=(m,) or w.shape!=(m,) or r.shape!=(m,m) or min(w0.min(),w.min())<=0
            or not all(np.all(np.isfinite(x)) for x in [w0,w,r])
            or not np.allclose(r.T@r,np.eye(m),rtol=0,atol=1e-8)
            or not np.isfinite(reference_electronic_cm) or not np.isfinite(local_electronic_cm)):
        raise ValueError('positive frequencies, orthogonal rotation and finite electronic origins required')
    scale=np.sqrt(w0[:,None]/w[None,:]);u=.5*r*(scale+1/scale);v=.5*r*(scale-1/scale)
    symplectic_error=max(float(np.max(abs(u@u.T-v@v.T-np.eye(m)))),float(np.max(abs(u@v.T-v@u.T))))
    if symplectic_error>1e-8:raise ValueError('noncanonical frequency/frame transformation')
    h=np.einsum('ia,ib->iab',u,u)+np.einsum('ia,ib->iab',v,v)
    p=np.einsum('ia,ib->iab',u,v)+np.einsum('ia,ib->iab',v,u)
    mean_coeff=u*u+v*v;mean_zero=np.sum(v*v,axis=1)
    x=reference_model.x_matrix_cm;k=x.copy();k[np.diag_indices(m)]*=2
    linear=reference_model.fundamentals_cm-w0-np.diag(x)
    constant=reference_model.zero_point_cm-reference_electronic_cm-.5*w0.sum()
    wh=.5*np.einsum('ij,iab,jba->ab',k,h,h,optimize=True)
    wp=.5*np.einsum('ij,iab,jab->ab',k,p,p,optimize=True)
    np.fill_diagonal(wh,0.)
    wpdiag=np.diag(wp).copy();np.fill_diagonal(wp,0.)
    def energies(states):
        a=np.asarray(states,float);mean=a@mean_coeff.T+mean_zero
        correction=constant+mean@linear+.5*np.einsum('bi,ij,bj->b',mean,k,mean)
        correction+=np.einsum('ba,ac,bc->b',a,wh,a+1,optimize=True)
        correction+=np.einsum('ba,ac,bc->b',a,wp,a,optimize=True)
        correction+=.5*(a@wp.sum(axis=1)+a@wp.sum(axis=0)+wp.sum())
        correction+=.5*(a*a+a+1)@wpdiag
        return local_electronic_cm+(a+.5)@w+correction
    vacuum=np.zeros((1,m),int);one=np.eye(m,dtype=int)
    zero=float(energies(vacuum)[0]);singles=energies(one);over=energies(2*one)
    xnew=np.zeros((m,m));xnew[np.diag_indices(m)]=(over-2*singles+zero)/2
    pairs=[(i,j) for i in range(m) for j in range(i)]
    if pairs:
        states=np.array([one[i]+one[j] for i,j in pairs]);e=energies(states)
        for value,(i,j) in zip(e,pairs):xnew[i,j]=xnew[j,i]=value-singles[i]-singles[j]+zero
    if np.any(singles-zero<=0):
        raise ValueError(f'transported occupation fundamentals nonpositive: minimum={min(singles-zero):.8g} cm-1')
    model=HDCPolynomialEnergy(zero,singles-zero,xnew,coordinate_id=reference_model.coordinate_id+':transported')
    return model,dict(schema='matrix.trinity.quadratic_occupation_transport.v1',
        symplectic_error=symplectic_error,rotation=r,bogoliubov_U=u,bogoliubov_V=v,
        reference_electronic_cm=reference_electronic_cm,local_electronic_cm=local_electronic_cm,
        closure='diagonal Fock expectation of frozen TS quadratic anharmonic normal-form operator',
        new_off_TS_anharmonic_derivatives=False,off_diagonal_coherence_included=False,
        higher_occupation_terms_ignored=False)


def transport_anharmonic_sam_field(reference_field, local_frequencies_cm, rotation):
    """Transport F3/F4/G1/G2 BEFORE perturbative elimination.

    The supplied reference is a native modal SAMVPT2Field. Coordinates and
    momenta use inverse transpose maps, with updated oscillator lengths.
    The local harmonic Hamiltonian is supplied by the path Hessian. Calling
    the native HDC owner afterwards recomputes every frequency denominator.
    """
    from .vibronic_vpt2 import SAMVPT2Field
    if not isinstance(reference_field,SAMVPT2Field):raise TypeError('native modal SAM field required')
    w0=reference_field.frequencies_cm;w=np.asarray(local_frequencies_cm,float);r=np.asarray(rotation,float)
    m=len(w0)
    if (w.shape!=(m,) or r.shape!=(m,m) or np.any(w<=0)
            or not np.all(np.isfinite(w)) or not np.all(np.isfinite(r))
            or not np.allclose(r.T@r,np.eye(m),rtol=0,atol=1e-8)):
        raise ValueError('positive local frequencies and orthogonal modal rotation required')
    q=r*np.sqrt(w0[:,None]/w[None,:]);p=r*np.sqrt(w[None,:]/w0[:,None])
    cubic=np.einsum('ijk,ia,jb,kc->abc',reference_field.cubic_cm,q,q,q,optimize=True)
    quartic=np.einsum('ijkl,ia,jb,kc,ld->abcd',reference_field.quartic_cm,q,q,q,q,optimize=True)
    first=np.einsum('ijk,ia,jb,kc->abc',reference_field.metric_first_cm,p,p,q,optimize=True)
    second=np.einsum('ijkl,ia,jb,kc,ld->abcd',reference_field.metric_second_cm,p,p,q,q,optimize=True)
    result=SAMVPT2Field(w,cubic,quartic,first,second,reference_field.constant_shift_cm,
        reference_field.quartic_coverage,reference_field.basis_id+':path_harmonic_transport')
    return result,dict(canonical_map_error=float(np.max(abs(q.T@p-np.eye(m)))),
        closure='frozen mass-coordinate cubic/quartic/metric field; local Hessian retained',
        perturbative_denominators_recomputed=True,new_anharmonic_observations=False)


def sonic_field_from_modal_sam(field):
    """Represent native modal tensors in a SONIC field for shared owners."""
    from .curvilinear_vpt2 import SonicVPT2Input
    from matrix_gf.harmonic import CM_PER_HARTREE
    from matrix_qm import HESSIAN_EIGENVALUE_TO_CM
    scale=HESSIAN_EIGENVALUE_TO_CM**2/CM_PER_HARTREE
    w=field.frequencies_cm
    return SonicVPT2Input(np.diag(w/CM_PER_HARTREE),field.cubic_cm/CM_PER_HARTREE,
        field.quartic_cm/CM_PER_HARTREE,np.diag(w/scale),
        field.metric_first_cm/scale,field.metric_second_cm/scale,
        field.constant_shift_cm,field.quartic_coverage)


def project_displaced_quartic_field(anchor, local_frequencies_cm, coordinate_map,
                                    momentum_map, displacement):
    """Translate a quartic anchor and project onto a local transverse bath.

    All coordinates are dimensionless normal coordinates. Rectangular maps
    permit a molecular fragment or a saddle (including its reaction axis) to
    contribute to a positive transverse bath. The caller supplies the maps
    and the displacement in the anchor frame; no unstable axis is occupied.
    The local Hessian replaces the translated quadratic block. Translation
    retains every available quartic contribution to the cubic field.
    """
    from .vibronic_vpt2 import SAMVPT2Field
    if not isinstance(anchor,SAMVPT2Field):raise TypeError('native modal anchor required')
    w=np.asarray(local_frequencies_cm,float);q=np.asarray(coordinate_map,float)
    p=np.asarray(momentum_map,float);d=np.asarray(displacement,float)
    n=len(anchor.frequencies_cm);m=len(w)
    if (w.ndim!=1 or q.shape!=(n,m) or p.shape!=(n,m) or d.shape!=(n,)
            or np.any(w<=0) or not all(np.all(np.isfinite(a)) for a in (w,q,p,d))):
        raise ValueError('finite anchor displacement/maps and positive local frequencies required')
    c=anchor.cubic_cm+np.einsum('ijkl,l->ijk',anchor.quartic_cm,d)
    first=np.zeros((n,n,n)) if anchor.metric_first_cm is None else anchor.metric_first_cm.copy()
    second=np.zeros((n,n,n,n)) if anchor.metric_second_cm is None else anchor.metric_second_cm
    first+=np.einsum('ijkl,l->ijk',second,d)
    return SAMVPT2Field(w,
        np.einsum('ijk,ia,jb,kc->abc',c,q,q,q,optimize=True),
        np.einsum('ijkl,ia,jb,kc,ld->abcd',anchor.quartic_cm,q,q,q,q,optimize=True),
        np.einsum('ijk,ia,jb,kc->abc',first,p,p,q,optimize=True),
        np.einsum('ijkl,ia,jb,kc,ld->abcd',second,p,p,q,q,optimize=True),
        anchor.constant_shift_cm,anchor.quartic_coverage,anchor.basis_id+':translated_transverse')


def interpolate_transverse_fields(fields, weights):
    """Combine projected endpoint fields, retaining one common local Hessian.

    Weights must be supplied independently of reference rates. HDCPT2 is
    applied AFTER this interpolation, so its denominators and resonances
    follow the local Hessian. Weight derivatives are not included: this is
    an interpolated local field, not derivatives of a blended potential.
    """
    from .vibronic_vpt2 import SAMVPT2Field
    fields=tuple(fields);a=np.asarray(weights,float)
    if (not fields or a.shape!=(len(fields),) or not np.all(np.isfinite(a))
            or np.any(a<0) or not np.isclose(a.sum(),1.,rtol=0,atol=1e-12)):
        raise ValueError('nonnegative normalized endpoint weights required')
    w=fields[0].frequencies_cm
    if any(not np.array_equal(f.frequencies_cm,w) for f in fields):
        raise ValueError('endpoint projections must share the exact local harmonic block')
    def combine(name):
        values=[getattr(f,name) for f in fields]
        if all(v is None for v in values):return None
        template=next(v for v in values if v is not None)
        return sum(t*(np.zeros_like(template) if v is None else v) for t,v in zip(a,values))
    coverage='complete' if all(f.quartic_coverage=='complete' for f in fields) else 'semidiagonal_model'
    return SAMVPT2Field(w,combine('cubic_cm'),combine('quartic_cm'),
        combine('metric_first_cm'),combine('metric_second_cm'),
        float(sum(t*f.constant_shift_cm for t,f in zip(a,fields))),coverage,
        ':'.join(f.basis_id for f in fields)+':endpoint_interpolation')


def sum_independent_fragment_fields(fields):
    """Sum projected fragment corrections, retaining ONE local harmonic bath.

    Each fragment is already projected into the same actual local bath.
    This sums anharmonic tensors and scalar corrections, never the harmonic
    frequencies. It assumes independent anchor fragments, not independent
    modes of the final interacting local field.
    """
    from .vibronic_vpt2 import SAMVPT2Field
    fields=tuple(fields)
    if not fields:raise ValueError('at least one projected fragment field required')
    w=fields[0].frequencies_cm
    if any(not np.array_equal(f.frequencies_cm,w) for f in fields):
        raise ValueError('fragment projections must share the exact local harmonic block')
    def add(name):
        arrays=[getattr(f,name) for f in fields];nonempty=[a for a in arrays if a is not None]
        return sum(nonempty) if nonempty else None
    return SAMVPT2Field(w,add('cubic_cm'),add('quartic_cm'),add('metric_first_cm'),
        add('metric_second_cm'),sum(f.constant_shift_cm for f in fields),
        'complete' if all(f.quartic_coverage=='complete' for f in fields) else 'semidiagonal_model',
        ':'.join(f.basis_id for f in fields)+':independent_anchor_fragments')


def modal_sam_in_physical_frame(field, source_displacements, masses,
                                 target_vectors, target_frequencies_cm):
    """Lock modal tensors and Cartesian columns to ONE eigenvector gauge.

    Source displacements are dx/dz in Angstrom for the coordinates of the
    supplied SONIC field. Modalization is called once; its returned modes
    build the physical frame of those very tensors. Signs, permutations and
    exact degeneracy rotations are reconciled with the requested frame.
    Frequencies alone are never used to identify modal phases.
    """
    from .curvilinear_vpt2 import _modalize_sonic_field
    from .vibronic_vpt2 import SAMVPT2Field
    w,modes,c,q,g1,g2=_modalize_sonic_field(field)
    mass=np.asarray(masses,float);source=np.asarray(source_displacements,float)
    target=np.asarray(target_vectors,float);local_w=np.asarray(target_frequencies_cm,float)
    if (source.shape!=(3*len(mass),len(w)) or target.shape!=source.shape
            or local_w.shape!=w.shape or np.any(mass<=0)
            or not all(np.all(np.isfinite(a)) for a in (mass,source,target,local_w))):
        raise ValueError('matching finite physical displacements, masses and target modal frame required')
    physical=np.repeat(np.sqrt(mass),3)[:,None]*(source@modes)
    physical/=np.linalg.norm(physical,axis=0)
    rotation=physical.T@target
    if (not np.allclose(rotation.T@rotation,np.eye(len(w)),atol=1e-7,rtol=0)
            or not np.allclose(rotation.T@np.diag(w*w)@rotation,np.diag(local_w*local_w),atol=1e-3,rtol=1e-7)):
        raise ValueError('target frame does not represent the same harmonic bath')
    sam=SAMVPT2Field(w,c,q,g1,g2,field.pseudopotential_cm,field.quartic_coverage,'native_modal_gauge')
    transported,metadata=transport_anharmonic_sam_field(sam,local_w,rotation)
    metadata.update(physical_mode_overlap=rotation,shared_modalization_for_tensors_and_frame=True)
    return transported,metadata
