from itertools import permutations
from math import factorial
from types import SimpleNamespace

import numpy as np
import pytest

from matrix_gaussian.normal_field_gauge import align_gaussian_normal_field
from matrix_chem import BOHR_TO_ANGSTROM
from matrix_rovib.isotopic_cubic import isotopic_masses_amu,rovibrational_kinematics_from_modes
from matrix_rovib.vibin import modes_from_hessian


@pytest.mark.parametrize('saved_displacements', [False, True])
def test_printed_field_gauge_is_recovered_for_every_supplied_order(tmp_path,monkeypatch,saved_displacements):
    xyz=np.array([[0.,0.,0.],[1.2,0.,0.],[.2,1.1,0.],[.3,.4,1.]])
    masses=isotopic_masses_amu(['C','O','N','H'])
    _,base=modes_from_hessian(masses,np.eye(12)[np.tril_indices(12)],xyz)
    D=base.reshape(6,12)*np.sqrt(np.repeat(masses,3))[None,:]
    h=D.T@np.diag(np.linspace(.01,.06,6))@D
    _,modes=modes_from_hessian(masses,h[np.tril_indices(12)],xyz)
    kin=rovibrational_kinematics_from_modes(xyz,masses,modes)
    diagonal=kin.inertia_derivatives_amu_sqrt_ang[:3].T
    assert np.all(np.max(abs(diagonal),axis=1)>1e-4)
    phases=np.array([-1,1,-1,-1,1,-1]);mapping=(6,5,4,3,2,1)
    lines=['Inertia Moments Derivatives w.r.t. Normal Modes']
    for a,harmonic in enumerate(mapping):
        i=harmonic-1;v=diagonal[i]*phases[i]*(0 if saved_displacements else 1)
        lines.append(f' Q({a+1:5d}) {v[0]:.10f} 0.0 {v[1]:.10f} 0.0 0.0 {v[2]:.10f}')
    log=tmp_path/'input.log';log.write_text('\n'.join(lines))
    fchk=tmp_path/'input.fchk';fchk.write_text('synthetic harmonic source')
    frozen=SimpleNamespace(cartesian_coordinates_bohr=xyz/float(BOHR_TO_ANGSTROM),
                           masses_amu=masses,cartesian_hessian_lower=h[np.tril_indices(12)])
    monkeypatch.setattr('matrix_gaussian.normal_field_gauge.read_gaussian_fchk',lambda path:frozen)
    blocks = ({'Anharmonic displacement flags':np.arange(1,7),
               'Anharmonic displacements':(modes.reshape(6,12)*phases[:,None]/
                                           np.sqrt(np.repeat(masses,3))[None,:]).ravel()}
              if saved_displacements else {})
    monkeypatch.setattr('matrix_gaussian.normal_field_gauge._read_fchk_blocks', lambda path:blocks)
    rng=np.random.default_rng(11);native={};printed={}
    for rank in (3,4):
        x=rng.normal(size=(6,)*rank)
        native[rank]=sum(x.transpose(p) for p in permutations(range(rank)))/factorial(rank)
        x=native[rank].copy()
        for axis in range(rank):
            shape=[1]*rank;shape[axis]=6;x*=phases.reshape(shape)
        printed[rank]=x
    aligned,*_,audit=align_gaussian_normal_field(log,fchk,printed,anharmonic_to_harmonic=mapping)
    assert audit['inertia_phase_anchors']==(0 if saved_displacements else 6)
    assert audit['checkpoint_displacement_phase_anchors']==(6 if saved_displacements else 0)
    for rank in (3,4):np.testing.assert_allclose(aligned[rank],native[rank],atol=1e-12)
    if saved_displacements:
        blocks['Anharmonic displacements'][0] += .1
        with pytest.raises(ValueError, match='differ from harmonic mode directions'):
            align_gaussian_normal_field(log,fchk,printed,anharmonic_to_harmonic=mapping)
