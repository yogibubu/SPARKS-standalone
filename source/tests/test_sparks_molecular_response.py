"""Autonomous molecular response primitives need no electronic executable."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from matrix_sparks.continuum.response import (
    project_adiabatic_kernel, kernel_from_auxiliary_integrals,
)
from matrix_sparks.continuum.molecular_response import GaussianResponseAuxiliary


def test_streamed_kernel_integrals_equal_pointwise_owner():
    rng=np.random.default_rng(72)
    values=rng.normal(size=(35,7));weights=rng.uniform(.1,1.,35)
    fxc=rng.uniform(-.2,0.,35)
    raw=rng.normal(size=(7,7));j=raw@raw.T
    metric=values.T@(weights[:,None]*values)
    xc=values.T@((weights*fxc)[:,None]*values)
    assert_allclose(kernel_from_auxiliary_integrals(metric,j,xc),
                    project_adiabatic_kernel(j,values,weights,fxc),atol=2e-16)
    with pytest.raises(ValueError,match='symmetric'):
        kernel_from_auxiliary_integrals(metric,j+np.tril(np.ones_like(j),-1),xc)


def test_auxiliary_neutrality_and_exact_coulomb_integrals_without_scf():
    from pyscf import gto
    molecule=gto.M(atom='H 0 0 -1; H 0 0 1',unit='Bohr',basis='sto-3g',verbose=0)
    auxiliary=GaussianResponseAuxiliary(molecule,{'H':[[0,[.3,1.]],[0,[1.,1.]],[1,[.3,1.]]]})
    assert auxiliary.size==9
    assert_allclose(auxiliary.charges,0.,atol=2e-14)
    assert np.linalg.eigvalsh(auxiliary.overlap).min()>0
    assert np.linalg.eigvalsh(auxiliary.coulomb).min()>0
    assert auxiliary(np.array([[0.,.2,0.]])).shape==(1,9)
    with pytest.raises(ValueError,match='Explicit auxiliary'):
        GaussianResponseAuxiliary(molecule,None)
