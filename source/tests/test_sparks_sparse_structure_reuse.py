"""Fixed outgoing graph and column ordering preserve repeated energy solves."""
import numpy as np
import pytest
from scipy.sparse import diags
from matrix_sparks.continuum import SparseResolventBackend,BoundaryWaves


@pytest.mark.parametrize('ordering',['NATURAL','MMD_AT_PLUS_A','MMD_ATA','COLAMD'])
@pytest.mark.parametrize('coupled',[False,True])
def test_reused_structure_matches_unrestricted_assembly(ordering,coupled):
    rng=np.random.default_rng(114)
    n,c=53,4
    h=diags([-.1*np.ones(n-1),np.linspace(1.,3.,n),-.1*np.ones(n-1)],[-1,0,1]).tocsc()
    s=diags(np.linspace(.8,1.2,n)).tocsc()
    boundary=np.zeros((n,c));boundary[-6:]=rng.normal(size=(6,c))
    sources=rng.normal(size=(n,3))
    cached=SparseResolventBackend(h,s,boundary,sources,permutation=ordering)
    original=SparseResolventBackend(h,s,boundary,sources,permutation=ordering,reuse_structure=False)
    signature=None
    for index,energy in enumerate([.1,1.,.5,.1]):
        arrays=[rng.normal(size=c) for _ in range(4)]
        if coupled:arrays=[np.diag(a)+.1*rng.normal(size=(c,c)) for a in arrays]
        waves=BoundaryWaves(*arrays)
        expected=original.evaluate(energy,waves)
        actual=cached.evaluate(energy,waves)
        np.testing.assert_allclose(actual.amplitudes,expected.amplitudes,rtol=2e-11,atol=2e-11)
        np.testing.assert_allclose(actual.scattering,expected.scattering,rtol=2e-11,atol=2e-11)
        np.testing.assert_allclose(cached.evaluate_amplitudes(energy,waves),expected.amplitudes,rtol=2e-11,atol=2e-11)
        assert cached.last_diagnostics['structure_reused']
        assert cached.last_diagnostics['ordering_reused']==(ordering!='NATURAL')
        template=cached._structure[0]
        current=(template.indices.tobytes(),template.indptr.tobytes())
        if signature is None:signature=current
        assert signature==current


def test_graph_contains_overlap_entries_missing_from_hamiltonian():
    h=diags(np.arange(1.,13.)).tocsc()
    s=diags([.1*np.ones(10),np.ones(12),.1*np.ones(10)],[-2,0,2]).tocsc()
    b=np.zeros((12,2));b[-2:]=np.eye(2)
    d=np.eye(12)[:,:3]
    waves=BoundaryWaves(np.ones(2),np.ones(2)*2,np.ones(2)*.2,np.ones(2)*.3)
    cached=SparseResolventBackend(h,s,b,d)
    reference=SparseResolventBackend(h,s,b,d,reuse_structure=False)
    for energy in [0.,.2,.4]:
        np.testing.assert_allclose(cached.evaluate_amplitudes(energy,waves),
                                   reference.evaluate_amplitudes(energy,waves),atol=1e-13,rtol=1e-12)


def test_large_boundary_support_keeps_sparse_assembly_fallback():
    n=2001
    solver=SparseResolventBackend(diags(np.ones(n)),diags(np.ones(n)),
                                 np.ones((n,1)),np.ones((n,1)))
    assert solver._structure is None
