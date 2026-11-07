import numpy as np
import pytest
from matrix_trinity.vibronic_harmonic import HarmonicVibronicModel
from matrix_trinity.vibronic_variational import harmonic_overlap_matrix
from matrix_trinity.gaussian_moment_gram import gaussian_overlap_on_lowering_closures


def test_sparse_rectangular_closures_reproduce_full_duschinsky_grid():
    angle = .37
    j = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    model = HarmonicVibronicModel([100.,140.],[110.,135.],j,[.4,-.2],400.,np.zeros(3),np.zeros((3,2)))
    initial = np.array([[0,0],[3,0],[0,2]])
    final = np.array([[0,0],[0,3],[2,0]])
    dense,_ = harmonic_overlap_matrix(model,initial,final)
    sparse,info = harmonic_overlap_matrix(model,initial,final,maximum_grid_entries=50)
    np.testing.assert_allclose(sparse,dense,atol=2e-14)
    assert info['overlap_algorithm'] == 'exact rectangular lowering closures'
    assert info['full_duschinsky_retained']
    with pytest.raises(MemoryError):
        harmonic_overlap_matrix(model,initial,final,maximum_grid_entries=10)


@pytest.mark.parametrize("n", [21, 40])
def test_many_mode_fundamental_overlaps_need_only_lowering_closures(n):
    model=HarmonicVibronicModel(np.arange(n)+100.,np.arange(n)+100.,np.eye(n),np.zeros(n),400.,np.zeros(3),np.zeros((3,n)))
    basis=np.vstack([np.zeros((1,n),int),np.eye(n,dtype=int)])
    overlap,info=harmonic_overlap_matrix(model,basis,basis,maximum_grid_entries=(n+1)**2)
    np.testing.assert_allclose(overlap,np.eye(n+1),atol=1e-13)
    assert info['gaussian_grid_entries']==(n+1)**2
