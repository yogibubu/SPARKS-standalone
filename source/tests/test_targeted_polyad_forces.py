import numpy as np
import pytest
from matrix_trinity.curvilinear_vpt2 import SonicVPT2Input
from matrix_trinity.targeted_polyads import CubicDressedWorkspace
from matrix_gf.harmonic import HESSIAN_EIGENVALUE_TO_CM


def test_retained_linear_force_has_exact_displaced_harmonic_ground_energy():
    frequency=180.
    field=SonicVPT2Input(np.array([[(frequency/HESSIAN_EIGENVALUE_TO_CM)**2]]),
        np.zeros((1,)*3),np.zeros((1,)*4),np.eye(1),np.zeros((1,)*3),np.zeros((1,)*4),0.)
    workspace=CubicDressedWorkspace(field,modal_linear_cm=[12.])
    np.testing.assert_allclose(workspace.matrix([(0,)])[0,0],-12.**2/(2*frequency),atol=1e-12)
    np.testing.assert_allclose(CubicDressedWorkspace(field).matrix([(0,)]),[[0.]],atol=1e-12)
    with pytest.raises(ValueError,match='linear force'):
        CubicDressedWorkspace(field,modal_linear_cm=[np.nan])
