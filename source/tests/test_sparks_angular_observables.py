import numpy as np
from matrix_sparks import photoelectron_angular_parameters
from matrix_sparks import fixed_photon_photoelectron_lines
from matrix_sparks.continuum import RealSphericalChannels


def test_atomic_s_initial_has_beta_two():
    channels=RealSphericalChannels([(1,1),(1,-1),(1,0)])
    result=photoelectron_angular_parameters(np.eye(3),channels,
        electron_energy_hartree=.2,residual_charge=1)
    np.testing.assert_allclose(result['beta'],2,atol=3e-14)


def test_s_outgoing_wave_is_isotropic():
    channels=RealSphericalChannels([(0,0)])
    result=photoelectron_angular_parameters([[1+1j],[2],[-.2j]],channels,
        electron_energy_hartree=.2,residual_charge=0)
    np.testing.assert_allclose(result['beta'],0,atol=3e-14)


def test_zero_strength_is_undefined_and_batch_scaling_preserves_beta():
    channels=RealSphericalChannels([(1,1),(1,-1),(1,0)])
    result=photoelectron_angular_parameters(np.stack([np.eye(3),.3j*np.eye(3),np.zeros((3,3))]),channels,
        electron_energy_hartree=.2,residual_charge=1)
    np.testing.assert_allclose(result['beta'][:2],2,atol=3e-14)
    assert np.isnan(result['beta'][2])


def test_fixed_photon_angular_contraction():
    channels=RealSphericalChannels([(1,1),(1,-1),(1,0)])
    result=fixed_photon_photoelectron_lines([.8],[.4],[[1]],
        lambda e: np.broadcast_to(np.eye(3)[...,None],(len(e),3,3,1)),
        angular_channels=channels,residual_charge=1)
    np.testing.assert_allclose(result['beta'],2,atol=3e-14)
