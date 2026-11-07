"""Improper operations must preserve strengths and retain odd HT amplitudes."""
import numpy as np
import pytest
from matrix_sparks.continuum.angular import RealSphericalChannels


def test_reflection_amplitude_sign_and_involution():
    channels = RealSphericalChannels([(l,m) for l in range(4) for m in range(-l,l+1)])
    reflection = np.diag([1.,1.,-1.])
    values = np.random.default_rng(19).normal(size=(2,3,len(channels.labels)))+1j
    result = channels.transform_dipole_amplitudes(values, reflection, dyson_character=-1)
    parity = (-1.)**(channels.labels[:,0]+abs(channels.labels[:,1]))
    np.testing.assert_allclose(result, -np.diag(reflection)[None,:,None]*values*parity,
                               rtol=0, atol=2e-14)
    np.testing.assert_allclose(channels.transform_dipole_amplitudes(result, reflection,
        dyson_character=-1), values, rtol=0, atol=3e-14)
    np.testing.assert_allclose(np.sum(abs(result)**2,axis=(1,2)),
                               np.sum(abs(values)**2,axis=(1,2)),rtol=1e-14)
    assert np.linalg.norm((values-result)/.1) > 0


def test_incomplete_rotated_channel_space_is_rejected():
    channels = RealSphericalChannels([(1,0)])
    with pytest.raises(ValueError,match='not closed'):
        channels.transform_dipole_amplitudes(np.ones((3,1)),
            np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]),dyson_character=1)
