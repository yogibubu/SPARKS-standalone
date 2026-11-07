"""Shared complex harmonics preserve real channels and Cartesian gradients."""
import numpy as np
import pytest
from matrix_sparks.continuum import angular


@pytest.mark.parametrize('labels', [
    [(l,m) for l in range(5) for m in range(-l,l+1)],
    [(3,-2),(1,0),(3,2),(2,-1),(0,0),(4,-4)],
])
def test_shared_harmonics_preserve_order_poles_and_gradient(labels, monkeypatch):
    channels=angular.RealSphericalChannels(labels)
    points=np.vstack((np.random.default_rng(17).normal(size=(31,3)),
                      [[0,0,2],[0,0,-3]]))
    real_call=angular.sph_harm_y
    calls=[]
    def counted(l,m,*args,**kwargs):
        calls.append((l,m))
        return real_call(l,m,*args,**kwargs)
    monkeypatch.setattr(angular,'sph_harm_y',counted)
    values,gradient=channels.evaluate_with_gradient(points)
    assert len(calls)==len(set((l,abs(m)) for l,m in labels))
    calls.clear()
    np.testing.assert_allclose(channels.evaluate(points),values,atol=3e-15,rtol=3e-14)
    assert len(calls)==len(set((l,abs(m)) for l,m in labels))
    r=np.linalg.norm(points,axis=1)
    theta=np.arccos(points[:,2]/r);phi=np.arctan2(points[:,1],points[:,0])
    expected=[]
    for l,m in labels:
        y=real_call(l,abs(m),theta,phi)
        expected.append(y.real if m==0 else np.sqrt(2)*(-1.)**abs(m)*(y.real if m>0 else y.imag))
    np.testing.assert_array_equal(channels.evaluate(points),np.column_stack(expected))
    step=2e-6
    for axis in range(3):
        delta=np.eye(3)[axis]*step
        fd=(channels.evaluate(points[:-2]+delta)-channels.evaluate(points[:-2]-delta))/(2*step)
        np.testing.assert_allclose(gradient[:-2,:,axis],fd,atol=3e-9,rtol=2e-7)
    assert np.isfinite(gradient).all()
