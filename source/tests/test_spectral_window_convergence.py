import numpy as np
import pytest
from matrix_trinity.spectral_metrics import spectral_window_convergence


def test_observable_test_accepts_unresolved_redistribution_but_rejects_scale_or_shift():
    x=np.linspace(-200,1200,1401)
    gaussian=lambda center:np.exp(-.5*((x-center)/50)**2)
    a=gaussian(0)+.5*gaussian(700)
    b=.99*gaussian(0)+.51*gaussian(701)
    result=spectral_window_convergence(x,a,b)
    assert result['passed']
    assert not result['intensity_rescaled']
    assert not spectral_window_convergence(x,a,1.1*a)['passed']
    assert not spectral_window_convergence(x,a,gaussian(10)+.5*gaussian(710))['passed']
    with pytest.raises(ValueError):spectral_window_convergence(x,a,-a)
