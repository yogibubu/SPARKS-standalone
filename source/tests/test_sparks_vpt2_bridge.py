"""The anharmonic bank must equal direct signed continuum-channel sums."""

import numpy as np
import pytest
from dataclasses import replace
from matrix_sparks import vpt2_photoionization_spectrum
from matrix_trinity import HarmonicVibronicModel, SAMVPT2Field, vpt2_vibronic_ti


@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_vpt2_bank_equals_all_complex_channels(scale):
    rng = np.random.default_rng(8)
    wi = np.array([900.0, 1600.0, 2700.0])
    wf = np.array([1000.0, 1700.0, 2800.0])
    f3 = np.zeros((3,) * 3)
    f4 = np.zeros((3,) * 4)
    for i in range(3):
        f3[(i,) * 3] = 8.0 + i
        f4[(i,) * 4] = 12.0 + i
    fi = SAMVPT2Field(wi, scale * f3, scale * f4)
    ff = SAMVPT2Field(wf, -scale * f3, scale * f4)
    model = HarmonicVibronicModel(
        wi, wf, np.eye(3), [0.2, -0.1, 0.15], 30000.0, [0, 0, 0], np.zeros((3, 3))
    )
    c = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    h = 0.1 * (rng.normal(size=(3, 2, 3)) + 1j * rng.normal(size=(3, 2, 3)))
    targets = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0)]
    bank = vpt2_photoionization_spectrum(
        model,
        fi,
        ff,
        c[None],
        h[None],
        electron_energies_hartree=[0.2],
        final_states=targets,
        workers=1,
    )
    summed = np.zeros_like(bank["strength_orders"][0, 3])
    for channel in range(2):
        active = replace(
            model,
            transition_dipole_au=c[:, channel],
            transition_gradient_dimensionless_au=h[:, channel],
        )
        direct = vpt2_vibronic_ti(active, fi, ff, final_states=targets, workers=1)
        summed += direct["strength_orders_au2"]
    np.testing.assert_allclose(bank["strength_orders"][0, 3], summed, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        bank["strength_orders"][0, :3].sum(axis=0), summed, rtol=2e-12, atol=2e-13
    )
    assert bank["diagnostics"]["operator_blocks"] == 2
    assert not bank["diagnostics"]["target_space_complete"]
