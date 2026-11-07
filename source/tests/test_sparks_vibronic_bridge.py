"""Continuum channel sums must equal the small complex polynomial bank."""

import numpy as np
import pytest
from matrix_sparks import continuum_fc_ht_operator_bank


def test_nuclear_overlap_cache_reuses_states_and_computes_only_missing(tmp_path, monkeypatch):
    from matrix_sparks import cached_harmonic_overlap_matrix
    from matrix_trinity import HarmonicVibronicModel
    from matrix_trinity import vibronic_variational as owner
    model = HarmonicVibronicModel([1000.], [1100.], [[1.]], [.3], 30000.,
                                  [1., 0., 0.], np.zeros((3, 1)))
    initial = np.array([[0], [1]])
    final = np.array([[0], [1], [2], [3]])
    expected, _ = owner.harmonic_overlap_matrix(model, initial, final)
    calls = []
    original = owner.harmonic_overlap_matrix
    def measured(m, i, f, **kw):
        calls.append(f.copy())
        return original(m, i, f, **kw)
    monkeypatch.setattr(owner, 'harmonic_overlap_matrix', measured)
    first, d = cached_harmonic_overlap_matrix(model, initial, final[:2], cache_directory=tmp_path)
    np.testing.assert_allclose(first, expected[:2], rtol=0, atol=0)
    full, d = cached_harmonic_overlap_matrix(model, initial, final, cache_directory=tmp_path)
    np.testing.assert_allclose(full, expected, rtol=0, atol=0)
    assert d['reused_final_states'] == 2 and d['computed_final_states'] == 2
    assert [len(c) for c in calls] == [2, 2]
    again, d = cached_harmonic_overlap_matrix(model, initial, final[[3, 1, 1]], cache_directory=tmp_path)
    np.testing.assert_allclose(again, expected[[3, 1, 1]], rtol=0, atol=0)
    assert len(calls) == 2 and d['computed_final_states'] == 0
    other_operator = HarmonicVibronicModel([1000.], [1100.], [[1.]], [.3], 50000.,
                                          [0., 5., 0.], .5*np.ones((3, 1)))
    reused, same = cached_harmonic_overlap_matrix(other_operator, initial, final,
                                                  cache_directory=tmp_path)
    np.testing.assert_allclose(reused, expected, rtol=0, atol=0)
    assert len(calls) == 2 and same['nuclear_identity'] == d['nuclear_identity']
    shifted = HarmonicVibronicModel([1000.], [1100.], [[1.]], [.4], 30000.,
                                    [1., 0., 0.], np.zeros((3, 1)))
    cached_harmonic_overlap_matrix(shifted, initial, final, cache_directory=tmp_path)
    assert len(calls) == 3
    path = tmp_path / (d['nuclear_identity'] + '.npz')
    with np.load(path) as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays['overlaps'][0, 0] += .1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='altered'):
        cached_harmonic_overlap_matrix(model, initial, final, cache_directory=tmp_path)
    with pytest.raises(MemoryError):
        cached_harmonic_overlap_matrix(model, initial, final, cache_directory=tmp_path,
                                       maximum_harmonic_pairs=1)


def test_complex_channel_gram_reduction():
    rng = np.random.default_rng(15)
    c = rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7))
    h = rng.normal(size=(3, 7, 3)) + 1j * rng.normal(size=(3, 7, 3))
    bank = continuum_fc_ht_operator_bank(c, h)
    moments = rng.normal(size=4) + 1j * rng.normal(size=4)
    fc = c * moments[0]
    ht = np.einsum("scm,m->sc", h, moments[1:])
    expected = np.array(
        [
            np.sum(abs(fc) ** 2),
            np.sum(abs(ht) ** 2),
            2 * np.vdot(ht, fc).real,
            np.sum(abs(fc + ht) ** 2),
        ]
    )
    values = np.einsum("a,jab,b->j", moments, bank.forms, moments.conj()).real
    np.testing.assert_allclose(values, expected, rtol=1e-14, atol=1e-13)
    assert bank.constants.shape == (4,) and bank.gradients.shape == (4, 3)


def test_operator_bank_matches_native_fcht_ti():
    from matrix_trinity import HarmonicVibronicModel, harmonic_vibronic_ti
    from matrix_sparks import harmonic_photoionization_spectrum

    c = np.array([[1 + 0.2j], [0.3 - 0.1j], [0.2 + 0.4j]])
    h = np.array([[[0.1 + 0.2j]], [[0.2 - 0.1j]], [[-0.1 + 0.1j]]])
    model = HarmonicVibronicModel([1000.0], [1100.0], [[1.0]], [0.3], 30000.0, c[:, 0], h[:, 0, :])
    direct = harmonic_vibronic_ti(
        model, temperature_K=0.0, maximum_final_quanta=8, maximum_class=1, workers=1
    )
    bank = harmonic_photoionization_spectrum(
        model,
        c[None, ...],
        h[None, ...],
        electron_energies_hartree=[0.2],
        method="TI",
        temperature_K=0.0,
        maximum_final_quanta=8,
        maximum_class=1,
        workers=1,
    )
    expected = np.array([line["weighted_strength_au2"] for line in direct["lines"]])
    np.testing.assert_allclose(bank["strengths"][0, 3], expected, rtol=2e-12, atol=2e-13)
