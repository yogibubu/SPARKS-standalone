import numpy as np
import pytest
from matrix_apoc import ElectronicState, read_gdv_ept_dysons
from matrix_apoc.gdv_dyson import write_ept_dyson_mo_export


def test_saved_vector_search_accepts_identical_records_and_rejects_ambiguity(tmp_path):
    from matrix_apoc.gdv_dyson import locate_saved_ept_vector
    path = tmp_path/'saved.rwf'
    vector = np.array([-9., .8, -.1, .2])
    np.r_[np.zeros(6), vector, np.zeros(3), vector].astype('<f8').tofile(path)
    offset, copies = locate_saved_ept_vector(path, 3, ionization_energy_ev=9.,
        printed_pole_strength=.69, block_words=8)
    assert offset == 6 and copies == (6, 13)
    np.r_[vector, [-9., -.8, .1, .2]].astype('<f8').tofile(path)
    with pytest.raises(ValueError, match='Multiple distinct'):
        locate_saved_ept_vector(path, 3, ionization_energy_ev=9., printed_pole_strength=.69)
    with pytest.raises(ValueError, match='No complete'):
        locate_saved_ept_vector(path, 3, ionization_energy_ev=10., printed_pole_strength=.69)


def test_saved_vector_roundtrip_preserves_all_signed_components(tmp_path):
    state = ElectronicState(atomic_numbers=np.array([2]), coordinates_bohr=np.zeros((1, 3)),
                            overlap_ao=np.eye(3), mo_coefficients=(np.eye(3),),
                            mo_occupations=(np.array([2., 0., 0.]),),
                            mo_energies_hartree=(np.array([-2., -.5, .3]),))
    path = tmp_path / 'dyson.log'
    vector = np.array([-.8, .1, -.002])
    write_ept_dyson_mo_export(path, state, vector, frozen_core=0,
                             ionization_energy_ev=12., pole_strength=float(vector@vector),
                             symmetry=2, root=1, provenance={'source': 'saved record'})
    (dyson,) = read_gdv_ept_dysons(path, state, frozen_core=0, method='NR2')
    np.testing.assert_array_equal(dyson.coefficients_ao, vector)
    with pytest.raises(FileExistsError):
        write_ept_dyson_mo_export(path, state, vector, frozen_core=0,
                                 ionization_energy_ev=12., pole_strength=float(vector@vector),
                                 symmetry=2, root=1, provenance={'source': 'saved record'})
