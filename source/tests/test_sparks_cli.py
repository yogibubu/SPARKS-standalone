"""Public command argument forwarding, including subcommand help."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from matrix_sparks import photoelectron_cli, response_cli


@pytest.mark.parametrize('command', ['photoelectron', 'response'])
def test_standalone_dispatch_forwards_subcommand_help(command, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools/standalone/sparks/src/sparks/__main__.py'
    spec = importlib.util.spec_from_file_location('sparks_dispatch_test', path)
    dispatcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatcher)
    calls = []
    owner = photoelectron_cli if command == 'photoelectron' else response_cli
    monkeypatch.setattr(owner, 'main', lambda argv: calls.append(argv))
    dispatcher.main([command, '--help'])
    assert calls == [['--help']]


def test_photoelectron_command_accepts_explicit_arguments(tmp_path, monkeypatch):
    continuum = tmp_path / 'continuum.npz'
    nuclear = tmp_path / 'nuclear.npz'
    output = tmp_path / 'spectrum.npz'
    np.savez(continuum, energies_hartree=[.1, 1.],
             coefficients=np.ones((2, 3, 1, 1), complex), operator_basis_id='same')
    np.savez(nuclear, transition_energies_hartree=[.5],
             nuclear_moments=[[1.]], operator_basis_id='same')
    monkeypatch.setattr(photoelectron_cli, 'require_compute_host', lambda: 'test')
    photoelectron_cli.main(['--continuum', str(continuum), '--nuclear', str(nuclear),
                           '--photon-energies-ev', '21.218', '--output', str(output)])
    with np.load(output, allow_pickle=False) as result:
        strengths = result['cross_sections_Mb']
        assert strengths.shape == (1, 4, 1)
        assert strengths[0, 3, 0] > 0
        np.testing.assert_allclose(strengths[0, 0], strengths[0, 3])
        np.testing.assert_array_equal(strengths[0, 1:3], 0)
