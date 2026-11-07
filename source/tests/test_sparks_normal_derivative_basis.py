"""Moving atomic supports in the general continuum derivative workflow."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def workflow():
    root = Path(__file__).resolve().parents[1]
    paths = [root / folder / "compute_sparks_normal_derivatives.py"
             for folder in ("scripts", "workflows")]
    path = next(p for p in paths if p.exists())
    spec = importlib.util.spec_from_file_location("normal_derivative_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atomic_supports_follow_displaced_geometry():
    module = workflow()
    settings = dict(atomic_radius=0.8, atomic_lmax=2, atomic_intervals=4)
    xyz = np.array([[0., 0., 0.], [0., 0., 2.]])
    blocks = module.atomic_blocks(settings, xyz)
    moved = module.atomic_blocks(settings, xyz + [0.1, -0.2, 0.3])
    assert len(blocks) == len(moved) == 2
    for i, (a, b) in enumerate(zip(blocks, moved)):
        np.testing.assert_allclose(a.centre, xyz[i])
        np.testing.assert_allclose(b.centre - a.centre, [0.1, -0.2, 0.3])
        assert a.size == b.size
        assert len(a.channels.labels) == 9


def test_central_default_and_invalid_atomic_support():
    module = workflow()
    assert module.atomic_blocks({}, [[0., 0., 0.]]) == ()
    with pytest.raises(ValueError, match="Positive atomic support"):
        module.atomic_blocks(dict(atomic_radius=-1, atomic_lmax=2, atomic_intervals=4),
                             [[0., 0., 0.]])


def test_atomic_integration_order_preserves_basis_and_checkpoint_identity(tmp_path):
    module = workflow()
    settings = dict(atomic_radius=0.8, atomic_lmax=2, atomic_intervals=4)
    refined = dict(settings, atomic_lebedev_order=29)
    a, = module.atomic_blocks(settings, [[0., 0., 0.]])
    b, = module.atomic_blocks(refined, [[0., 0., 0.]])
    assert a.size == b.size
    assert len(b.channels.directions) > len(a.channels.directions)
    source = tmp_path / "source"
    source.write_text("unchanged electronic data")
    assert module._fingerprint([source], settings) != module._fingerprint([source], refined)
