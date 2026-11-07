"""Batch density and gradient magnitude using APOC states and compiled AO services."""

import ctypes
from pathlib import Path

import numpy as np


class ElectronicGrid:
    """Own a native grid context; separate instances are independent.

    A context reuses workspace and must not be called concurrently. Recreate it
    after changing geometry, basis or density. No electronic calculation runs.
    """

    def __init__(self, library: str | Path, molecule, coefficients, occupations):
        if not molecule.cart:
            raise ValueError("The qualified native grid requires Cartesian AOs")
        if np.iscomplexobj(coefficients):
            raise ValueError("The qualified grid accepts real AO coefficients only")
        self._library = ctypes.CDLL(str(Path(library).resolve()))
        lib = self._library
        pointer = ctypes.c_void_p
        lib.sparks_grid_create.argtypes = [ctypes.c_int] * 5 + [pointer] * 6
        lib.sparks_grid_create.restype = pointer
        lib.sparks_grid_destroy.argtypes = [pointer]
        lib.sparks_grid_evaluate.argtypes = [
            pointer,
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
        ]
        lib.sparks_grid_evaluate.restype = ctypes.c_int
        atm = np.ascontiguousarray(molecule._atm, dtype=np.int32)
        bas = np.ascontiguousarray(molecule._bas, dtype=np.int32)
        env = np.ascontiguousarray(molecule._env, dtype=np.float64)
        loc = np.ascontiguousarray(molecule.ao_loc_nr(), dtype=np.int32)
        coeff = np.asfortranarray(coefficients, dtype=np.float64)
        occ = np.ascontiguousarray(occupations, dtype=np.float64)
        if coeff.ndim != 2 or coeff.shape != (int(loc[-1]), len(occ)):
            raise ValueError("Coefficient/occupation dimensions do not match the AO basis")
        self._context = lib.sparks_grid_create(
            len(atm),
            len(bas),
            len(coeff),
            len(occ),
            len(env),
            *[x.ctypes.data for x in (atm, bas, loc, env, coeff, occ)],
        )
        if not self._context:
            raise ValueError("Native grid rejected the electronic state or unsupported basis")

    @classmethod
    def from_electronic_state(cls, library, state):
        """Reuse APOC AO conventions and the actual total density, including spin."""
        from matrix_apoc.population import pyscf_molecule_from_electronic_state
        from matrix_apoc.state import natural_orbitals

        molecule = pyscf_molecule_from_electronic_state(state)
        density = sum(state.density_ao(channel) for channel in range(len(state.mo_coefficients)))
        orbitals = natural_orbitals(density, state.overlap_ao, cutoff=0.0)
        return cls(library, molecule, orbitals.coefficients, orbitals.occupations)

    def evaluate(self, points_bohr, *, derivatives=True):
        """Return total density and |gradient density| on points with shape (N,3)."""
        if not self._context:
            raise RuntimeError("Grid context is closed")
        points = np.asfortranarray(points_bohr, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite (N,3) coordinates in bohr")
        density, gradient = np.empty(len(points)), np.empty(len(points))
        if len(points):
            status = self._library.sparks_grid_evaluate(
                self._context,
                len(points),
                points.ctypes.data,
                density.ctypes.data,
                gradient.ctypes.data,
                int(derivatives),
            )
            if status:
                raise RuntimeError(f"Native grid evaluation failed ({status})")
        return density, gradient

    def close(self):
        if getattr(self, "_context", None):
            self._library.sparks_grid_destroy(self._context)
            self._context = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
