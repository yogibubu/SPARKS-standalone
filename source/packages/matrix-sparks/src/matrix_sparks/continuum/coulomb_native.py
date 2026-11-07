"""Exact streamed libcint Coulomb contraction through the SPARKS C++ driver."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

import numpy as np


@lru_cache(maxsize=1)
def load_coulomb_kernel():
    """Build/cache the small driver against the installed libcint headers.

    Integral functions are supplied by the loaded PySCF library at evaluation;
    this driver neither links another integral engine nor requires TIRESIA.
    """
    import pyscf
    root = Path(pyscf.__file__).parent / 'lib'
    include = root / 'deps' / 'include'
    source = Path(__file__).parents[1] / 'native' / 'coulomb_contract.cpp'
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler or shutil.which(compiler[0]) is None:
        raise FileNotFoundError('A C++ compiler is required for direct Coulomb contraction')
    key = hashlib.sha256(source.read_bytes() + (include/'cint.h').read_bytes()
                         + (include/'cint_funcs.h').read_bytes()
                         + repr((platform.system(), platform.machine(), compiler)).encode()).hexdigest()[:24]
    folder = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache')) / 'matrix-sparks'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ('coulomb-' + key + ('.dylib' if platform.system()=='Darwin' else '.so'))
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            output = Path(temporary)/target.name
            command = compiler + ['-O3','-std=c++17','-fPIC',
                                  '-dynamiclib' if platform.system()=='Darwin' else '-shared',
                                  '-I'+str(include),str(source),'-o',str(output)]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError('SPARKS Coulomb driver compilation failed: '+process.stderr)
            os.replace(output,target)
    library = ctypes.CDLL(str(target))
    function = library.sparks_coulomb_contract
    pointer = ctypes.c_void_p
    function.argtypes = [pointer,pointer,ctypes.c_int,pointer,ctypes.c_int,
                         pointer,pointer,pointer,pointer,ctypes.c_int,ctypes.c_int,pointer]
    function.restype = ctypes.c_int
    return library


class DirectCoulombContraction:
    """Stream shell-pair blocks into a real density contraction, in double precision."""
    def __init__(self, molecule, *, workspace_bytes=64 * 1024**2):
        from pyscf.gto import moleintor
        self.library = load_coulomb_kernel()
        self.molecule = molecule
        name = molecule._add_suffix('int1e_grids')
        self.integral = getattr(moleintor.libcgto, name)
        self.locations = np.ascontiguousarray(molecule.ao_loc_nr(), dtype=np.int32)
        shell_size = int(np.max(np.diff(self.locations)))
        if (int(workspace_bytes) != workspace_bytes
                or workspace_bytes < 8 * shell_size**2):
            raise ValueError("Workspace must hold one shell-pair integral block")
        self.tile = min(312, int(workspace_bytes)//(8 * shell_size**2))
        self.optimizer = moleintor.make_cintopt(molecule._atm, molecule._bas, molecule._env, name)

    def __call__(self, points, density):
        from pyscf.gto.mole import NGRIDS, PTR_GRIDS
        if np.iscomplexobj(points) or np.iscomplexobj(density):
            raise ValueError("Direct Coulomb contraction requires real coordinates and density")
        points = np.asarray(points, dtype=float)
        mol = self.molecule
        density = np.ascontiguousarray(density, dtype=float)
        nao = int(self.locations[-1])
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or density.shape != (nao,nao) or not np.isfinite(density).all()):
            raise ValueError("Require finite (N,3) points and an AO density matrix")
        env = np.append(mol._env, points.ravel())
        env[NGRIDS], env[PTR_GRIDS] = len(points), len(mol._env)
        result = np.empty(len(points))
        if len(points):
            status = self.library.sparks_coulomb_contract(
                ctypes.cast(self.integral,ctypes.c_void_p), mol._atm.ctypes.data, mol.natm,
                mol._bas.ctypes.data, mol.nbas, env.ctypes.data, self.locations.ctypes.data,
                self.optimizer, density.ctypes.data, len(points), self.tile, result.ctypes.data)
            if status:
                raise RuntimeError('Direct Coulomb shell contraction failed')
        return result
