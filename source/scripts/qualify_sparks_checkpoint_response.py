"""Reconstruct a molecular continuum response without any TIRESIA input.

Inputs are Gaussian density/orbital checkpoints and a signed Gaussian EPT
Dyson export. All continuum, auxiliary, response and screening matrices are
constructed afresh in SPARKS. Reference imports and external executable calls
are forbidden during the scientific calculation.
"""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np


class NoReferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('matrix_sparks.reference') or fullname.startswith('tiresia'):
            raise ImportError('Reference code is forbidden in checkpoint-only qualification: '+fullname)
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('density','checkpoint','dyson','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frozen-core',type=int,required=True)
    p.add_argument('--lmax',type=int,default=4)
    p.add_argument('--intervals',type=int,default=48)
    p.add_argument('--radius',type=float,default=18.)
    p.add_argument('--angular-order',type=int,default=18)
    p.add_argument('--auxiliary',choices=('compact','expanded'),default='compact')
    p.add_argument('--kernel',choices=('hartree','alda'),required=True)
    p.add_argument('--backend',choices=('spectral','sparse'),default='spectral')
    p.add_argument('--photon-energies',type=float,nargs='+',default=[.8,1.2,1.6])
    p.add_argument('--require-reference-absent',action='store_true')
    a=p.parse_args()
    sys.meta_path.insert(0,NoReferenceImports())
    from matrix_sparks.execution import require_compute_host
    host=require_compute_host()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    from matrix_apoc import electronic_state_from_gaussian_fchk,read_gdv_ept_dysons
    from matrix_sparks.continuum import (RealSphericalChannels,LocalPhotoionizationCalculation,
        AODensityGradient,LB94Potential,CoulombTailPotential,lda_xc_density_derivative)
    from matrix_sparks.continuum.molecular_response import GaussianResponseAuxiliary,LocalMolecularTDDFT
    from scipy.constants import fine_structure,physical_constants
    import matrix_sparks
    reference_present=(Path(matrix_sparks.__file__).parent/'reference').exists()
    if a.require_reference_absent and reference_present:
        raise RuntimeError('This proof requires the optional reference package to be physically absent')
    import platform
    platform.uname().processor  # Resolve OS metadata before the no-executable calculation.
    def forbid_reference_runtime(event, arguments):
        if event in ('subprocess.Popen','os.system','os.posix_spawn'):
            raise RuntimeError('External executables are forbidden during standalone response construction')
        if event=='open' and isinstance(arguments[0],(str,bytes)):
            path=str(arguments[0]).lower()
            if 'tiresia' in path:
                raise RuntimeError('TIRESIA files are forbidden during checkpoint-only response construction')
    sys.addaudithook(forbid_reference_runtime)
    start=perf_counter()
    density=electronic_state_from_gaussian_fchk(a.density)
    orbital_state=electronic_state_from_gaussian_fchk(a.checkpoint)
    dyson,=read_gdv_ept_dysons(a.dyson,orbital_state,frozen_core=a.frozen_core,method='NR2')
    channels=RealSphericalChannels([(l,m) for l in range(a.lmax+1) for m in range(-l,l+1)],
                                  polar_order=a.angular_order,azimuthal_order=2*a.angular_order+1)
    edges=np.unique(np.r_[np.linspace(0,a.radius,a.intervals+1),
                          np.linalg.norm(density.coordinates_bohr,axis=1)])
    density_grid=AODensityGradient(density)
    xc=CoulombTailPotential(LB94Potential(density_grid),inner_radius=10.,outer_radius=15.,charge=1.)
    calculation=LocalPhotoionizationCalculation(density,dyson,channels,edges,xc,
        asymptotic_charge=float(np.sum(density.atomic_numbers))-sum(
            np.trace(density.density_ao(i)@density.overlap_ao) for i in range(density.spin_channels))+1,
        outer_radius=120.,dipole_boundary=True)
    if a.auxiliary=='compact':
        shells=[[0,[.2,1.]],[0,[.8,1.]],[0,[3.2,1.]],[1,[.2,1.]],[1,[.8,1.]],[2,[.2,1.]]]
    else:
        shells=([[0,[v,1.]] for v in (.1,.2,.8,3.2,12.8)]+
                [[1,[v,1.]] for v in (.1,.2,.8,3.2)]+
                [[2,[v,1.]] for v in (.1,.2,.8)])
    # Explicit numerical auxiliary functions, not an electronic structure job.
    molecule=calculation.hartree.molecule
    basis={molecule.atom_symbol(i):shells for i in range(molecule.natm)}
    auxiliary=GaussianResponseAuxiliary(calculation.hartree.molecule,basis)
    response=LocalMolecularTDDFT(calculation,density,auxiliary,density_grid=density_grid,
        xc_kernel=lda_xc_density_derivative if a.kernel=='alda' else None,backend=a.backend)
    np.savez_compressed(a.output/'assembly.npz',overlap=calculation.operators.overlap.toarray(),
        hamiltonian=calculation.operators.hamiltonian.toarray(),
        boundary=calculation.operators.boundary.toarray(),occupied=response.occupied,
        occupied_energies_hartree=response.energies,kernel=response.kernel,
        sources=response.sources,dyson_sources=response.dyson_sources,
        auxiliary_transform=auxiliary.transform,auxiliary_coulomb=auxiliary.coulomb)
    rows=[]
    for i,omega in enumerate(a.photon_energies):
        value=response.evaluate(omega)
        np.savez_compressed(a.output/f'response-{i:03d}.npz',susceptibility=value['susceptibility'],
            effective_potential=value['effective_potential'],polarizability=value['polarizability'],
            dyson_amplitudes=value['dyson']['amplitudes'] if value['dyson'] else np.empty((0,0)))
        for key in ('susceptibility','effective_potential','polarizability'):
            del value[key]
        for row in value['channels']:del row['amplitudes']
        if value['dyson']:del value['dyson']['amplitudes']
        factor=4*np.pi**2*fine_structure*omega/3*physical_constants['Bohr radius'][0]**2/1e-22
        value['total_independent_cross_section_Mb']=factor*value['independent_strength']
        value['total_screened_cross_section_Mb']=factor*value['screened_strength']
        rows.append(value)
        print(json.dumps(value),flush=True)
    loaded=[name for name in sys.modules if name.startswith('matrix_sparks.reference')]
    if loaded:raise RuntimeError('Reference modules were imported: '+str(loaded))
    implementations={name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name,module in list(sys.modules.items())
        if name.startswith(('matrix_sparks.','matrix_apoc.'))
        and getattr(module,'__file__',None) and str(module.__file__).endswith('.py')}
    record=dict(host=host,seconds=perf_counter()-start,inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (a.density,a.checkpoint,a.dyson)},configuration=vars(a)|{'density':str(a.density),
        'checkpoint':str(a.checkpoint),'dyson':str(a.dyson),'output':str(a.output)},
        diagnostics=response.diagnostics,occupied_energies_hartree=response.energies.tolist(),
        reference_imports=loaded,reference_inputs=[],external_executables=[],
        reference_package_present=reference_present,
        implementation_sha256=implementations,electronic_jobs_launched=0,results=rows)
    (a.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
