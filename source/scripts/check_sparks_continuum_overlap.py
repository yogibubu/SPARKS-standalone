"""Check the compact-enriched overlap without constructing an electronic field.

The same radial and compact-basis assembly is used as in the continuum solve.
A local Schur complement diagnoses underintegrated or dependent atomic blocks;
no overlap direction is removed and no potential is computed or changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from matrix_apoc import electronic_state_from_gaussian_fchk,reorient_electronic_state
from matrix_sparks.execution import require_compute_host
from matrix_sparks.continuum import RealSphericalChannels,assemble_sparse_radial
from matrix_sparks.continuum.multicentre import assemble_multicentre,compact_overlap_diagnostics
from compare_sparks_orbital_sources import atomic_blocks_from_settings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();host=require_compute_host();start=perf_counter()
    if args.output.exists():raise FileExistsError(args.output)
    spec=json.loads(args.specification.read_text());settings=spec['continuum']
    reference=electronic_state_from_gaussian_fchk(spec['dyson_checkpoint'])
    state=electronic_state_from_gaussian_fchk(spec['density_checkpoint'])
    state,_=reorient_electronic_state(state,reference.coordinates_bohr)
    lmax=settings['lmax']
    ch=RealSphericalChannels([(l,m) for l in range(lmax+1) for m in range(-l,l+1)],
                            **settings.get('angular_quadrature',{}))
    cuts=np.linalg.norm(state.coordinates_bohr,axis=1)
    edges=np.unique(np.r_[np.linspace(0,settings['radius'],settings['intervals']+1),cuts])
    central=assemble_sparse_radial(edges,ch.angular_momenta,
        lambda r:np.zeros((len(r),len(ch.labels),len(ch.labels))),
        degree=7,quadrature=settings['quadrature'])
    atoms=atomic_blocks_from_settings(settings,state.coordinates_bohr,state.atomic_numbers)
    if atoms:
        final=assemble_multicentre(central,ch,atoms,np.zeros(len(state.atomic_numbers)),
                                  state.coordinates_bohr,overlap_only=True)
        diagnostics=compact_overlap_diagnostics(final)
    else:
        final=central
        nr=central.overlap.shape[0]//len(ch.labels)
        minimum=float(np.linalg.eigvalsh(central.overlap[:nr,:nr].toarray())[0])
        diagnostics=dict(positive=minimum>0,local_dimension=0,
            minimum_normalized_schur_eigenvalue=None,maximum_normalized_schur_eigenvalue=None,
            negative_directions=0,minimum_central_radial_eigenvalue=minimum)
    report=dict(host=host,seconds=perf_counter()-start,dimension=final.overlap.shape[0],
        **diagnostics,continuum=settings,
        input_sha256={key:hashlib.sha256(Path(spec[key]).read_bytes()).hexdigest()
                      for key in ['density_checkpoint','dyson_checkpoint']},
        scope='Overlap qualification only; no electronic field or source approximation')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
