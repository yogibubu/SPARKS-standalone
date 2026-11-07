"""Export complete signed Dyson vectors from a finite, completed Gaussian batch.

Native root/pole summaries, MO maps and same-run orbital energies are required.
This invokes only the saved-file reader and rwfdump, never an electronic job.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from matrix_sparks.execution import require_compute_host


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plans', type=Path, nargs='+', required=True)
    p.add_argument('--rwfdump', type=Path, required=True)
    p.add_argument('--frozen-core', type=int, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    host = require_compute_host()
    if a.workers < 1:
        raise ValueError('Positive worker count required')
    jobs = {}
    for plan in a.plans:
        data = json.loads(plan.read_text())
        for state in data['states']:
            for job in state['jobs']:
                path = (plan.resolve().parent / job['input']).parent
                jobs[str(path)] = path
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def export(folder):
        log = folder/'job.log'
        text = log.read_text()
        if 'Normal termination of Gaussian' not in text:
            raise ValueError(f'Incomplete Gaussian calculation: {folder}')
        roots = re.findall(r'Final roots for representation\s+(\d+)(.*?)(?=Final roots for representation|Leave Link|\Z)', text, re.S)
        candidates = []
        for symmetry, section in roots:
            for index, match in enumerate(re.finditer(
                r'Eigenvalue \(eV\)\s+([-+\d.]+).*?polestrength\s+([-+\d.]+)\s+([-+\d.]+)', section),1):
                candidates.append((int(symmetry),index,-float(match[1]),float(match[3])))
        if len(candidates) != 1:
            raise ValueError(f'Expected one isolated native EPT pole, found {len(candidates)}: {folder}')
        symmetry, root, energy, strength = candidates[0]
        target = folder/'dyson.log'
        receipt = target.with_suffix('.json')
        if target.exists() and receipt.exists():
            return dict(folder=str(folder), reused=True, **json.loads(receipt.read_text()))
        mapping = folder/'map694.txt'
        with (folder/'rwfdump.log').open('w') as output:
            subprocess.run([str(a.rwfdump), str(folder/'job.rwf'),str(mapping),'694i'],
                           stdout=output,stderr=subprocess.STDOUT,check=True,env=environment)
        command = [sys.executable,str(Path(__file__).with_name('export_saved_ept_dyson.py')),
            '--rwf',str(folder/'job.rwf'),'--checkpoint',str(folder/'job.fchk'),
            '--log',str(log),'--mapping-dump',str(mapping),'--output',str(target),
            '--frozen-core',str(a.frozen_core),'--symmetry',str(symmetry),'--root',str(root),
            '--ionization-energy-ev',str(energy),'--printed-pole-strength',str(strength)]
        with (folder/'dyson-export.log').open('w') as output:
            subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True,env=environment)
        result = dict(folder=str(folder), reused=False, **json.loads(receipt.read_text()))
        print(json.dumps({key:result[key] for key in ('folder','ionization_energy_ev','pole_strength','ao_norm_error')}),flush=True)
        return result

    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        rows = list(executor.map(export,jobs.values()))
    a.output.write_text(json.dumps(dict(host=host, jobs=len(rows), electronic_jobs_launched=0,
                                       results=rows),indent=2)+'\n')


if __name__ == '__main__':
    main()
