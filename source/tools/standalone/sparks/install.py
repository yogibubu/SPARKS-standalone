"""Install SPARKS from verified offline wheels or the supplied sources."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def verify(root, record, wheelhouse=False):
    for name, expected in record['files'].items():
        if wheelhouse and Path(name).name != name:
            raise ValueError('Invalid wheel filename')
        path = (root / ('wheels' if wheelhouse else '') / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Invalid archive path')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'File hash mismatch: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='store_true',
                        help='Build supplied sources; requires internet and a C++ compiler')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = json.loads((root / 'source-manifest.json').read_text())
    verify(root, source)
    if args.source:
        # Native builds are restricted to the authorized compute hosts.
        execution = runpy.run_path(str(root / 'source/packages/matrix-sparks/src/matrix_sparks/execution.py'))
        execution['require_compute_host']()
        with tempfile.TemporaryDirectory(prefix='sparks-install-') as directory:
            stage = Path(directory)
            projects = []
            for name in sorted(source['components']):
                target = stage / name
                shutil.copytree(root / 'source/packages' / name, target)
                projects.append(str(target))
            target = stage / 'sparks-standalone'
            shutil.copytree(root / 'source/tools/standalone/sparks', target)
            projects.append(str(target))
            subprocess.run([sys.executable, '-m', 'pip', 'install', *projects,
                            'pytest>=8,<9'], check=True)
    else:
        wheel_record = root / 'wheelhouse-manifest.json'
        if not wheel_record.exists():
            parser.error('Source archive: use --source on a compute host, or obtain the matching offline wheels')
        wheels = json.loads(wheel_record.read_text())
        if wheels['scientific_revision'] != source['scientific_revision']:
            raise ValueError('Wheels and source revisions differ')
        verify(root, wheels, wheelhouse=True)
        version = json.loads((root / 'source-manifest.json').read_text())['components']['matrix-sparks']
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                        '--find-links', str(root / 'wheels'),
                        f'sparks-standalone=={version}', 'pytest>=8,<9'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)


if __name__ == '__main__':
    main()
