"""Check that reference package relocation preserves generated Fortran bytes."""

import argparse
import hashlib
import importlib
import json
import pkgutil
import socket
from pathlib import Path

from .. import __path__ as reference_path
from ..overlays.angular_cache import prepare_angular_cache
from ..overlays.gto_batch import prepare_gto_batch
from ..overlays.gto_cache import prepare_gto_cache
from ..overlays.gto_fast import prepare_gto_fast
from ..overlays.hartree_batch import prepare_hartree_batch
from ..overlays.sparse_blocks import prepare_sparse_blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    host = socket.getfqdn().lower()
    if "oracle" not in host and "mac.homenet" not in host and "mac-studio" not in host:
        raise RuntimeError("Reference qualification runs only on authorized compute hosts")
    modules = []
    for entry in pkgutil.walk_packages(reference_path, "matrix_sparks.reference.tiresia."):
        importlib.import_module(entry.name)
        modules.append(entry.name)
    root = args.output.parent / "layout-overlays"
    root.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    prepare_gto_cache(source, root / "gto_cache")
    prepare_gto_fast(root / "gto_cache", root / "gto_fast")
    prepare_gto_batch(source, root / "gto_cache", root / "gto_fast", root / "gto_batch")
    prepare_hartree_batch(source, root / "gto_batch", root / "hartree_batch")
    prepare_sparse_blocks(source, root / "sparse_blocks")
    prepare_angular_cache(source, root / "angular_cache")
    expected_hashes = {}
    for manifest_path in args.baseline_manifest:
        manifest = json.loads(manifest_path.read_text())
        for profile in (
            "gto_cache",
            "gto_fast",
            "gto_batch",
            "hartree_batch",
            "sparse_blocks",
            "angular_cache",
        ):
            entry = manifest.get(profile)
            if not entry:
                continue
            for item in entry.get("changes", [entry]):
                expected_hashes[profile + "/" + Path(item["source"]).name] = item[
                    "candidate_sha256"
                ]
            if "module_sha256" in entry:
                expected_hashes[profile + "/matrix_hartree_batch.f90"] = entry["module_sha256"]
    checks = []
    for candidate in sorted(root.glob("*/*.f90")):
        relative = candidate.relative_to(root)
        original = source / "gnu_compat" / relative
        expected = (
            expected_hashes[str(relative)]
            if args.baseline_manifest
            else hashlib.sha256(original.read_bytes()).hexdigest()
        )
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if expected != actual:
            raise ArithmeticError(f"Reference Fortran changed: {relative}")
        checks.append({"path": str(relative), "sha256": actual})
    report = dict(
        host=socket.getfqdn(), imported_modules=modules, unchanged_fortran=checks, status="passed"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(status="passed", imported_modules=len(modules), unchanged_fortran=len(checks))
        )
    )


if __name__ == "__main__":
    main()
