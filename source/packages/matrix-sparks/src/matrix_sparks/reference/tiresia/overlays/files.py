"""Emit reviewable reference overlays with reproducible source identities."""

import difflib
import hashlib
from pathlib import Path


def write_overlay(
    changes: dict[Path, str],
    destination: Path,
    *,
    source_root: Path | None = None,
    digest_field: str = "original_sha256",
) -> list[dict[str, str]]:
    """Write changed copies and one patch without altering upstream files.

    ``source_root`` makes provenance paths relative where a source tree is
    available. ``digest_field`` preserves the existing manifest schema.
    """
    metadata = []
    patch = []
    for filename, updated in changes.items():
        original = filename.read_text()
        output = destination / filename.name
        if not output.exists() or output.read_text() != updated:
            output.write_text(updated)
        source_name = str(filename.relative_to(source_root)) if source_root else str(filename)
        metadata.append(
            {
                "source": source_name,
                digest_field: hashlib.sha256(original.encode()).hexdigest(),
                "candidate_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            }
        )
        patch.extend(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=str(filename.relative_to(source_root)) if source_root else filename.name,
                tofile=f"{destination.name}/{filename.name}",
            )
        )
    patch_name = destination.name.replace("_", "-") + ".patch"
    (destination / patch_name).write_text("".join(patch))
    return metadata
