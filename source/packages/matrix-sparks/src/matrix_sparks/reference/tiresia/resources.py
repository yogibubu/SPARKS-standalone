"""Read the explicit Fortran reference adapters shipped with SPARKS."""

from pathlib import Path


def fortran_source(name: str) -> str:
    """Load an owned adapter, never an arbitrary user or upstream path."""
    if Path(name).name != name:
        raise ValueError("Expected a template filename")
    return (Path(__file__).with_name("fortran") / name).read_text()
