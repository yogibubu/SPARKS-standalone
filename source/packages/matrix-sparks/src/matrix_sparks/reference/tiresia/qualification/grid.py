"""Qualify SPARKS batches and APOC state reuse without running an SCF job."""

import argparse
import json
import socket
import sys
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower() != "oracle":
        raise RuntimeError("Run qualification on Oracle")
    sys.path.insert(0, str(args.python_source))
    import numpy as np
    from matrix_apoc.state import ElectronicState, gaussian_basis_from_pyscf
    from pyscf import gto

    from matrix_sparks import ElectronicGrid

    mol = gto.M(
        atom="O 0 0 0; H 0.1 0.4 0.8; H 0.3 0.9 -0.2",
        cart=True,
        verbose=0,
        basis={"O": [[l, [0.25, 1.0], [0.8, 0.3]] for l in range(5)], "H": "sto-3g"},
    )
    overlap = mol.intor_symmetric("int1e_ovlp")
    eigenvalues, vectors = np.linalg.eigh(overlap)
    coefficients = (vectors / np.sqrt(eigenvalues))[:, :8]
    occupations = np.array([2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0])
    rng = np.random.default_rng(26102026)
    errors = []
    with ElectronicGrid(args.library, mol, coefficients, occupations) as grid:
        for count in (0, 1, 55, 56, 57, 255, 256, 257, 1031):
            xyz = rng.normal(size=(count, 3)) * 3
            den, grad = grid.evaluate(xyz)
            if not count:
                continue
            ao = mol.eval_gto("GTOval_cart_deriv1", xyz)
            mo = ao @ coefficients
            refden = np.sum(mo[0] ** 2 * occupations, axis=1)
            refgrad = np.linalg.norm(
                2 * np.sum(mo[1:] * mo[0][None, :, :] * occupations, axis=2), axis=0
            )
            if not np.allclose(den, refden, rtol=2e-13, atol=2e-13):
                raise ArithmeticError("Density mismatch")
            if not np.allclose(grad, refgrad, rtol=2e-13, atol=2e-13):
                raise ArithmeticError("Gradient mismatch")
            scalar, zero = grid.evaluate(xyz, derivatives=False)
            if not np.allclose(scalar, refden, rtol=2e-13, atol=2e-13) or np.any(zero):
                raise ArithmeticError("Density-only mismatch")
            errors.append(
                dict(
                    points=count,
                    density_max_absolute=float(np.max(abs(den - refden))),
                    gradient_max_absolute=float(np.max(abs(grad - refgrad))),
                )
            )
    try:
        grid.evaluate([[0, 0, 0]])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Closed context accepted")
    # Explicitly differ from MO occupations, preserving total electron number:
    # from_electronic_state must import the supplied actual density.
    actual_occupations = occupations + np.array([0.0, 0.0, 0.0, -0.2, 0.1, 0.1, 0.0, 0.0])
    density = (coefficients * actual_occupations) @ coefficients.T
    state = ElectronicState(
        atomic_numbers=mol.atom_charges(),
        coordinates_bohr=mol.atom_coords(),
        overlap_ao=overlap,
        mo_coefficients=(coefficients,),
        mo_occupations=(occupations,),
        density_matrices_ao=(density,),
        basis=gaussian_basis_from_pyscf(mol),
        charge=0,
        multiplicity=1,
    )
    xyz = rng.normal(size=(257, 3))
    ao = mol.eval_gto("GTOval_cart", xyz)
    target = np.einsum("pi,ij,pj->p", ao, density, ao)
    with ElectronicGrid.from_electronic_state(args.library, state) as grid:
        actual, _ = grid.evaluate(xyz)
        if not np.allclose(actual, target, rtol=2e-12, atol=2e-12):
            raise ArithmeticError("APOC actual-density transport mismatch")
    spin_state = replace(
        state,
        mo_coefficients=(coefficients, coefficients),
        mo_occupations=(occupations / 2, occupations / 2),
        density_matrices_ao=(density / 2, density / 2),
    )
    with ElectronicGrid.from_electronic_state(args.library, spin_state) as grid:
        spin_density, _ = grid.evaluate(xyz)
        if not np.allclose(spin_density, target, rtol=2e-12, atol=2e-12):
            raise ArithmeticError("APOC spin density sum mismatch")
    report = dict(
        host=socket.gethostname(),
        seed=26102026,
        no_scf_calculation=True,
        cartesian_shells="s–g",
        batch_boundaries=errors,
        closed_context_checked=True,
        apoc_actual_density_max_absolute=float(np.max(abs(actual - target))),
    )
    report["apoc_spin_density_max_absolute"] = float(np.max(abs(spin_density - target)))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
