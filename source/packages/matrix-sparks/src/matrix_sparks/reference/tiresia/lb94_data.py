"""Read diagnostic AO/density exports; electronic-state ownership stays in APOC."""

from pathlib import Path
import numpy as np


def read_grid_state(path, atomic_numbers):
    """Convert exported libcint arrays through the existing APOC state contract."""
    from pyscf import gto
    from matrix_apoc.state import ElectronicState, gaussian_basis_from_pyscf

    with Path(path).open("rb") as stream:
        na, ns, nf, nm, ne = np.fromfile(stream, dtype="<i4", count=5)
        atm = np.fromfile(stream, dtype="<i4", count=6 * na).reshape(na, 6)
        bas = np.fromfile(stream, dtype="<i4", count=8 * ns).reshape(ns, 8)
        loc = np.fromfile(stream, dtype="<i4", count=ns + 1)
        env = np.fromfile(stream, dtype="<f8", count=ne)
        coefficients = np.fromfile(stream, dtype="<f8", count=nf * nm).reshape(nf, nm, order="F")
        occupations = np.fromfile(stream, dtype="<f8", count=nm)
        if stream.read(1):
            raise ValueError("Trailing AO export data")
    numbers = np.asarray(atomic_numbers, dtype=int)
    if len(numbers) != na or loc[-1] != nf:
        raise ValueError("AO/nuclear dimensions differ")
    coordinates = np.array([env[p : p + 3] for p in atm[:, 1]])
    mol = gto.Mole()
    mol._atm = atm.copy()
    mol._atm[:, 0] = numbers
    mol._bas = bas
    mol._env = env
    mol._atom = [(gto.mole._std_symbol(int(z)), p.tolist()) for z, p in zip(numbers, coordinates)]
    mol.cart = True
    mol.spin = 0
    mol.charge = int(np.sum(numbers) - round(np.sum(occupations)))
    mol._built = True
    raw_overlap = mol.intor_symmetric("int1e_ovlp")
    # TIRESIA's libcint shell coefficients retain its radial normalization.
    # Import into canonical APOC/PySCF AOs with the explicit diagonal map;
    # never renormalize the density or the molecular orbitals themselves.
    labels = [f"{symbol}{i}" for i, (symbol, _) in enumerate(mol._atom)]
    basis_by_atom = {label: [] for label in labels}
    for sh in range(mol.nbas):
        exponents = mol.bas_exp(sh)
        contractions = mol.bas_ctr_coeff(sh)
        shell = [int(mol.bas_angular(sh))]
        shell.extend(
            [[float(exponent), *row.tolist()] for exponent, row in zip(exponents, contractions)]
        )
        basis_by_atom[labels[mol.bas_atom(sh)]].append(shell)
    canonical = gto.M(
        atom=list(zip(labels, coordinates)),
        basis=basis_by_atom,
        unit="Bohr",
        cart=True,
        charge=mol.charge,
        spin=0,
        verbose=0,
    )
    overlap = canonical.intor_symmetric("int1e_ovlp")
    scale = np.sqrt(np.diag(raw_overlap) / np.diag(overlap))
    if not np.allclose(raw_overlap, scale[:, None] * overlap * scale, rtol=2e-12, atol=2e-13):
        raise ValueError("AO import requires a transformation beyond shell normalization")
    coefficients = scale[:, None] * coefficients
    mol = canonical
    return ElectronicState(
        atomic_numbers=numbers,
        coordinates_bohr=coordinates,
        overlap_ao=overlap,
        mo_coefficients=(coefficients,),
        mo_occupations=(occupations,),
        basis=gaussian_basis_from_pyscf(mol),
        charge=mol.charge,
        source=str(path),
        method="imported TIRESIA reference density",
    )


def read_projection(path):
    with Path(path).open("rb") as stream:
        nv, ni, ng, order, nsp = np.fromfile(stream, dtype="<i4", count=5)
        xg = np.fromfile(stream, dtype="<f8", count=ng)
        wg = np.fromfile(stream, dtype="<f8", count=ng)
        knots = np.fromfile(stream, dtype="<f8", count=nsp + order + 1)
        rho = np.fromfile(stream, dtype="<f8", count=nv * ni * ng).reshape(nv, ni * ng, order="F")
        xc = np.fromfile(stream, dtype="<f8", count=nv * ni * ng).reshape(nv, ni * ng, order="F")
        if stream.read(1):
            raise ValueError("Trailing projection data")
    boundaries = np.unique(knots)
    half = np.diff(boundaries) / 2
    radii = (boundaries[:-1, None] + half[:, None] * (xg + 1)).ravel()
    weights = (half[:, None] * wg).ravel()
    return dict(radii=radii, weights=weights, knots=knots, degree=int(order) - 1, rho=rho, xc=xc)


def read_angular(path):
    with Path(path).open("rb") as stream:
        n, nv = np.fromfile(stream, dtype="<i4", count=2)
        directions = np.fromfile(stream, dtype="<f8", count=3 * n).reshape(n, 3, order="F")
        weighted_harmonics = np.fromfile(stream, dtype="<f8", count=nv * n).reshape(
            nv, n, order="F"
        )
        angular = np.fromfile(stream, dtype="<i4", count=nv)
        if stream.read(1):
            raise ValueError("Trailing angular data")
    return directions, weighted_harmonics, angular


def read_electronic_potential(path, projection):
    """Read existing fort.23 Hartree+XC multipoles, before nuclear addition.

    The multicentre reference writes two records (potential and LB94 matching
    shift). It can append Poisson coefficients in other execution modes; those
    are not part of this point-value comparison. Dimensions come from the
    separately exported radial/XC grid, never from a guessed angular cutoff.
    """
    from scipy.io import FortranFile

    with FortranFile(path, "r") as stream:
        values = stream.read_reals(np.float64)
        shift = stream.read_reals(np.float64)
    shape = projection["xc"].shape
    if values.size != np.prod(shape) or shift.shape != (1,):
        raise ValueError("Reference potential records do not match the exported grid")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(shift)):
        raise ValueError("Nonfinite reference potential")
    return values.reshape(shape, order="F"), float(shift[0])


def rounded_reference_xc(density, gradient):
    """Audit only: reproduce TIRESIA EXCHG/FUDERVWN's printed legacy constants.

    Production uses LibXC VWN. This diagnostic isolates parameter rounding
    from density, quadrature and gradient-correction implementation errors.
    """
    rho, grad = np.broadcast_arrays(np.asarray(density, float), np.asarray(gradient, float))
    if np.any(rho <= 0):
        raise ValueError("The legacy reference formula requires positive density")
    r6 = rho ** (1 / 6)
    r3 = r6 * r6
    gamma, lgamma, sgamma = 0.6203504908994, -0.4774706527671, 0.7876233178997
    cb, x0, b, c, cc, q, ca = -0.0311676, -0.104980, 3.72744, 12.9352, 1.24742, 6.15199, 0.0310907
    a = gamma + b * sgamma * r6 + c * r3
    epsilon = ca * (
        lgamma
        - 2 * cb * np.log(sgamma - x0 * r6)
        + (cb - 1) * np.log(a)
        + cc * np.arctan(q * r6 / (2 * sgamma + b * r6))
    )
    derivative = (
        ca
        / 3
        * (
            1
            - cc * q * sgamma * r6 / (q * q * r3 + (2 * sgamma + b * r6) ** 2)
            - cb * sgamma / (sgamma - x0 * r6)
            + (cb - 1) * (gamma + 0.5 * b * sgamma * r6) / a
        )
    )
    lda = -((3 / np.pi) ** (1 / 3)) * r3 + epsilon - derivative
    x = 1.25992105 * grad / rho / rho ** (1 / 3)
    correction = (
        -0.05
        * rho ** (1 / 3)
        * x
        * x
        / 1.25992105
        / (1 + 0.15 * x * np.log(x + np.sqrt(x * x + 1)))
    )
    return lda + correction
