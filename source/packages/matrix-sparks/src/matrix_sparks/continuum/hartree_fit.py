"""Optional charge-constrained Coulomb fitting of an existing AO density.

Uses PySCF's existing auxiliary integrals, not an electronic SCF calculation.
The exact AOHartreePotential remains the reference/default. An explicit
auxiliary basis and independent potential/matrix checks are required before
adopting a fitted field for molecular observables.
"""

import hashlib
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .molecular import AOHartreePotential


class AODensityFittedHartreePotential:
    """Fit rho in the Coulomb metric while preserving its actual electron count.

    Minimize ||rho-sum_P c_P chi_P||_C with integral(sum_P c_P chi_P)=Tr(D*S).
    All three-centre AO-pair tensors are streamed in auxiliary-shell blocks.
    No renormalization of D, point-charge fitting or nuclear softening occurs.
    The workspace bound applies to each integral block, not the auxiliary
    metric/factorization or the complete Python process.
    """

    def __init__(self, state, auxiliary_basis, *, workspace_bytes=64*1024**2,
                 cholesky_tolerance=None):
        from pyscf import df, gto, lib
        from pyscf.gto.ft_ao import ft_ao

        if auxiliary_basis is None:
            raise ValueError('Select an explicit auxiliary basis')
        if cholesky_tolerance is not None and (
            not np.isfinite(cholesky_tolerance) or not 0 < cholesky_tolerance < 1
        ):
            raise ValueError('Cholesky tolerance must lie strictly between zero and one')
        reference = AOHartreePotential(state, workspace_bytes=workspace_bytes)
        mol, density = reference.molecule, reference.density
        aux = df.addons.make_auxmol(mol, auxiliary_basis)
        aux.cart = mol.cart
        locations = aux.ao_loc_nr()
        naux = int(locations[-1])
        na = len(density)
        pairs = np.tril_indices(na)
        packed = 2*density[pairs]
        packed[pairs[0] == pairs[1]] *= .5
        max_functions = int(workspace_bytes)//(8*len(packed))
        if max_functions < max(np.diff(locations)):
            raise ValueError('Workspace cannot hold one auxiliary shell integral block')
        atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                         aux._atm, aux._bas, aux._env)
        optimizer = gto.moleintor.make_cintopt(atm, bas, env, mol._add_suffix('int3c2e'))
        rhs = np.empty(naux)
        start = 0
        while start < aux.nbas:
            stop = start+1
            while stop < aux.nbas and locations[stop+1]-locations[start] <= max_functions:
                stop += 1
            integrals = df.incore.aux_e2(
                mol, aux, aosym='s2ij', cintopt=optimizer,
                shls_slice=(0,mol.nbas,0,mol.nbas,start,stop),
            )
            rhs[locations[start]:locations[stop]] = packed @ integrals
            start = stop
        metric = aux.intor('int2c2e')
        selected = np.arange(naux)
        if cholesky_tolerance is not None:
            scale = np.sqrt(metric.diagonal())
            if np.any(scale <= 0):
                raise ArithmeticError('Invalid auxiliary Coulomb metric diagonal')
            normalized = metric/scale[:,None]/scale[None,:]
            _, pivots, rank = lib.pivoted_cholesky(normalized, tol=cholesky_tolerance)
            selected = pivots[:rank]
        retained_metric = metric[np.ix_(selected,selected)]
        factor = cho_factor(retained_metric, lower=True, check_finite=False)
        charges = ft_ao(aux,np.zeros((1,3)))[0]
        if np.max(abs(charges.imag)) > 1e-13:
            raise ArithmeticError('Auxiliary integrals must be real')
        charges = charges.real[selected]
        retained_rhs = rhs[selected]
        coefficients = cho_solve(factor,retained_rhs,check_finite=False)
        unconstrained_charge = float(charges @ coefficients)
        direction = cho_solve(factor,charges,check_finite=False)
        denominator = float(charges @ direction)
        if denominator <= 0:
            raise ArithmeticError('Invalid auxiliary Coulomb charge constraint')
        multiplier = (reference.electron_number-unconstrained_charge)/denominator
        coefficients += multiplier*direction
        if not np.all(np.isfinite(coefficients)):
            raise ArithmeticError('Nonfinite Coulomb fit')
        self.auxiliary_molecule = aux
        self.coefficients = np.zeros(naux)
        self.coefficients[selected] = coefficients
        self.electron_number = reference.electron_number
        self.batch_size = max(1,int(workspace_bytes)//(8*naux))
        self.diagnostics = dict(
            auxiliary_basis=auxiliary_basis if isinstance(auxiliary_basis,str) else 'explicit per-atom basis',
            auxiliary_basis_sha256=hashlib.sha256(repr(auxiliary_basis).encode()).hexdigest(),
            auxiliary_functions=naux,
            retained_auxiliary_functions=len(selected),
            cholesky_tolerance=cholesky_tolerance,
            electron_number=self.electron_number,
            unconstrained_charge=unconstrained_charge,
            fitted_charge=float(charges@coefficients),
            charge_constraint_error=float(charges@coefficients-self.electron_number),
            stationarity_residual_max=float(abs(retained_metric@coefficients-retained_rhs-multiplier*charges).max()),
            point_charge_exponent=1e16,
        )

    def __call__(self, points_bohr):
        from pyscf import gto

        if np.iscomplexobj(points_bohr):
            raise ValueError('Hartree coordinates must be real')
        points=np.asarray(points_bohr,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.all(np.isfinite(points)):
            raise ValueError('Hartree coordinates must be finite (N,3) points')
        output=np.empty(len(points))
        for start in range(0,len(points),self.batch_size):
            end=min(start+self.batch_size,len(points))
            # Standard libcint point-charge limit. The associated Gaussian is
            # normalized to unit charge, with width 1e-8 bohr; this represents
            # the evaluation probe, not softened molecular nuclei.
            probes=gto.fakemol_for_charges(points[start:end],expnt=1e16)
            values=gto.intor_cross('int2c2e',self.auxiliary_molecule,probes)
            output[start:end]=self.coefficients@values
        return output
