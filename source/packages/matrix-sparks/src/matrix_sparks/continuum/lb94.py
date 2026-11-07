"""Local LB94 model potential from an explicit total density and its gradient.

The spin-unpolarized model combines LibXC LDA exchange/VWN correlation with
van Leeuwen--Baerends' beta=0.05 gradient correction (PRA 49, 2421, 1994).
It is a model potential, not the derivative of a fitted exchange energy.
No SCF, density fitting, density cutoff or Coulomb-tail replacement is selected.
"""

from functools import lru_cache

import numpy as np


def lb94_gradient_correction(density, gradient_magnitude, *, beta=0.05):
    """Spin-unpolarized LB correction, stable also for tiny Gaussian densities.

    x=2^(1/3)|grad rho|/rho^(4/3),
    dv=-beta*(rho/2)^(1/3)*x^2/(1+3*beta*x*asinh(x)).
    Inputs are the total density and total-density gradient in atomic units.
    """
    if np.iscomplexobj(density) or np.iscomplexobj(gradient_magnitude):
        raise ValueError("Physical density and gradient must be real")
    rho, grad = np.broadcast_arrays(
        np.asarray(density, float), np.asarray(gradient_magnitude, float)
    )
    if (
        not np.all(np.isfinite(rho))
        or not np.all(np.isfinite(grad))
        or np.any(rho < 0)
        or np.any(grad < 0)
        or not np.isfinite(beta)
        or beta <= 0
    ):
        raise ValueError("Require finite nonnegative density/gradient and positive beta")
    if np.any((rho == 0) & (grad != 0)):
        raise ValueError("A zero density cannot have a nonzero physical gradient")
    output = np.zeros(rho.shape)
    active = (rho > 0) & (grad > 0)
    lr, lg = np.log(rho[active]), np.log(grad[active])
    lx = np.log(2) / 3 + lg - 4 * lr / 3
    # log(asinh(exp(lx))) avoids overflowing x, x^2, and rho^(-4/3).
    lasinh = np.empty_like(lx)
    small, large = lx < -20, lx > 350
    middle = ~(small | large)
    lasinh[small] = lx[small]
    lasinh[large] = np.log(lx[large] + np.log(2))
    lasinh[middle] = np.log(np.arcsinh(np.exp(lx[middle])))
    log_denominator = np.logaddexp(-lx, np.log(3 * beta) + lasinh)
    output[active] = -np.exp(np.log(beta) + lg - lr - log_denominator)
    if not np.all(np.isfinite(output)):
        raise ArithmeticError("LB94 correction exceeds floating-point range")
    return output


@lru_cache(maxsize=1)
def _lda_reference():
    from pyscf.dft import libxc

    # LibXC's normal DFT density screening would introduce a finite jump in a
    # continuum potential. Own a separate functional; do not alter global DFT
    # defaults or replace the physical density by a floor.
    code = "SPARKS_LB94_LDA_VWN"
    libxc.register_custom_functional_(
        code,
        "LDA_X,LDA_C_VWN",
        omega=[0.0, 0.0],
        density_threshold=np.finfo(float).tiny,
        restricted=True,
        unrestricted=False,
    )
    return libxc, code


def lb94_xc(density, gradient_magnitude, *, beta=0.05):
    """LibXC LDA_X/LDA_C_VWN plus the spin-scaled LB gradient correction."""
    libxc, code = _lda_reference()
    correction = lb94_gradient_correction(density, gradient_magnitude, beta=beta)
    rho = np.broadcast_to(np.asarray(density, float), correction.shape)
    if rho.size == 0:
        return correction
    flat = rho.ravel()
    lda = np.empty_like(flat)
    # VWN5 has v_c = (4/3)*A*(b*x0-c)/gamma*rho^(1/3)
    # + O(rho^(1/2)). For rho^(1/6) <= machine epsilon the next term
    # is below relative floating-point resolution. Evaluate this analytic
    # limit instead of overflowing LibXC derivatives; the density is unchanged.
    dilute = flat <= np.finfo(float).eps ** 6
    gamma = (3 / (4 * np.pi)) ** (1 / 3)
    coefficient = (
        -((3 / np.pi) ** (1 / 3)) + (4 / 3) * 0.0310907 * (3.72744 * (-0.10498) - 12.9352) / gamma
    )
    lda[dilute] = coefficient * np.cbrt(flat[dilute])
    if np.any(~dilute):
        lda[~dilute] = libxc.eval_xc(code, flat[~dilute], spin=0, deriv=1)[1][0]
    result = lda.reshape(rho.shape) + correction
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("LB94 XC evaluation lost finite precision")
    return result


class LB94Potential:
    """Spatial XC callable accepting any explicit density/gradient provider.

    density_grid(points) returns (rho, |grad rho|), e.g. ElectronicGrid.evaluate
    or AODensityGradient. Add nuclear and Hartree fields separately.
    """

    def __init__(self, density_grid, *, beta=0.05):
        self.density_grid = density_grid
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("LB94 beta must be positive and finite")
        self.beta = float(beta)

    def __call__(self, points_bohr):
        density, gradient = self.density_grid(points_bohr)
        return lb94_xc(density, gradient, beta=self.beta)


class AODensityGradient:
    """Reuse APOC/PySCF AOs for the supplied total density; no electronic job."""

    def __init__(self, state, *, batch_size=4096):
        from matrix_apoc.population import pyscf_molecule_from_electronic_state

        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        self.molecule = pyscf_molecule_from_electronic_state(state)
        self.density = sum(state.density_ao(i) for i in range(state.spin_channels))
        if (
            np.iscomplexobj(self.density)
            or not np.all(np.isfinite(self.density))
            or not np.allclose(self.density, self.density.T, rtol=1e-12, atol=1e-13)
        ):
            raise ValueError("Density must be finite, real and symmetric")
        self.batch_size = int(batch_size)

    def __call__(self, points_bohr):
        points = np.asarray(points_bohr, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("Require finite (N,3) coordinates in bohr")
        density, gradient = np.empty(len(points)), np.empty(len(points))
        evaluator = "GTOval_cart_deriv1" if self.molecule.cart else "GTOval_sph_deriv1"
        for start in range(0, len(points), self.batch_size):
            stop = min(start + self.batch_size, len(points))
            ao = self.molecule.eval_gto(evaluator, points[start:stop])
            contracted = ao[0] @ self.density
            density[start:stop] = np.einsum("pi,pi->p", contracted, ao[0])
            vector = 2 * np.einsum("pi,xpi->px", contracted, ao[1:])
            # hypot scales before squaring: dilute Gaussian tails can have
            # finite components whose squares underflow to zero.
            gradient[start:stop] = np.hypot.reduce(vector, axis=1)
        return density, gradient


def lda_xc_density_derivative(density):
    """Spin-unpolarized LDA_X/VWN5 d(v_xc)/d(rho) for an explicit ALDA kernel.

    This is a separate response model, not the functional derivative of LB94.
    Strictly positive physical densities are required; no density floor or
    empirical screening factor is introduced. Unsupported tail values fail.
    """
    if np.iscomplexobj(density):
        raise ValueError('Kernel density must be real')
    rho=np.asarray(density,float)
    if not np.all(np.isfinite(rho)) or np.any(rho<=0):
        raise ValueError('ALDA kernel requires finite strictly positive density')
    if not rho.size:
        return np.empty_like(rho)
    libxc,code=_lda_reference()
    _,_,fxc,_=libxc.eval_xc(code,rho.ravel(),spin=0,deriv=2)
    result=np.asarray(fxc[0]).reshape(rho.shape)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError('Nonfinite local XC derivative; review the physical density domain')
    return result
