"""Pull the resident split-charge response through moving Gaussian sites.

ORACLE supplies site identities and response channels; SMITH supplies the site
positions and coordinate derivatives. This module neither perceives lone pairs
nor assigns a spatial direction to a degenerate electronic state.
"""
from dataclasses import dataclass, field
import numpy as np

from .charge_response import (ChargeResponseSite, PersistentSplitChargeResponse,
                              SplitChargeChannel, with_reference_channel_biases)


@dataclass(frozen=True)
class MappedChargeResponseResult:
    energy_correction_hartree: float
    gradient_correction_hartree_per_bohr: np.ndarray
    site_charges: np.ndarray
    charge_flows: np.ndarray
    residual_norm: float


@dataclass(frozen=True)
class MappedSplitChargeResponse:
    """Variational charge response for arbitrary differentiable site maps.

    The correction is added ONCE to the permanent Gaussian electrostatics,
    exactly as for ``PersistentSplitChargeResponse``. Site positions are in
    bohr and the Jacobian is d(flattened site positions)/d(Cartesian bohr).
    Widths, reference charges, hardnesses and biases are fixed parameters.
    Atomic, lone-pair and singly-occupied-density sites use the same solver.
    """
    reference_charges_e: tuple[float, ...]
    gaussian_widths_bohr: tuple[float, ...]
    channels: tuple[SplitChargeChannel, ...]
    tolerance: float = 1.e-12
    _runtime: PersistentSplitChargeResponse = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.reference_charges_e or len(self.reference_charges_e) != len(self.gaussian_widths_bohr):
            raise ValueError('one finite reference charge and positive width per site required')
        sites = tuple(ChargeResponseSite(q, w, ((i, 1.),)) for i, (q, w) in enumerate(
            zip(self.reference_charges_e, self.gaussian_widths_bohr)))
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('positive finite response tolerance required')
        if self.channels and max(max(c.left_site, c.right_site) for c in self.channels) >= len(sites):
            raise ValueError('response channel references an unavailable site')
        object.__setattr__(self, '_runtime', PersistentSplitChargeResponse(
            sites, self.channels, tolerance=self.tolerance, backend='direct'))

    def _geometry(self, site_coordinates_bohr, site_jacobian):
        xyz, jac = np.asarray(site_coordinates_bohr, float), np.asarray(site_jacobian, float)
        if (xyz.shape != (len(self.reference_charges_e), 3) or jac.ndim != 2
                or jac.shape[0] != xyz.size or jac.shape[1] == 0
                or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(jac))):
            raise ValueError('finite site coordinates and their complete Cartesian Jacobian required')
        return xyz, jac

    def evaluate(self, site_coordinates_bohr, site_jacobian):
        xyz, jac = self._geometry(site_coordinates_bohr, site_jacobian)
        response = self._runtime.solve(xyz)
        return MappedChargeResponseResult(response.energy_correction_hartree,
            jac.T @ response.gradient_correction_hartree_per_bohr,
            response.site_charges.copy(), response.charge_flows.copy(), response.residual_norm)

    def hessian_vector_product(self, site_coordinates_bohr, site_jacobian,
                               cartesian_direction, site_jacobian_directional_derivative):
        """Exact relaxed H v, including curvature of the off-atom site map.

        The final argument is dJ[x+tv]/dt at t=0, with the same shape as J.
        It is required even when zero, so nonlinear map curvature is never
        silently discarded.
        """
        xyz, jac = self._geometry(site_coordinates_bohr, site_jacobian)
        direction = np.asarray(cartesian_direction, float).reshape(-1)
        jac_dot = np.asarray(site_jacobian_directional_derivative, float)
        if (direction.shape != (jac.shape[1],) or jac_dot.shape != jac.shape
                or not np.all(np.isfinite(direction)) or not np.all(np.isfinite(jac_dot))):
            raise ValueError('finite Cartesian direction and complete site Jacobian derivative required')
        response = self._runtime.solve(xyz)
        site_product = self._runtime.hessian_vector_product(xyz, (jac @ direction).reshape(xyz.shape))
        return jac.T @ site_product + jac_dot.T @ response.gradient_correction_hartree_per_bohr


def isolated_fragment_channel_biases(site_coordinates_bohr, sites, channels, fragment_sites):
    """Calibrate each intrafragment channel on its isolated fragment only.

    The complete partition is supplied by ORACLE. Interfragment transfer must
    be introduced separately with an explicit asymptotic charge convention;
    it is not silently enabled by a polarization-only construction.
    """
    xyz = np.asarray(site_coordinates_bohr, float)
    groups = tuple(tuple(map(int, g)) for g in fragment_sites)
    if (xyz.shape != (len(sites), 3) or not np.all(np.isfinite(xyz))
            or any(not g for g in groups)
            or sorted(a for g in groups for a in g) != list(range(len(sites)))):
        raise ValueError('a complete disjoint fragment partition and finite site positions required')
    owners = {a: i for i, g in enumerate(groups) for a in g}
    if any(c.left_site not in owners or c.right_site not in owners
           or owners[c.left_site] != owners[c.right_site] for c in channels):
        raise ValueError('polarization channels must remain within their declared fragment')
    result = list(channels)
    for group in groups:
        index = {a: i for i, a in enumerate(group)}
        selected = [(j, c) for j, c in enumerate(channels) if c.left_site in index]
        local_sites = tuple(ChargeResponseSite(sites[a].reference_charge,
            sites[a].gaussian_width_bohr, ((i, 1.),)) for i, a in enumerate(group))
        local_channels = tuple(SplitChargeChannel(index[c.left_site], index[c.right_site],
            c.hardness_hartree, c.switch_radius_bohr, c.switch_exponent,
            0., c.overlap_bias_hartree)
            for _, c in selected)
        biased = with_reference_channel_biases(xyz[list(group)], local_sites, local_channels)
        for (j, original), local in zip(selected, biased):
            result[j] = SplitChargeChannel(original.left_site, original.right_site,
                original.hardness_hartree, original.switch_radius_bohr,
                original.switch_exponent, local.reference_bias_hartree,
                local.overlap_bias_hartree)
    return tuple(result)


def isolated_fragment_electronegativity_channels(site_coordinates_bohr, sites,
        channels, fragment_sites, fragment_chemical_potentials_hartree):
    """Reference SQE electronegativities on isolated neutral/charged fragments.

    chi_i = mu_fragment - (J_isolated q0)_i is fixed at the supplied reference.
    A channel has bias -s(R)*(chi_left-chi_right), with zero constant bias.
    Every transfer drive therefore vanishes at separation. A common shift
    of all chemical potentials cancels exactly. Chemical potentials are
    physical fitted inputs, not Gaussian electrostatic site potentials.
    """
    from .charge_response import GaussianChargeResponseOperator
    xyz = np.asarray(site_coordinates_bohr, float)
    groups = tuple(tuple(map(int, g)) for g in fragment_sites)
    mu = np.asarray(fragment_chemical_potentials_hartree, float)
    if (xyz.shape != (len(sites), 3) or not np.all(np.isfinite(xyz))
            or any(not g for g in groups)
            or sorted(i for g in groups for i in g) != list(range(len(sites)))
            or mu.shape != (len(groups),) or not np.all(np.isfinite(mu))):
        raise ValueError('complete disjoint site fragments and finite chemical potentials required')
    chi = np.empty(len(sites))
    for group, chemical_potential in zip(groups, mu):
        ids = list(group)
        local = tuple(ChargeResponseSite(sites[a].reference_charge,
            sites[a].gaussian_width_bohr, ((j, 1.),)) for j, a in enumerate(ids))
        operator = GaussianChargeResponseOperator.compile(xyz[ids], local, backend='direct')
        q = np.array([s.reference_charge for s in local])
        chi[ids] = chemical_potential - operator.matrix_vector(q)
    if any(max(c.left_site, c.right_site) >= len(sites) for c in channels):
        raise ValueError('response channel references an unavailable site')
    return tuple(SplitChargeChannel(c.left_site, c.right_site, c.hardness_hartree,
        c.switch_radius_bohr, c.switch_exponent, 0.,
        float(chi[c.right_site]-chi[c.left_site])) for c in channels)
