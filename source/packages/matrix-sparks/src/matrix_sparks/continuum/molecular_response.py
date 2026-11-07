"""Molecular screening assembled from independent local continuum operators.

Gaussian auxiliary Coulomb integrals use PySCF/libcint without an SCF job.
External Cartesian dipoles remain exact separate operators, not fits in the
auxiliary space. No TIRESIA data or reference modules are imported.
"""
from time import perf_counter
import numpy as np
from scipy.linalg import eigh, qr, solve
from scipy.sparse import csc_matrix

from .response import (GalerkinSternheimerResponse, SpectralSternheimerResponse,
                       orbital_potential_sources, kernel_from_auxiliary_integrals, screen_response)
from .response_boundary import CoulombResponseBoundary
from .resolvent import SparseResolventBackend


class GaussianResponseAuxiliary:
    """Explicit charge-neutral Gaussian response subspace and exact Coulomb metric.

    One exact total-charge constraint removes the monopole direction. There is
    no numerical rank selection, density fitting or overlap regularization.
    """

    def __init__(self, molecule, auxiliary_basis):
        from pyscf import df
        from pyscf.gto.ft_ao import ft_ao
        if auxiliary_basis is None:
            raise ValueError('Explicit auxiliary basis required')
        self.molecule = df.addons.make_auxmol(molecule, auxiliary_basis)
        self.molecule.cart = molecule.cart
        charges = ft_ao(self.molecule, np.zeros((1,3)))[0]
        if not np.all(np.isfinite(charges)) or np.max(abs(charges.imag)) > 1e-12:
            raise ValueError('Finite real auxiliary charge integrals required')
        charges = charges.real
        if len(charges) < 2 or np.linalg.norm(charges) == 0:
            raise ValueError('Auxiliary basis must admit a nonempty neutral subspace')
        self.transform = qr(charges[:,None], mode='full')[0][:,1:]
        self.charges = charges@self.transform
        self.coulomb = self.transform.T@self.molecule.intor('int2c2e')@self.transform
        self.overlap = self.transform.T@self.molecule.intor('int1e_ovlp')@self.transform
        self.size = len(self.overlap)
        self.specification = auxiliary_basis

    def __call__(self, points):
        evaluator = 'GTOval_cart' if self.molecule.cart else 'GTOval_sph'
        return self.molecule.eval_gto(evaluator, points)@self.transform


class _MolecularBoundary:
    def __init__(self, calculation):
        self.calculation = calculation
        self.boundary = calculation.operators.boundary
        self.fallback = CoulombResponseBoundary(
            self.boundary, calculation.radius, calculation.channels.angular_momenta,
            calculation.charge, outer_radius=calculation.outer_radius,
            tail_potential=calculation.sampled_tail)

    def log_derivative(self, energy):
        z = complex(energy)
        if z.imag == 0 and z.real > 0:
            waves = self.calculation.boundary_waves(z.real)
            values = [np.diag(v) if np.asarray(v).ndim == 1 else v for v in
                      (waves.regular, waves.irregular, waves.regular_derivative,
                       waves.irregular_derivative)]
            f,g,fp,gp = values
            return solve((g+1j*f).T, (gp+1j*fp).T, check_finite=False).T
        return self.fallback.log_derivative(z)

    def __call__(self, energy):
        return -.5*self.boundary@csc_matrix(self.log_derivative(energy))@self.boundary.T


class LocalMolecularTDDFT:
    """Build susceptibility, local kernel and screened molecular amplitudes.

    The supplied local calculation owns the field, radial basis and exterior.
    Its initial density must be closed-shell. Occupied projector orbitals are
    explicitly recomputed as the lowest stationary eigenvectors of this local
    Hamiltonian, rather than treating imported Gaussian MOs as its eigenstates.
    xc_kernel is an explicit density derivative provider; None specifies the
    Hartree-only response. LB94 alone never selects an XC response kernel.

    The current quadrature owner supports a central spherical continuum basis;
    atomic-enriched operators are rejected rather than integrated incorrectly.
    """

    def __init__(self, calculation, density_state, auxiliary, *, density_grid,
                 xc_kernel, backend='spectral', quadrature_tolerance=2e-3):
        start = perf_counter()
        op = calculation.operators
        radial, channels = calculation.radial, calculation.channels
        n = len(channels.labels)*radial.size
        if op.hamiltonian.shape != (n,n):
            raise NotImplementedError('Molecular response quadrature requires central operators')
        if density_state.spin_channels != 1:
            raise ValueError('This molecular response owner requires a closed-shell density')
        molecule = calculation.hartree.molecule
        if (not np.array_equal(density_state.atomic_numbers,molecule.atom_charges())
                or not np.allclose(density_state.coordinates_bohr,molecule.atom_coords(),rtol=0,atol=1e-10)
                or not np.allclose(density_state.density_ao(0),calculation.hartree.density,
                                   rtol=1e-12,atol=1e-13)):
            raise ValueError('Response density must share the local field electronic state and frame')
        occupation = np.asarray(density_state.mo_occupations[0])
        if np.any((occupation != 0)&(occupation != 2)):
            raise ValueError('Closed-shell integer occupations required')
        nocc = int(np.count_nonzero(occupation == 2))
        if not 0 < nocc < n or backend not in ('spectral','sparse'):
            raise ValueError('Nonempty occupied/virtual spaces and explicit response backend required')
        if not np.isfinite(quadrature_tolerance) or quadrature_tolerance <= 0:
            raise ValueError('Positive explicit quadrature tolerance required')
        h,s = op.hamiltonian.toarray(), op.overlap.toarray()
        energies,c = eigh(h,s,subset_by_index=(0,nocc-1),check_finite=False)
        if np.any(energies >= 0):
            raise ValueError('Occupied local orbitals must all be bound')
        p = auxiliary.size+4
        sources = np.zeros((nocc,n,p))
        dyson_sources = np.zeros((n,p))
        metric = np.zeros((auxiliary.size,auxiliary.size))
        xc_integral = np.zeros_like(metric)
        occupied_metric = np.zeros((nocc,nocc))
        radial_values = radial.evaluate(radial.nodes)
        nc,nr = len(channels.labels),radial.size
        for radius, weight, row in zip(radial.nodes,radial.weights,radial_values):
            active = np.flatnonzero(row != 0)
            indices = (np.arange(nc)[:,None]*nr+active[None,:]).ravel()
            basis = (channels.values[:,:,None]*row[active][None,None,:]/radius).reshape(
                len(channels.weights),-1)
            occupied_values = basis@c[indices]
            points = radius*channels.directions
            w = weight*radius**2*channels.weights
            aux = auxiliary(points)
            potential = np.column_stack((aux, points, np.ones(len(points))))
            sources[:,indices,:] += orbital_potential_sources(basis,occupied_values,potential,w)
            dyson_values = calculation.dyson_orbital(points)[:,None]
            dyson_sources[indices] += orbital_potential_sources(basis,dyson_values,potential,w)[0]
            occupied_metric += occupied_values.T@(w[:,None]*occupied_values)
            metric += aux.T@(w[:,None]*aux)
            if xc_kernel is not None:
                rho,_ = density_grid(points)
                fxc = np.asarray(xc_kernel(rho))
                if fxc.shape != w.shape or not np.all(np.isfinite(fxc)):
                    raise ValueError('Explicit finite local XC derivative required on every quadrature point')
                xc_integral += aux.T@((w*fxc)[:,None]*aux)
        metric_error = np.linalg.norm(metric-auxiliary.overlap)/np.linalg.norm(auxiliary.overlap)
        occupied_error = np.linalg.norm(occupied_metric-np.eye(nocc))
        constant_error = np.linalg.norm(sources[:,:,-1]-(op.overlap@c).T)
        if max(metric_error,occupied_error,constant_error) > quadrature_tolerance:
            raise ArithmeticError('Response quadrature does not reproduce its independent metrics: '
                                  +str((metric_error,occupied_error,constant_error)))
        # Exact analytic constant-operator source, independently checked above.
        sources[:,:,-1] = (op.overlap@c).T
        kernel_aux = kernel_from_auxiliary_integrals(metric,auxiliary.coulomb,xc_integral)
        kernel = np.zeros((p,p));kernel[:auxiliary.size,:auxiliary.size]=kernel_aux
        self.calculation = calculation
        self.auxiliary = auxiliary
        self.energies = energies
        self.occupied = c
        self.sources = sources
        self.dyson_sources = dyson_sources
        self.occupations = np.full(nocc,2.)
        self.kernel = kernel
        self.boundary = _MolecularBoundary(calculation)
        owner,keyword = ((SpectralSternheimerResponse,'boundary') if backend == 'spectral'
                         else (GalerkinSternheimerResponse,'boundary_self_energy'))
        self.response = owner(h,s,c,energies,self.occupations,sources,**{keyword:self.boundary})
        self.external = np.zeros((p,4));self.external[-4:]=np.eye(4)
        self.diagnostics = dict(dimension=n,occupied_states=nocc,auxiliary_functions=auxiliary.size,
            auxiliary_charge_error=float(np.max(abs(auxiliary.charges))),
            auxiliary_overlap_relative_quadrature_error=float(metric_error),
            occupied_overlap_quadrature_error=float(occupied_error),
            constant_source_quadrature_error=float(constant_error),
            preparation_seconds=perf_counter()-start,backend=backend,
            kernel='Hartree' if xc_kernel is None else 'Hartree plus explicit local XC derivative',
            occupied_reference='stationary eigenvectors of the declared local Hamiltonian')

    def evaluate(self, photon_energy):
        omega = float(photon_energy)
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError('Positive real photon energy required')
        chi = self.response.evaluate(omega)
        screened = screen_response(chi,self.kernel,self.external)
        polarizability = screened.induced_density[-4:-1,:3]
        rows=[];independent_strength=0.;screened_strength=0.
        op=self.calculation.operators
        for i,(energy,occupation,source) in enumerate(zip(self.energies,self.occupations,self.sources)):
            kinetic=omega+energy
            if kinetic <= 0:continue
            solver=SparseResolventBackend(op.hamiltonian,op.overlap,op.boundary,source)
            vertex=solver.evaluate(kinetic,self.calculation.boundary_waves(kinetic)).amplitudes
            direct=vertex[-4:-1]
            amplitude=screened.effective_potential[:,:3].T@vertex
            bare=float(occupation*np.sum(abs(direct)**2))
            strength=float(occupation*np.sum(abs(amplitude)**2))
            independent_strength+=bare;screened_strength+=strength
            rows.append(dict(orbital=i,kinetic_energy_hartree=float(kinetic),
                             independent_strength=bare,screened_strength=strength,
                             amplitudes=amplitude))
        absorptive=float(-np.trace(polarizability).imag/np.pi)
        relative=abs(absorptive-screened_strength)/max(abs(absorptive),1e-100)
        dyson_row = None
        kinetic = omega-self.calculation.dyson.ionization_energy_ev/27.211386245988
        if kinetic > 0:
            solver = SparseResolventBackend(op.hamiltonian,op.overlap,op.boundary,self.dyson_sources)
            vertex = solver.evaluate(kinetic,self.calculation.boundary_waves(kinetic)).amplitudes
            amplitude = screened.effective_potential[:,:3].T@vertex
            direct = self.calculation.evaluate(kinetic).amplitudes
            direct_error = np.linalg.norm(vertex[-4:-1]-direct)/max(np.linalg.norm(direct),1e-100)
            dyson_row = dict(kinetic_energy_hartree=float(kinetic),
                pole_strength=self.calculation.dyson.pole_strength,
                independent_strength=float(np.sum(abs(direct)**2)),
                screened_strength=float(np.sum(abs(amplitude)**2)),amplitudes=amplitude,
                direct_source_relative_difference=float(direct_error))
        return dict(photon_energy_hartree=omega,susceptibility=chi,
                    effective_potential=screened.effective_potential,
                    polarizability=polarizability,channels=rows,
                    dyson=dyson_row,
                    independent_strength=independent_strength,screened_strength=screened_strength,
                    absorptive_strength=absorptive,optical_identity_relative_error=relative,
                    gauge_response_max=float(np.max(abs(screened.induced_density[:,-1]))),
                    screening_backward_error=screened.backward_error)
