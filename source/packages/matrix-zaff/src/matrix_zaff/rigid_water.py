"""Rigid Gaussian-charge water recipes and microsolvation interactions.

SENTINEL owns exploration and CYPHER owns ensemble analysis. This module owns
only the shared water parameters and the ZAFF pair Hamiltonian. The solute
may optionally be evaluated by an independently supplied ZAFF seed model.
"""
from __future__ import annotations
from dataclasses import dataclass
from importlib.resources import files
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .radial import ExpPEPotential, DoubleExponentialPotential
from .nonbonded import PersistentGaussianElectrostaticOperator, PersistentVerletNeighborList
from .electrostatic_exclusions import ExcludedGaussianElectrostaticOperator
from .tip3p_fb import ANGSTROM_TO_BOHR, KCAL_PER_MOL_TO_HARTREE


@dataclass(frozen=True)
class RigidWaterModel:
    name: str
    oh_distance_angstrom: float
    hoh_angle_degree: float
    hydrogen_charge_e: float
    gaussian_sigmas_angstrom: tuple[float, float, float]
    oxygen_exppe: ExpPEPotential  # kcal/mol, angstrom
    source_doi: str

    @property
    def charges_e(self):
        return np.array([-2*self.hydrogen_charge_e, self.hydrogen_charge_e, self.hydrogen_charge_e])

    def reference_geometry_bohr(self):
        a=math.radians(self.hoh_angle_degree)/2
        r=self.oh_distance_angstrom*ANGSTROM_TO_BOHR
        return np.array([[0.,0.,0.],[r*math.cos(a),r*math.sin(a),0.],[r*math.cos(a),-r*math.sin(a),0.]])

    def coordinates_from_poses(self, poses_bohr_radian):
        poses=np.asarray(poses_bohr_radian,dtype=float)
        if poses.ndim!=2 or poses.shape[1]!=6 or not np.all(np.isfinite(poses)):
            raise ValueError('Water poses require finite (nwater,6) translations/rotation vectors')
        return np.einsum('wij,sj->wsi',Rotation.from_rotvec(poses[:,3:]).as_matrix(),self.reference_geometry_bohr())+poses[:,:3,None].transpose(0,2,1)


def rigid_water_model(name='degauss-3p-exppe'):
    """Load an explicitly selected recipe; never modify the ZAFF0 library."""
    if str(name).lower().replace('_','-') not in {'degauss-3p-exppe','degauss-exppe'}:
        raise ValueError(f'Unknown rigid Gaussian water model: {name}')
    record=json.loads(files('matrix_zaff').joinpath('data/degauss_3p_exppe.json').read_text())
    source=record['published_parameters']; fitted=record['exppe_parameters']
    return RigidWaterModel('degauss-3p-exppe',source['oh_distance_angstrom'],source['hoh_angle_degree'],source['hydrogen_charge_e'],
        (source['oxygen_gaussian_sigma_angstrom'],source['hydrogen_gaussian_sigma_angstrom'],source['hydrogen_gaussian_sigma_angstrom']),
        ExpPEPotential(**fitted),record['source_doi'])


def degauss_double_exponential():
    """Published O--O reference, kcal/mol and angstrom (Table 2)."""
    record=json.loads(files('matrix_zaff').joinpath('data/degauss_3p_exppe.json').read_text())
    p=record['published_parameters']
    return DoubleExponentialPotential(p['oxygen_epsilon_kcal_per_mol'],p['oxygen_rmin_angstrom'],p['repulsive_exponent'],p['attractive_exponent'])


@dataclass(frozen=True)
class RigidWaterEvaluation:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    components_hartree: dict[str,float]


class RigidWaterMicrosolvation:
    """Finite-cluster E/G with rigid water, optional ZAFF solute and TIP3P wall.

    No water intramolecular bonded field is fitted. Intermolecular Gaussian
    electrostatics uses all O/H sites; water H has no van der Waals channel.
    Solute--water O pairs use geometric epsilon, additive half-radii and
    arithmetic alpha mixing unless explicit per-solute-site pairs are given.
    Cross electrostatics uses the supplied reference solute charges; its internal
    charge-response model is retained by the solute evaluator. This mixing
    prescription is separate from the published water model.
    """
    def __init__(self, water_count, *, water_model='degauss-3p-exppe', solute_model=None,
                 solute_oxygen_pairs=None, semiaxes_angstrom=None, electrostatic_backend='auto',
                 pair_switch_angstrom=10., pair_cutoff_angstrom=12., validate_rigid=True):
        if isinstance(water_count,bool) or int(water_count)!=water_count or water_count<1:
            raise ValueError('A positive integer water count is required')
        self.water_count=int(water_count); self.water=rigid_water_model(water_model)
        self.validate_rigid=bool(validate_rigid)
        self.switch=float(pair_switch_angstrom);self.cutoff=float(pair_cutoff_angstrom)
        if not 0<self.switch<self.cutoff:raise ValueError('Pair switch must lie inside cutoff')
        self.solute_runtime=None; self.solute_count=0; self._solute_cache=None
        q=[]; widths=[];numbers=[];self.cross=[]
        if solute_model is not None:
            from .seed_runtime import ZaffSeedDynamicsRuntime
            self.solute_runtime=ZaffSeedDynamicsRuntime(solute_model)
            self.solute_count=len(solute_model['atoms'])
            base=self.solute_runtime.electrostatics
            q=list(base.charges);widths=list(base.widths_bohr);numbers=list(base.atomic_numbers)
            for atom in solute_model['nonbonded_atoms']:
                eps=math.sqrt(float(atom['epsilon_hartree'])/KCAL_PER_MOL_TO_HARTREE*self.water.oxygen_exppe.epsilon)
                self.cross.append(ExpPEPotential(eps,float(atom['rmin_half_angstrom'])+self.water.oxygen_exppe.r_min/2,
                    (math.sqrt(160.)+self.water.oxygen_exppe.alpha)/2))
        elif solute_oxygen_pairs is not None:
            raise ValueError('Cross parameters require a solute model')
        if solute_oxygen_pairs is not None:
            if len(solute_oxygen_pairs)!=self.solute_count or not all(isinstance(p,ExpPEPotential) for p in solute_oxygen_pairs):
                raise ValueError('Supply one ExpPEPotential per solute site')
            self.cross=list(solute_oxygen_pairs)
        self._solute_electrostatics=None
        if self.solute_count:
            self._solute_electrostatics=PersistentGaussianElectrostaticOperator(np.array(q),tuple(numbers),backend=electrostatic_backend,widths_bohr=np.array(widths))
        q=np.r_[q,np.tile(self.water.charges_e,self.water_count)]
        widths=np.r_[widths,np.tile(np.array(self.water.gaussian_sigmas_angstrom)*ANGSTROM_TO_BOHR,self.water_count)]
        numbers=tuple(numbers)+ (8,1,1)*self.water_count
        gaussian=PersistentGaussianElectrostaticOperator(q,numbers,backend=electrostatic_backend,widths_bohr=widths)
        exclusions=[]
        for w in range(self.water_count):
            i=self.solute_count+3*w;exclusions.extend(((i,i+1),(i,i+2),(i+1,i+2)))
        self.electrostatics=ExcludedGaussianElectrostaticOperator(gaussian,exclusions)
        self.oxygen_indices=self.solute_count+3*np.arange(self.water_count)
        self.radial_indices=np.r_[np.arange(self.solute_count),self.oxygen_indices]
        self.pairs=PersistentVerletNeighborList(cutoff_bohr=self.cutoff*ANGSTROM_TO_BOHR,skin_bohr=ANGSTROM_TO_BOHR)
        self.confinement=None
        if semiaxes_angstrom is not None:
            from .tip3p_fb import tip3p_fb_uvdw_confinement
            self.confinement=tip3p_fb_uvdw_confinement(np.zeros(3),np.asarray(semiaxes_angstrom)*ANGSTROM_TO_BOHR,np.eye(3))

    def _coordinates(self, coordinates_bohr):
        xyz=np.asarray(coordinates_bohr,dtype=float)
        if xyz.shape!=(self.solute_count+3*self.water_count,3) or not np.all(np.isfinite(xyz)):
            raise ValueError('Coordinates must contain solute then O,H,H for each water')
        if self.validate_rigid:
            waters=xyz[self.solute_count:].reshape(-1,3,3)
            ref=self.water.reference_geometry_bohr()
            for i,j in ((0,1),(0,2),(1,2)):
                if not np.allclose(np.linalg.norm(waters[:,i]-waters[:,j],axis=1),np.linalg.norm(ref[i]-ref[j]),atol=2e-7,rtol=0):
                    raise ValueError('Water geometry is not rigid for the selected model')
        return xyz

    def evaluate(self, coordinates_bohr):
        xyz=self._coordinates(coordinates_bohr)
        electro=self.electrostatics.evaluate(xyz)
        energy=float(electro.energy_hartree);gradient=electro.gradient_hartree_per_bohr.reshape(-1,3).copy()
        components={'water_and_cross_electrostatics':energy,'short_range':0.,'solute':0.,'confinement':0.}
        if self.solute_count:
            solute_xyz=xyz[:self.solute_count]
            subtraction=self._solute_electrostatics.evaluate(solute_xyz)
            energy-=subtraction.energy_hartree; gradient[:self.solute_count]-=subtraction.gradient_hartree_per_bohr.reshape(-1,3)
            components['water_and_cross_electrostatics']-=subtraction.energy_hartree
            if self._solute_cache is None or not np.array_equal(solute_xyz,self._solute_cache[0]):
                self._solute_cache=(solute_xyz.copy(),self.solute_runtime.evaluate(solute_xyz/ANGSTROM_TO_BOHR))
            solute=self._solute_cache[1];energy+=solute.energy_hartree;gradient[:self.solute_count]+=solute.gradient_hartree_per_bohr.reshape(-1,3)
            components['solute']=float(solute.energy_hartree)
        for a,b in self.pairs.pairs(xyz[self.radial_indices]):
            if a<self.solute_count and b<self.solute_count:continue
            i,j=self.radial_indices[a],self.radial_indices[b]
            delta=(xyz[i]-xyz[j])/ANGSTROM_TO_BOHR;r=float(np.linalg.norm(delta))
            if r>=self.cutoff:continue
            if r<1e-10:raise ValueError('Coincident short-range sites')
            potential=self.cross[a] if a<self.solute_count else self.water.oxygen_exppe
            radial=potential.derivatives(r);value=radial.energy;first=radial.first
            if r>self.switch:
                x=(r-self.switch)/(self.cutoff-self.switch)
                sw=1-10*x**3+15*x**4-6*x**5
                ds=(-30*x*x+60*x**3-30*x**4)/(self.cutoff-self.switch)
                first=sw*first+ds*value;value*=sw
            value*=KCAL_PER_MOL_TO_HARTREE;first*=KCAL_PER_MOL_TO_HARTREE/ANGSTROM_TO_BOHR
            energy+=value;components['short_range']+=value
            g=first*delta/r;gradient[i]+=g;gradient[j]-=g
        if self.confinement is not None:
            from .tip3p_fb import TIP3P_FB_UVDW_TYPE
            wall=self.confinement.evaluate_single_type(xyz[self.oxygen_indices],TIP3P_FB_UVDW_TYPE)
            energy+=wall.energy_hartree;gradient[self.oxygen_indices]+=wall.gradient_hartree_per_bohr.reshape(-1,3)
            components['confinement']=float(wall.energy_hartree)
        return RigidWaterEvaluation(float(energy),gradient,components)

    def energy_gradient(self, coordinates_bohr):
        v=self.evaluate(coordinates_bohr)
        return v.energy_hartree,v.gradient_hartree_per_bohr

    def energy(self, coordinates_bohr):
        return self.evaluate(coordinates_bohr).energy_hartree
