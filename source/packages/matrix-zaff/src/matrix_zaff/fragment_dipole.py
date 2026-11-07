"""Covariant neutral-fragment dipoles with the resident Thole radial form.

Dipoles are fragment properties, not intermolecular atomic Coulomb charges.
Their linear Cartesian maps transport rigid rotations exactly. Intrafragment
property response away from the reference is only the declared linear map.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

BOHR_A = .529177210903


@dataclass(frozen=True)
class FragmentDipoleInteraction:
    center_weights: np.ndarray       # B COM minus A COM
    dipole_weights: np.ndarray       # two translation-invariant dipole maps
    polarizabilities_bohr3: tuple[float, float]
    provenance: str
    damping: float = .39

    def __post_init__(self):
        c = np.array(self.center_weights, float, copy=True)
        w = np.array(self.dipole_weights, float, copy=True)
        alpha = np.asarray(self.polarizabilities_bohr3, float)
        if (c.ndim != 1 or not len(c) or w.shape != (2, len(c))
                or not all(np.all(np.isfinite(v)) for v in (c, w, alpha))
                or alpha.shape != (2,) or np.any(alpha <= 0)
                or abs(c.sum()) > 1e-10 or np.max(abs(w.sum(axis=1))) > 1e-10
                or not np.isfinite(self.damping) or self.damping <= 0 or not self.provenance):
            raise ValueError('neutral covariant fragment maps, positive polarizabilities and provenance required')
        c.setflags(write=False); w.setflags(write=False)
        object.__setattr__(self, 'center_weights', c)
        object.__setattr__(self, 'dipole_weights', w)

    @classmethod
    def from_references(cls, coordinates_angstrom, fragment_atoms_zero_based,
                        masses_amu, dipoles_e_bohr, polarizabilities_bohr3, *, provenance):
        xyz = np.asarray(coordinates_angstrom, float)/BOHR_A
        masses = np.asarray(masses_amu, float)
        dipoles = np.asarray(dipoles_e_bohr, float)
        groups = tuple(np.asarray(g, int) for g in fragment_atoms_zero_based)
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or masses.shape != (len(xyz),)
                or np.any(masses <= 0) or dipoles.shape != (2,3) or len(groups) != 2
                or sorted(np.concatenate(groups).tolist()) != list(range(len(xyz)))
                or not all(np.all(np.isfinite(v)) for v in (xyz,masses,dipoles))):
            raise ValueError('two complete disjoint fragments and finite reference properties required')
        centers = np.zeros(len(xyz)); weights = np.zeros((2,len(xyz)))
        for k, group in enumerate(groups):
            centers[group] = (2*k-1)*masses[group]/sum(masses[group])
            # Centering improves conditioning; the zero scalar row enforces
            # translation invariance even for linear fragments.
            local = xyz[group]-xyz[group].mean(axis=0)
            design = np.vstack((np.ones(len(group)),local.T))
            target = np.r_[0.,dipoles[k]]
            values = np.linalg.lstsq(design,target,rcond=1e-12)[0]
            if np.max(abs(design@values-target)) > 1e-7:
                raise ValueError('reference geometry cannot transport this dipole component')
            weights[k,group] = values
        return cls(centers,weights,tuple(polarizabilities_bohr3),provenance)

    def to_dict(self):
        return dict(schema='matrix.zaff.fragment-dipole.v1',
            center_weights=self.center_weights.tolist(),dipole_weights=self.dipole_weights.tolist(),
            polarizabilities_bohr3=list(self.polarizabilities_bohr3),
            damping=self.damping,provenance=self.provenance)

    @classmethod
    def from_dict(cls, record):
        if record.get('schema') != 'matrix.zaff.fragment-dipole.v1':
            raise ValueError('unsupported fragment multipole record')
        return cls(*(record[k] for k in ('center_weights','dipole_weights','polarizabilities_bohr3','provenance')),damping=record['damping'])

    @property
    def potential_id(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()

    def _radial(self, radius):
        r = np.asarray(radius,float)
        if np.any(r <= 0):
            raise ValueError('distinct fragment centers required for dipole tensor')
        k = self.damping/np.sqrt(np.prod(self.polarizabilities_bohr3))
        x = k*r**3; exp = np.exp(-x)
        f3 = -np.expm1(-x)
        f5 = f3-x*exp
        # Stable second-order zero of the same resident Thole factor.
        small = x < 1e-3
        if np.any(small):
            import math
            series = sum((-1)**n*(n-1)/math.factorial(n)*x**n for n in range(2,9))
            f5 = np.where(small,series,f5)
        dx = 3*k*r*r; ddx = 6*k*r
        df3 = exp*dx; ddf3 = exp*(ddx-dx*dx)
        df5 = x*exp*dx; ddf5 = exp*((1-x)*dx*dx+x*ddx)
        a = np.stack((f3/r**3,df3/r**3-3*f3/r**4,
                      ddf3/r**3-6*df3/r**4+12*f3/r**5),axis=-1)
        b = -3*np.stack((f5/r**5,df5/r**5-5*f5/r**6,
                         ddf5/r**5-10*df5/r**6+30*f5/r**7),axis=-1)
        return a,b

    def __call__(self, geometries_angstrom):
        xyz = np.asarray(geometries_angstrom,float)/BOHR_A
        if (xyz.ndim != 3 or xyz.shape[1:] != (len(self.center_weights),3)
                or not np.all(np.isfinite(xyz))):
            raise ValueError('finite batch in original Cartesian atom order required')
        r = np.einsum('i,nij->nj',self.center_weights,xyz)
        mu = np.einsum('ki,nij->nkj',self.dipole_weights,xyz)
        a,b = self._radial(np.linalg.norm(r,axis=1))
        return a[:,0]*np.einsum('ni,ni->n',mu[:,0],mu[:,1])+b[:,0]*np.einsum('ni,ni->n',mu[:,0],r)*np.einsum('ni,ni->n',mu[:,1],r)

    def cartesian_jet(self, geometry_angstrom):
        xyz = np.asarray(geometry_angstrom,float)
        self(xyz[None])  # shared shape and finite-value qualification
        maps = np.kron(np.vstack((self.center_weights,self.dipole_weights)),np.eye(3))
        y = maps@xyz.ravel()/BOHR_A; r,u,v = y.reshape(3,3)
        radius = np.linalg.norm(r); unit = r/radius
        a,b = self._radial(radius)
        gr = np.r_[unit,np.zeros(6)]; hr = np.zeros((9,9))
        hr[:3,:3] = (np.eye(3)-np.outer(unit,unit))/radius
        z = u@v; gz = np.r_[np.zeros(3),v,u]; hz = np.zeros((9,9))
        hz[3:6,6:] = hz[6:,3:6] = np.eye(3)
        s,t = u@r,v@r
        gs,gt = np.r_[u,r,np.zeros(3)],np.r_[v,np.zeros(3),r]
        hs,ht = np.zeros((9,9)),np.zeros((9,9))
        hs[:3,3:6] = hs[3:6,:3] = np.eye(3)
        ht[:3,6:] = ht[6:,:3] = np.eye(3)
        h = s*t; gh = t*gs+s*gt
        hh = t*hs+s*ht+np.outer(gs,gt)+np.outer(gt,gs)
        ga,gb = a[1]*gr,b[1]*gr
        ha = a[2]*np.outer(gr,gr)+a[1]*hr
        hb = b[2]*np.outer(gr,gr)+b[1]*hr
        gradient = z*ga+a[0]*gz+h*gb+b[0]*gh
        hessian = z*ha+a[0]*hz+np.outer(ga,gz)+np.outer(gz,ga)+h*hb+b[0]*hh+np.outer(gb,gh)+np.outer(gh,gb)
        result = maps.T@hessian@maps
        return float(a[0]*z+b[0]*h),maps.T@gradient,(result+result.T)/2
