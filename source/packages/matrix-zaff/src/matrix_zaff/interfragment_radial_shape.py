"""Analytic pair profiles with either a single well or pure repulsion.

The derivative is factored in a bounded reciprocal-distance coordinate.
Positive Bernstein amplitudes retain its sign on the entire radial axis;
no sampled scan or additive correction is used to repair radial shape.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import beta, betainc, comb
from .interfragment_inverse_power import InterfragmentInversePower

BOHR_ANGSTROM = .529177210903


@dataclass(frozen=True)
class ShapePreservingInterfragment:
    pairs: tuple[tuple[int, int], ...]
    pair_groups: tuple[int, ...]
    positive_amplitudes_hartree: np.ndarray
    derivative_zeros: np.ndarray
    scale_angstrom: float
    provenance: str
    leading_power: int = 4

    def __post_init__(self):
        a = np.array(self.positive_amplitudes_hartree, float, copy=True)
        z = np.array(self.derivative_zeros, float, copy=True)
        if (a.ndim != 2 or not 1 <= a.shape[1] <= 65 or z.shape != (len(a),)
                or not np.all(np.isfinite(a)) or np.any(a < 0)
                or not np.all(np.isfinite(z))
                or not isinstance(self.leading_power, int) or self.leading_power < 2
                or np.any(z >= self.leading_power / (a.shape[1] + self.leading_power))
                or not np.isfinite(self.scale_angstrom) or self.scale_angstrom <= 0):
            raise ValueError('nonnegative amplitudes, positive scale and a repulsive inner branch required')
        # Reuse the resident pair/catalog validation. Its numerical radial
        # coefficients are not used by this functional form.
        catalog = InterfragmentInversePower(tuple(map(tuple, self.pairs)),
            tuple(self.pair_groups), np.zeros((len(a), 4)), provenance=self.provenance)
        a.setflags(write=False); z.setflags(write=False)
        object.__setattr__(self, 'pairs', catalog.pairs)
        object.__setattr__(self, 'pair_groups', catalog.pair_groups)
        object.__setattr__(self, 'positive_amplitudes_hartree', a)
        object.__setattr__(self, 'derivative_zeros', z)

    @property
    def degree(self):
        return self.positive_amplitudes_hartree.shape[1] - 1

    def _basis(self, radii, zeros):
        """Values and first/second derivatives with respect to angstrom."""
        r = np.asarray(radii, float)
        if not np.all(np.isfinite(r)) or np.any(r <= 0):
            raise ValueError('positive finite pair distances required')
        q = self.scale_angstrom / (r + self.scale_angstrom)
        q = q[..., None]
        k = np.arange(self.degree + 1); n = self.degree
        c = comb(n, k)
        b = c * q**k * (1-q)**(n-k)
        bprime = np.zeros_like(b)
        if n:
            bprime[..., 1:] += c[1:] * k[1:] * q**(k[1:]-1) * (1-q)**(n-k[1:])
            bprime[..., :-1] -= c[:-1] * (n-k[:-1]) * q**k[:-1] * (1-q)**(n-k[:-1]-1)
        p = self.leading_power
        i4 = c * beta(k+p, n-k+1) * betainc(k+p, n-k+1, q)
        i5 = c * beta(k+p+1, n-k+1) * betainc(k+p+1, n-k+1, q)
        z = np.asarray(zeros)[..., None]
        value = i5-z*i4
        dq = -q*q/self.scale_angstrom
        ddq = 2*q**3/self.scale_angstrom**2
        vq = q**(p-1)*(q-z)*b
        vqq = ((p-1)*q**(p-2)*(q-z)+q**(p-1))*b+q**(p-1)*(q-z)*bprime
        return value, vq*dq, vqq*dq*dq+vq*ddq

    def _geometry(self, geometries_angstrom):
        xyz = np.asarray(geometries_angstrom, float)
        if (xyz.ndim != 3 or xyz.shape[2] != 3
                or max(max(v) for v in self.pairs) >= xyz.shape[1]
                or not np.all(np.isfinite(xyz))):
            raise ValueError('finite batch in the stored atom order required')
        return xyz

    def design(self, geometries_angstrom):
        xyz = self._geometry(geometries_angstrom)
        rows = np.zeros((len(xyz), *self.positive_amplitudes_hartree.shape))
        for (a,b), group in zip(self.pairs, self.pair_groups, strict=True):
            r = np.linalg.norm(xyz[:,a]-xyz[:,b], axis=1)
            rows[:,group] += self._basis(r, self.derivative_zeros[group])[0]
        return rows.reshape(len(xyz), -1)

    def gradient_design(self, geometries_angstrom):
        xyz = self._geometry(geometries_angstrom)
        rows = np.zeros((len(xyz), len(xyz[0]), 3, *self.positive_amplitudes_hartree.shape))
        for (a,b), group in zip(self.pairs, self.pair_groups, strict=True):
            delta = xyz[:,a]-xyz[:,b]; r = np.linalg.norm(delta,axis=1)
            first = self._basis(r, self.derivative_zeros[group])[1]
            force = BOHR_ANGSTROM*delta[:,:,None]/r[:,None,None]*first[:,None,:]
            rows[:,a,:,group] += force; rows[:,b,:,group] -= force
        return rows.reshape(len(xyz), xyz.shape[1]*3, -1)

    def __call__(self, geometries_angstrom):
        return self.design(geometries_angstrom) @ self.positive_amplitudes_hartree.ravel()

    def cartesian_jet(self, geometry_angstrom):
        xyz = self._geometry(np.asarray(geometry_angstrom)[None])[0]
        gradient = np.zeros_like(xyz); hessian = np.zeros((xyz.size,xyz.size)); energy = 0.
        for (a,b), group in zip(self.pairs, self.pair_groups, strict=True):
            delta = xyz[a]-xyz[b]; radius = np.linalg.norm(delta); unit = delta/radius
            v,d,dd = self._basis(radius,self.derivative_zeros[group])
            amplitude = self.positive_amplitudes_hartree[group]
            energy += float(v@amplitude); first = float(d@amplitude); second = float(dd@amplitude)
            gradient[a] += first*unit; gradient[b] -= first*unit
            block = (second-first/radius)*np.outer(unit,unit)+first/radius*np.eye(3)
            ia,ib = slice(3*a,3*a+3),slice(3*b,3*b+3)
            hessian[ia,ia] += block; hessian[ib,ib] += block
            hessian[ia,ib] -= block; hessian[ib,ia] -= block
        return energy,gradient.ravel()*BOHR_ANGSTROM,hessian*BOHR_ANGSTROM**2

    def radial_shape_certificate(self):
        records = []
        for zero, amplitudes in zip(self.derivative_zeros,self.positive_amplitudes_hartree,strict=True):
            active = bool(np.any(amplitudes > 0))
            well = zero > 0
            records.append(dict(passed=active,branch='single_well' if well else 'repulsive',
                positive_stationary_count=int(well) if active else 0,
                stationary_distances_angstrom=[self.scale_angstrom*(1/zero-1)] if well else []))
        return dict(passed=all(v['passed'] for v in records),groups=records,
            scope='each native pair on all R>0; excludes additive corrections',
            collision_limit='finite repulsive inner branch',separated_limit=f'zero; R^-{self.leading_power} or faster')

    def to_dict(self):
        return dict(schema='matrix.zaff.shape-preserving-interfragment.v1',pairs=self.pairs,
            pair_groups=self.pair_groups,positive_amplitudes_hartree=self.positive_amplitudes_hartree.tolist(),
            derivative_zeros=self.derivative_zeros.tolist(),scale_angstrom=self.scale_angstrom,
            provenance=self.provenance,leading_power=self.leading_power)

    @classmethod
    def from_dict(cls, record):
        if record.get('schema') != 'matrix.zaff.shape-preserving-interfragment.v1':
            raise ValueError('unsupported radial shape record')
        return cls(**{key:record[key] for key in cls.__dataclass_fields__})

    @property
    def potential_id(self):
        import json,hashlib
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()
