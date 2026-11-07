"""Charge-free interfragment inverse-power component, with analytic E/G/H.

Physical pair groups and fragments are supplied by ORACLE; no topology or
atom typing is perceived here. Intrafragment energy is owned by the parent
fields and must be subtracted from fitting targets exactly once.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InterfragmentInversePower:
    pairs: tuple[tuple[int, int], ...]  # zero-based original atom order
    pair_groups: tuple[int, ...]
    coefficients_hartree: np.ndarray  # group x power, signed coefficients
    powers: tuple[int, ...] = (4, 6, 10, 12)
    reference_distance_angstrom: float = 3.
    provenance: str = ''
    softening_angstrom: float | tuple[float, ...] = 0.

    def __post_init__(self):
        p = tuple(self.powers)
        c = np.array(self.coefficients_hartree, float, copy=True)
        if (not self.pairs or len(self.pairs) != len(set(tuple(sorted(v)) for v in self.pairs))
                or any(len(v) != 2 or min(v) < 0 or v[0] == v[1] for v in self.pairs)
                or len(self.pair_groups) != len(self.pairs)
                or any(not isinstance(v, (int, np.integer)) or v < 0 for v in self.pair_groups)
                or c.ndim != 2 or c.shape[1] != len(p)
                or max(self.pair_groups) >= len(c) or not np.all(np.isfinite(c))
                or not p or any(not isinstance(v, int) or v <= 0 for v in p)
                or any(b <= a for a, b in zip(p, p[1:]))
                or not np.isfinite(self.reference_distance_angstrom)
                or self.reference_distance_angstrom <= 0 or not self.provenance):
            raise ValueError('unique atom pairs, compatible finite groups/powers and provenance required')
        c.setflags(write=False)
        object.__setattr__(self, 'coefficients_hartree', c)
        soft = np.asarray(self.softening_angstrom, float)
        if soft.ndim == 0:
            soft = np.full(len(c), float(soft))
        if soft.shape != (len(c),) or not np.all(np.isfinite(soft)) or np.any(soft < 0):
            raise ValueError('one finite nonnegative softening length per radial group required')
        object.__setattr__(self, 'softening_angstrom', tuple(map(float, soft)))

    def to_dict(self):
        record = dict(schema='matrix.zaff.interfragment-inverse-power.v1',
            pairs=[list(map(int,v))for v in self.pairs],pair_groups=list(map(int,self.pair_groups)),
            coefficients_hartree=self.coefficients_hartree.tolist(),powers=list(self.powers),
            reference_distance_angstrom=self.reference_distance_angstrom,provenance=self.provenance)
        if any(self.softening_angstrom):
            record.update(schema='matrix.zaff.interfragment-inverse-power.v2',
                          softening_angstrom=list(self.softening_angstrom))
        return record

    @classmethod
    def single_well(cls, pairs, pair_groups, minimum_distances_angstrom,
                    positive_amplitudes_hartree, *, provenance,
                    reference_distance_angstrom=3.):
        """Construct the native 4/6/10/12 form with one radial minimum.

        Put x=(Rref/R)^2 and x0=(Rref/Re)^2. Factoring dV/dx as
        x*(x-x0)*(a*x^3+b*x^2+x0*b*x+d), with a,b,d>0,
        guarantees the derivative sign on the entire positive axis. This
        retains the original powers, including finite-range cancellations,
        without fitting an arbitrary polynomial and repairing it afterwards.
        """
        re=np.asarray(minimum_distances_angstrom,float)
        amplitude=np.asarray(positive_amplitudes_hartree,float)
        if (re.ndim!=1 or amplitude.shape!=(len(re),3)
                or not np.all(np.isfinite(re)) or not np.all(np.isfinite(amplitude))
                or np.any(re<=0) or np.any(amplitude<=0)
                or not np.isfinite(reference_distance_angstrom)
                or reference_distance_angstrom<=0):
            raise ValueError('positive finite radial minima and shape amplitudes required')
        x0=(reference_distance_angstrom/re)**2
        a,b,d=amplitude.T
        coeff=np.column_stack((-x0*d/2,(d-x0*x0*b)/3,
                               (b-x0*a)/5,a/6))
        return cls(tuple(map(tuple,pairs)),tuple(pair_groups),coeff,
                   reference_distance_angstrom=reference_distance_angstrom,
                   provenance=provenance)

    def radial_shape_certificate(self):
        """Check every positive stationary radius, not a sampled scan.

        This certifies radial pair functions only. It does not certify a sum
        with arbitrary angular corrections or an added radial interpolant.
        Ambiguous multiple roots are rejected rather than labelled safe.
        """
        from numpy.polynomial import Polynomial
        records=[]
        for row, soft in zip(self.coefficients_hartree, self.softening_angstrom):
            derivative=np.zeros(max(self.powers)+1)
            derivative[np.array(self.powers)]=np.array(self.powers)*row
            nz=np.flatnonzero(derivative)
            if not len(nz):
                records.append(dict(passed=False,reason='zero radial function'))
                continue
            polynomial=Polynomial(derivative[nz[0]:]/np.max(abs(derivative)))
            roots=polynomial.roots()
            positive=sorted(float(z.real)for z in roots
                            if abs(z.imag)<=1e-8*max(1.,abs(z.real)) and z.real>0)
            accessible = [t for t in positive if soft == 0 or t < self.reference_distance_angstrom/soft]
            core = (float(row @ ((self.reference_distance_angstrom/soft)**np.array(self.powers)))
                    if soft else None)
            # The lowest retained coefficient fixes the separated branch;
            # the highest fixes collision repulsion.
            passed=bool(derivative[nz[0]]<0 and derivative[nz[-1]]>0
                        and len(accessible)==1 and (core is None or core > 0)
                        and polynomial.deriv()(accessible[0])>1e-10)
            records.append(dict(passed=passed,positive_stationary_count=len(accessible),
                collision_energy_hartree=core,
                stationary_distances_angstrom=[float(np.sqrt((self.reference_distance_angstrom/t)**2-soft**2)) for t in accessible]))
        return dict(passed=all(v['passed']for v in records),groups=records,
                    scope='each radial pair on all R>0; excludes additive corrections')

    @classmethod
    def from_dict(cls, record):
        if record.get('schema') not in ('matrix.zaff.interfragment-inverse-power.v1',
                                        'matrix.zaff.interfragment-inverse-power.v2'):
            raise ValueError('unsupported interfragment inverse-power record')
        return cls(tuple(map(tuple,record['pairs'])),tuple(record['pair_groups']),
            record['coefficients_hartree'],powers=tuple(record['powers']),
            reference_distance_angstrom=record['reference_distance_angstrom'],provenance=record['provenance'],
            softening_angstrom=record.get('softening_angstrom', 0.))

    @property
    def potential_id(self):
        import hashlib,json
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()

    def design(self, geometries_angstrom):
        xyz = np.asarray(geometries_angstrom, float)
        if (xyz.ndim != 3 or xyz.shape[2] != 3
                or max(max(v) for v in self.pairs) >= xyz.shape[1]
                or not np.all(np.isfinite(xyz))):
            raise ValueError('finite batch in original Cartesian atom order required')
        result = np.zeros((len(xyz), *self.coefficients_hartree.shape))
        for (a, b), group in zip(self.pairs, self.pair_groups, strict=True):
            radius = np.sqrt(np.sum((xyz[:, a]-xyz[:, b])**2, axis=1)+self.softening_angstrom[group]**2)
            if np.any(radius == 0):
                raise ValueError('coincident interfragment atoms have divergent repulsive energy')
            result[:, group] += (self.reference_distance_angstrom/radius[:, None])**np.array(self.powers)
        return result.reshape(len(xyz), -1)

    def __call__(self, geometries_angstrom):
        return self.design(geometries_angstrom) @ self.coefficients_hartree.ravel()

    def gradient_design(self, geometries_angstrom):
        """Coefficient-linear Cartesian gradient rows in inverse bohr."""
        xyz = np.asarray(geometries_angstrom, float)
        self.design(xyz)  # shared validation
        rows = np.zeros((len(xyz), xyz.shape[1], 3, *self.coefficients_hartree.shape))
        powers = np.asarray(self.powers)
        for (a, b), group in zip(self.pairs, self.pair_groups, strict=True):
            delta = xyz[:, a]-xyz[:, b]
            r2 = np.sum(delta*delta, axis=1)+self.softening_angstrom[group]**2
            values = (self.reference_distance_angstrom/np.sqrt(r2)[:, None])**powers
            first = -.529177210903*delta[:, :, None]*powers[None, None, :]*values[:, None, :]/r2[:, None, None]
            rows[:, a, :, group] += first
            rows[:, b, :, group] -= first
        return rows.reshape(len(xyz), -1, self.coefficients_hartree.size)

    def cartesian_jet(self, geometry_angstrom):
        """E in Hartree, gradient/Hessian with respect to Cartesian bohr."""
        xyz = np.asarray(geometry_angstrom, float)
        self.design(xyz[None])  # shared geometry validation
        n = xyz.size
        gradient, hessian = np.zeros_like(xyz), np.zeros((n, n))
        energy = 0.
        powers = np.asarray(self.powers)
        for (a, b), group in zip(self.pairs, self.pair_groups, strict=True):
            d = xyz[a]-xyz[b]
            r2 = float(d@d)+self.softening_angstrom[group]**2
            terms = self.coefficients_hartree[group]*(self.reference_distance_angstrom/np.sqrt(r2))**powers
            energy += float(sum(terms))
            first_over_r = -sum(powers*terms)/r2
            gradient[a] += first_over_r*d; gradient[b] -= first_over_r*d
            block = first_over_r*np.eye(3)+sum(powers*(powers+2)*terms)*np.outer(d,d)/r2**2
            ia, ib = slice(3*a, 3*a+3), slice(3*b, 3*b+3)
            hessian[ia, ia] += block; hessian[ib, ib] += block
            hessian[ia, ib] -= block; hessian[ib, ia] -= block
        bohr_A = .529177210903
        return energy, gradient.ravel()*bohr_A, hessian*bohr_A**2
