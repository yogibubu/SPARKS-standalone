"""Bounded two/three-primitive residuals with analytic Cartesian E/G/H.

ORACLE supplies the pair equivalences. Every term contains at most three
distinct averaged native inverse-distance factors; repeated factors retain
their radial powers. Corrections stay bounded
at collision and decay at least as R^-4, preserving a fixed dipolar tail
and the positive divergent core of the preceding pair field.
"""
from dataclasses import dataclass
from itertools import combinations_with_replacement
import hashlib,json
import numpy as np
from .interfragment_inverse_power import InterfragmentInversePower


@dataclass(frozen=True)
class BoundedInterfragmentCouplings:
    pairs: tuple
    pair_groups: tuple
    monomials: tuple
    coefficients_hartree: np.ndarray
    provenance: str
    scale_angstrom: float = 3.
    primitive_orbits: tuple = ()

    def __post_init__(self):
        count=max(self.pair_groups)+1
        InterfragmentInversePower(self.pairs,self.pair_groups,np.zeros((count,1)),powers=(4,),provenance=self.provenance)
        terms=tuple(tuple(map(int,t))for t in self.monomials)
        coefficients=np.array(self.coefficients_hartree,float,copy=True)
        if (any(len(t)<2 or len(set(t))>3 or min(t)<0 or max(t)>=count for t in terms)
                or len(set(terms))!=len(terms) or coefficients.shape!=(len(terms),)
                or not np.all(np.isfinite(coefficients)) or not np.isfinite(self.scale_angstrom)
                or self.scale_angstrom<=0):
            raise ValueError('finite bounded couplings of at most three distinct primitives required')
        coefficients.setflags(write=False)
        object.__setattr__(self,'monomials',terms)
        object.__setattr__(self,'coefficients_hartree',coefficients)
        if self.primitive_orbits:
            orbits=tuple(tuple(tuple(map(int,t))for t in orbit)for orbit in self.primitive_orbits)
            if (tuple(self.pair_groups)!=tuple(range(len(self.pairs)))
                    or len(orbits)!=len(terms)
                    or any(not orbit or len(set(orbit))!=len(orbit) or term not in orbit
                        or any(len(t)!=len(term) or len(set(t))>3 or min(t)<0
                               or max(t)>=count for t in orbit)
                        for term,orbit in zip(terms,orbits))):
                raise ValueError('complete monomial orbits over individual pair primitives required')
            object.__setattr__(self,'primitive_orbits',orbits)

    @staticmethod
    def symmetry_terms(pairs,atom_permutations,maximum_factors,maximum_distinct_primitives=3):
        """Reynolds orbits from supplied ORACLE permutations; no perception.

        Products must be formed before symmetry averaging. Products of
        separately averaged primitives discard correlations between contacts.
        Permutations use zero-based atom indices in the stored atom order.
        """
        pairs=tuple(map(tuple,pairs));lookup={tuple(sorted(p)):i for i,p in enumerate(pairs)}
        maps=[]
        for permutation in atom_permutations:
            permutation=tuple(permutation)
            if (sorted(permutation)!=list(range(len(permutation)))
                    or max(max(p)for p in pairs)>=len(permutation)):
                raise ValueError('complete zero-based atom permutations required')
            try:
                maps.append(tuple(lookup[tuple(sorted((permutation[a],permutation[b])))]for a,b in pairs))
            except KeyError as error:
                raise ValueError('symmetry operation does not preserve the physical pair catalog')from error
        if not maps or tuple(range(len(pairs)))not in maps:
            raise ValueError('supplied symmetry group must contain the identity')
        maps=set(maps)
        if any(tuple(a[b[i]]for i in range(len(pairs)))not in maps for a in maps for b in maps):
            raise ValueError('supplied atom operations are not a closed pair-permutation group')
        catalog={}
        for term in BoundedInterfragmentCouplings.terms(len(pairs),maximum_factors,maximum_distinct_primitives):
            orbit=tuple(sorted({tuple(sorted(operation[i]for i in term))for operation in maps}))
            catalog[orbit[0]]=orbit
        representatives=tuple(sorted(catalog))
        return representatives,tuple(catalog[t]for t in representatives)

    def _orbits(self):
        return self.primitive_orbits or tuple((t,)for t in self.monomials)

    @staticmethod
    def terms(group_count,maximum_factors,maximum_distinct_primitives=3):
        if (not isinstance(group_count,int) or group_count<1
                or not isinstance(maximum_factors,int) or maximum_factors<0
                or not isinstance(maximum_distinct_primitives,int)
                or not 1<=maximum_distinct_primitives<=3):
            raise ValueError('positive catalog size and at most three distinct primitives required')
        return tuple(t for degree in range(2,maximum_factors+1)
                     for t in combinations_with_replacement(range(group_count),degree)
                     if len(set(t))<=maximum_distinct_primitives)

    def _descriptor_jet(self,xyz,second=False,first=True):
        xyz=np.asarray(xyz,float)
        if xyz.ndim!=3 or xyz.shape[-1]!=3 or not np.all(np.isfinite(xyz)):
            raise ValueError('finite Cartesian batch required')
        count=max(self.pair_groups)+1;ncart=xyz.shape[1]*3
        q=np.zeros((len(xyz),count));jac=np.zeros((len(xyz),count,ncart))if first or second else None
        hess=np.zeros((len(xyz),count,ncart,ncart))if second else None
        normalization=np.bincount(self.pair_groups,minlength=count);a=self.scale_angstrom**2
        for (i,j),group in zip(self.pairs,self.pair_groups):
            d=xyz[:,i]-xyz[:,j];denominator=np.sum(d*d,axis=1)+a
            q[:,group]+=a/denominator/normalization[group]
            if not first and not second:continue
            derivative=-2*a*d/denominator[:,None]**2*.529177210903/normalization[group]
            ia,ib=slice(3*i,3*i+3),slice(3*j,3*j+3)
            jac[:,group,ia]+=derivative;jac[:,group,ib]-=derivative
            if second:
                block=(-2*a*np.eye(3)[None]/denominator[:,None,None]**2
                    +8*a*np.einsum('ni,nj->nij',d,d)/denominator[:,None,None]**3)*.529177210903**2/normalization[group]
                hess[:,group,ia,ia]+=block;hess[:,group,ib,ib]+=block
                hess[:,group,ia,ib]-=block;hess[:,group,ib,ia]-=block
        return q,jac,hess

    def design(self,xyz):
        q,_,_=self._descriptor_jet(xyz,first=False)
        if not self.monomials:return np.zeros((len(q),0))
        return np.column_stack([sum(np.prod(q[:,t],axis=1)for t in orbit)/len(orbit)
                                for orbit in self._orbits()])

    def gradient_design(self,xyz):
        q,jac,_=self._descriptor_jet(xyz)
        result=np.zeros((len(q),jac.shape[2],len(self.monomials)))
        for k,orbit in enumerate(self._orbits()):
            for term in orbit:
                for a,g in enumerate(term):
                    rest=term[:a]+term[a+1:]
                    result[:,:,k]+=jac[:,g]*np.prod(q[:,rest],axis=1)[:,None]/len(orbit)
        return result

    def __call__(self,xyz):
        return self.design(xyz)@self.coefficients_hartree

    def cartesian_jet(self,xyz):
        q,jac,hess=self._descriptor_jet(np.asarray(xyz)[None],second=True)
        q,jac,hess=q[0],jac[0],hess[0];value=0.;g=np.zeros(len(q));h=np.zeros((len(q),len(q)))
        for orbit,coefficient in zip(self._orbits(),self.coefficients_hartree):
            c=coefficient/len(orbit)
            for term in orbit:
                value+=c*np.prod(q[list(term)])
                for a,i in enumerate(term):
                    rest=term[:a]+term[a+1:];g[i]+=c*np.prod(q[list(rest)])
                    for b,j in enumerate(term):
                        if a==b:continue
                        remainder=tuple(t for k,t in enumerate(term)if k not in (a,b))
                        h[i,j]+=c*np.prod(q[list(remainder)])
        result=jac.T@h@jac+np.einsum('i,ijk->jk',g,hess)
        return float(value),jac.T@g,(result+result.T)/2

    def to_dict(self):
        record=dict(schema='matrix.zaff.bounded-interfragment-couplings.v1',pairs=self.pairs,
            pair_groups=self.pair_groups,monomials=self.monomials,
            coefficients_hartree=self.coefficients_hartree.tolist(),
            provenance=self.provenance,scale_angstrom=self.scale_angstrom)
        if self.primitive_orbits:
            record.update(schema='matrix.zaff.bounded-interfragment-couplings.v2',
                          primitive_orbits=self.primitive_orbits)
        return record

    @classmethod
    def from_dict(cls,record):
        if record.get('schema')not in ('matrix.zaff.bounded-interfragment-couplings.v1',
                                       'matrix.zaff.bounded-interfragment-couplings.v2'):
            raise ValueError('unsupported native interfragment couplings')
        if record['schema'].endswith('v2')and not record.get('primitive_orbits'):
            raise ValueError('symmetrized primitive orbits missing')
        return cls(tuple(map(tuple,record['pairs'])),tuple(record['pair_groups']),
            tuple(map(tuple,record['monomials'])),record['coefficients_hartree'],record['provenance'],record['scale_angstrom'],
            tuple(tuple(map(tuple,orbit))for orbit in record.get('primitive_orbits',())))

    @property
    def potential_id(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()


class InterfragmentSum:
    """Composition of resident invariant E/G/H interactions, in original atom order."""
    def __init__(self,*components):
        if not components or any(not getattr(v,'potential_id','')for v in components):
            raise ValueError('nonempty identified resident interactions required')
        self.components=tuple(components)
        self.potential_id=hashlib.sha256(':'.join(v.potential_id for v in components).encode()).hexdigest()

    def __call__(self,xyz):
        return sum(v(xyz)for v in self.components)

    def cartesian_jet(self,xyz):
        values=[v.cartesian_jet(xyz)for v in self.components]
        e,g,h=(sum(v[k]for v in values)for k in range(3))
        return float(e),g,(h+h.T)/2

    def energy_gradient(self,xyz):
        e,g,_=self.cartesian_jet(xyz)
        return e,g
