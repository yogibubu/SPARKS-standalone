"""Separable reparametrizations of an independent primitive coordinate chart.

The scaled reciprocal, cosine and Morse coordinates have unit derivative at
reference, preserving units and the reference tangent gauge. Potential jets
and the full contravariant kinetic jet must be changed together. This module
does not replace SMITH symmetry construction or resolve redundant charts.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import numpy as np


@lru_cache(maxsize=5)
def _partitions(size):
    if size==0:return ((),)
    result=[]
    for p in _partitions(size-1):
        result.append(p+((size-1,),))
        for i in range(len(p)):
            result.append(p[:i]+(p[i]+(size-1,),)+p[i+1:])
    return tuple(result)


@dataclass(frozen=True)
class PrimitiveReparametrization:
    kinds: tuple
    reference: tuple
    morse_alpha: tuple | None = None

    def __post_init__(self):
        kinds=tuple(self.kinds);r=np.asarray(self.reference)
        if (r.ndim!=1 or len(r)!=len(kinds) or not len(r) or np.iscomplexobj(r)
                or not np.all(np.isfinite(r))):raise ValueError('one finite real reference per primitive required')
        alpha=np.zeros(len(r)) if self.morse_alpha is None else np.asarray(self.morse_alpha)
        if alpha.shape!=r.shape or np.iscomplexobj(alpha) or not np.all(np.isfinite(alpha)):
            raise ValueError('one finite real Morse scale per primitive required')
        for i,kind in enumerate(kinds):
            if kind not in {'linear','inverse_distance','cosine_angle','morse_stretch'}:
                raise ValueError('unknown primitive reparametrization')
            if kind in {'inverse_distance','morse_stretch'} and r[i]<=0:
                raise ValueError('positive reference distance required')
            if kind=='cosine_angle' and not (0<r[i]<np.pi and np.sin(r[i])>1e-7):
                raise ValueError('cosine chart requires a regular noncollinear angle')
            if kind=='morse_stretch' and alpha[i]<=0:
                raise ValueError('positive inverse-length Morse scale required')
        object.__setattr__(self,'kinds',kinds)
        object.__setattr__(self,'reference',tuple(map(float,r)))
        object.__setattr__(self,'morse_alpha',tuple(map(float,alpha)))

    def _coordinates(self,values):
        q=np.asarray(values)
        if (q.ndim<1 or q.shape[-1]!=len(self.kinds) or np.iscomplexobj(q)
                or not np.all(np.isfinite(q))):raise ValueError('finite real coordinates with matching last dimension required')
        return np.array(q,dtype=float,copy=True)

    def forward(self,primitive_values):
        x=self._coordinates(primitive_values);q=np.empty_like(x)
        for i,(kind,r,a) in enumerate(zip(self.kinds,self.reference,self.morse_alpha)):
            z=x[...,i]
            if kind=='linear':q[...,i]=z-r
            elif kind=='inverse_distance':
                if np.any(z<=0):raise ValueError('positive distances required')
                q[...,i]=r*(1-r/z)
            elif kind=='cosine_angle':
                if np.any((z<=0)|(z>=np.pi)):raise ValueError('angle outside regular cosine chart')
                q[...,i]=(np.cos(r)-np.cos(z))/np.sin(r)
            else:
                if np.any(z<=0):raise ValueError('positive distances required')
                q[...,i]=-np.expm1(-a*(z-r))/a
        if not np.all(np.isfinite(q)):raise ValueError('primitive map overflow')
        return q

    def inverse(self,mapped_displacements):
        q=self._coordinates(mapped_displacements);x=np.empty_like(q)
        for i,(kind,r,a) in enumerate(zip(self.kinds,self.reference,self.morse_alpha)):
            z=q[...,i]
            if kind=='linear':x[...,i]=r+z
            elif kind=='inverse_distance':
                if np.any(z>=r):raise ValueError('reciprocal chart pole crossed')
                x[...,i]=r/(1-z/r)
            elif kind=='cosine_angle':
                c=np.cos(r)-z*np.sin(r)
                if np.any(abs(c)>=1):raise ValueError('cosine chart boundary crossed')
                x[...,i]=np.arccos(c)
            else:
                if np.any(a*z>=1):raise ValueError('Morse coordinate boundary crossed')
                x[...,i]=r-np.log1p(-a*z)/a
                if np.any(x[...,i]<=0):raise ValueError('Morse inverse gives a nonpositive distance')
        return x

    def inverse_derivatives(self):
        """d^k primitive_i / dq_i^k at reference, k=1,...,4."""
        derivatives=np.zeros((4,len(self.kinds)));derivatives[0]=1.
        for i,(kind,r,a) in enumerate(zip(self.kinds,self.reference,self.morse_alpha)):
            if kind=='inverse_distance':derivatives[1:,i]=[2/r,6/r**2,24/r**3]
            elif kind=='cosine_angle':
                c=1/np.tan(r);derivatives[1:,i]=[-c,1+3*c*c,-9*c-15*c**3]
            elif kind=='morse_stretch':derivatives[1:,i]=[a,2*a*a,6*a**3]
        return derivatives

    def transform_potential_jet(self,gradient,hessian,cubic,quartic):
        """Exact chain rule through order four, including nonstationary terms."""
        n=len(self.kinds);source={}
        for order,value in enumerate((gradient,hessian,cubic,quartic),1):
            t=np.asarray(value)
            if t.shape!=(n,)*order or np.iscomplexobj(t) or not np.issubdtype(t.dtype,np.number) or not np.all(np.isfinite(t)):
                raise ValueError('complete finite potential jet required')
            source[order]=np.asarray(t,float)
        d=self.inverse_derivatives();out=[]
        for order in range(1,5):
            t=np.zeros((n,)*order)
            for ids in np.ndindex(t.shape):
                for partition in _partitions(order):
                    indices=[];factor=1.
                    for block in partition:
                        index=ids[block[0]]
                        if any(ids[j]!=index for j in block):factor=0.;break
                        indices.append(index);factor*=d[len(block)-1,index]
                    if factor:t[ids]+=source[len(partition)][tuple(indices)]*factor
            out.append(t)
        return tuple(out)

    def transform_metric_jet(self,metric,first=None,second=None):
        """Transform G/G1/G2 and the scalar for a flat-measure wavefunction.

        Returns the ADDITION to an existing reference scalar, in units of the
        kinetic scale (hbar^2 in the original coordinate units). Add it once.
        G derivative indices are last. The source may already be curvilinear.
        """
        n=len(self.kinds);G=np.asarray(metric)
        G1=np.zeros((n,)*3) if first is None else np.asarray(first)
        G2=np.zeros((n,)*4) if second is None else np.asarray(second)
        if (G.shape!=(n,n) or G1.shape!=(n,)*3 or G2.shape!=(n,)*4 or
                not all(np.issubdtype(t.dtype,np.number) and not np.iscomplexobj(t) and np.all(np.isfinite(t)) for t in (G,G1,G2)) or
                not np.allclose(G,G.T) or not np.allclose(G1,G1.swapaxes(0,1)) or
                not np.allclose(G2,G2.swapaxes(0,1)) or not np.allclose(G2,G2.swapaxes(2,3)) or np.linalg.eigvalsh(G).min()<=0):
            raise ValueError('complete positive metric jet required')
        G,G1,G2=(np.asarray(t,float) for t in (G,G1,G2))
        inverse=self.inverse_derivatives()
        # a_i(q)=dq_i/dprimitive_i. At reference a=1.
        a1=-inverse[1];a2=2*inverse[1]**2-inverse[2]
        B1=np.empty_like(G1);B2=np.empty_like(G2)
        for i,j,k in product(range(n),repeat=3):
            A1=a1[i]*(i==k)+a1[j]*(j==k)
            B1[i,j,k]=G1[i,j,k]+G[i,j]*A1
            for l in range(n):
                A1l=a1[i]*(i==l)+a1[j]*(j==l)
                A2=(a2[i]*(i==k)*(i==l)+a2[j]*(j==k)*(j==l)
                    +a1[i]*a1[j]*((i==k)*(j==l)+(i==l)*(j==k)))
                B2[i,j,k,l]=(G2[i,j,k,l]+G1[i,j,l]*A1+G1[i,j,k]*A1l
                    +G1[i,j,k]*inverse[1,k]*(k==l)+G[i,j]*A2)
        logJ1=-a1;logJ2=np.diag(-(a2-a1*a1))
        scalar=(np.einsum('iji,j',B1,logJ1)+np.sum(G*logJ2))/4+logJ1@G@logJ1/8
        return G.copy(),B1,B2,float(scalar)
