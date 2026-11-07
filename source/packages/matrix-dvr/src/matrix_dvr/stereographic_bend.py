"""Regular internal integration chart for a nonlinear four-atom terminal bend.

The integration chart contains three distances, a reference-plane angle and
both stereographic bend components. It is not a SMITH basis-selection policy.
The reference-plane atom keeps the whole molecule nonlinear when the reactive
three-atom chain becomes straight. No angular phase is evaluated at that point.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FourAtomBendingChart:
    # axis origin, bend center, terminal atom, reference-plane atom; zero based.
    order: tuple

    def __post_init__(self):
        if (len(self.order)!=4 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in self.order)
                or sorted(self.order)!=list(range(4))):
            raise ValueError('a complete zero-based four-atom permutation required')
        object.__setattr__(self,'order',tuple(map(int,self.order)))

    @classmethod
    def from_smith_definition(cls, definition):
        """Select a four-atom reactive bend from the frozen primitive graph.

        Two protected reaction distances define the chain. A unique ordinary
        distance to the fourth atom identifies the reference-plane endpoint.
        No atom identity, frequency sign or fitted rate enters the selection.
        Ambiguous graphs require an explicit a priori order.
        """
        if len(definition.reference_coordinates_angstrom)!=4:
            raise ValueError('this integration chart requires a four-atom contract')
        reactive={frozenset(i-1 for i in p.atoms) for p in definition.primitives
                  if p.function=='R' and p.family=='TS_REACTION_DISTANCE'}
        ordinary={frozenset(i-1 for i in p.atoms) for p in definition.primitives
                  if p.function=='R' and p.family=='STRETCH'}
        candidates=set()
        for first in reactive:
            for second in reactive:
                center=first&second;ends=first^second
                if len(center)!=1 or len(ends)!=2:continue
                residual=set(range(4))-(first|second)
                if len(residual)!=1:continue
                d=next(iter(residual));b=next(iter(center))
                for a in ends:
                    if frozenset((a,d)) in ordinary:
                        c=next(iter(ends-{a}));candidates.add((a,b,c,d))
        if len(candidates)!=1:
            raise ValueError('reaction graph does not identify one terminal bend and reference plane')
        return cls(next(iter(candidates)))

    def values(self, coordinates):
        x=np.asarray(coordinates,float)
        if x.shape!=(4,3) or not np.all(np.isfinite(x)):raise ValueError('finite four-atom geometry required')
        a,b,c,d=self.order
        axis=x[b]-x[a];r=np.linalg.norm(axis)
        spectator=x[d]-x[a];s=np.linalg.norm(spectator)
        terminal=x[c]-x[b];h=np.linalg.norm(terminal)
        if min(r,s,h)<=0:raise ValueError('positive chart distances required')
        axis/=r;cosine=np.dot(spectator,axis)/s
        perpendicular=spectator-axis*np.dot(spectator,axis);norm=np.linalg.norm(perpendicular)
        if norm<=1e-10*s:raise ValueError('reference-plane atom collinear with the bend axis')
        e1=perpendicular/norm;e2=np.cross(axis,e1);direction=terminal/h
        denominator=1+np.dot(direction,axis)
        if denominator<=1e-10:raise ValueError('antipodal stereographic pole reached')
        return np.array([s,np.arccos(np.clip(cosine,-1,1)),r,h,
            np.dot(direction,e1)/denominator,np.dot(direction,e2)/denominator])

    def geometry_torch(self, values):
        """Differentiable fixed embedding; no alignment or SVD is differentiated."""
        import torch
        s,alpha,r,h,u,v=values.unbind(-1)
        zero=s*0;denominator=1+u*u+v*v
        rows=[None]*4;a,b,c,d=self.order
        rows[a]=torch.stack((zero,zero,zero),dim=-1)
        rows[b]=torch.stack((zero,zero,r),dim=-1)
        rows[d]=torch.stack((s*torch.sin(alpha),zero,s*torch.cos(alpha)),dim=-1)
        rows[c]=torch.stack((h*2*u/denominator,h*2*v/denominator,
            r+h*(1-u*u-v*v)/denominator),dim=-1)
        return torch.stack(rows,dim=-2)

    def geometry(self, values):
        q=np.asarray(values,float)
        if (q.shape[-1:]!=(6,) or not np.all(np.isfinite(q)) or np.any(q[..., [0,2,3]]<=0)
                or np.any(q[...,1]<=0) or np.any(q[...,1]>=np.pi)):
            raise ValueError('positive distances, regular plane angle and finite bend components required')
        s,alpha,r,h,u,v=np.moveaxis(q,-1,0);zero=s*0;denominator=1+u*u+v*v
        rows=[None]*4;a,b,c,d=self.order
        rows[a]=np.stack((zero,zero,zero),axis=-1)
        rows[b]=np.stack((zero,zero,r),axis=-1)
        rows[d]=np.stack((s*np.sin(alpha),zero,s*np.cos(alpha)),axis=-1)
        rows[c]=np.stack((h*2*u/denominator,h*2*v/denominator,r+h*(1-u*u-v*v)/denominator),axis=-1)
        return np.stack(rows,axis=-2)

    def jet(self, values, *, transform=None, second=True):
        """Batched exact geometry jets, optionally restricted by an affine map."""
        import torch
        q=np.asarray(values,float);self.geometry(q)  # validate physical domain
        single=q.ndim==1;q=np.atleast_2d(q)
        if q.ndim!=2:raise ValueError('one chart point or a chart-point matrix required')
        t=np.eye(6) if transform is None else np.asarray(transform,float)
        if t.ndim!=2 or t.shape[0]!=6 or not np.all(np.isfinite(t)):raise ValueError('finite six-row tangent map required')
        tensor=torch.tensor(q,dtype=torch.float64);T=torch.tensor(t,dtype=torch.float64)
        def local(origin,point):return self.geometry_torch(origin+T@point)
        zero=torch.zeros(t.shape[1],dtype=torch.float64)
        first=torch.func.vmap(torch.func.jacfwd(local,argnums=1),in_dims=(0,None))(tensor,zero)
        result=dict(coordinates_angstrom=self.geometry(q),tangents=first.numpy())
        if second:
            derivative=torch.func.jacfwd(torch.func.jacfwd(local,argnums=1),argnums=1)
            result['second_derivatives']=torch.func.vmap(derivative,in_dims=(0,None))(tensor,zero).numpy()
        return {k:v[0] for k,v in result.items()} if single else result

    def to_dict(self):
        return dict(schema='matrix.dvr.four_atom_stereographic_bend.v1',order=list(self.order),
            units='angstrom,radian,dimensionless',scope='regular internal integration chart; no SMITH selection replaced')
