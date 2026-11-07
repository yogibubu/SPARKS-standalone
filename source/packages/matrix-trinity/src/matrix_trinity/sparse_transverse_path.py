"""Sparse positive transverse Hessians in a geometry-defined moving frame.

The frame uses geometry, IRC tangent and a reference bath only. A matrix-log
interpolant preserves a positive harmonic block without repairing frequencies.
No off-knot Hessian or mode vector is required to evaluate the interpolant.
"""
import numpy as np
from scipy.interpolate import interp1d


def geometric_transverse_frame(xyz, masses, tangent, reference_vectors):
    """Nearest reference frame orthogonal to rigid motion and the IRC tangent.

    All vectors and the tangent are mass weighted and in one Cartesian gauge.
    The geometry must be nonlinear; linear geometries require a separate
    continuous rigid-motion gauge and are explicitly rejected here.
    """
    xyz=np.asarray(xyz,float);mass=np.asarray(masses,float)
    t=np.asarray(tangent,float);ref=np.asarray(reference_vectors,float)
    if (xyz.shape!=(len(mass),3) or t.shape!=(3*len(mass),)
            or ref.shape[0]!=len(t) or np.any(mass<=0)
            or not all(np.all(np.isfinite(a)) for a in (xyz,mass,t,ref))):
        raise ValueError('finite geometry, positive masses and consistent mass-weighted vectors required')
    if not np.allclose(ref.T@ref,np.eye(ref.shape[1]),atol=1e-8,rtol=0):
        raise ValueError('orthonormal reference bath required')
    centered=xyz-np.average(xyz,axis=0,weights=mass)
    roots=np.sqrt(mass)[:,None]
    translations=[(roots*np.broadcast_to(a,xyz.shape)).ravel() for a in np.eye(3)]
    rotations=[(roots*np.cross(np.broadcast_to(a,xyz.shape),centered)).ravel() for a in np.eye(3)]
    rigid=np.column_stack(translations+rotations)
    u,s,_=np.linalg.svd(rigid,full_matrices=False)
    if s[-1]<=s[0]*1e-10:raise ValueError('nonlinear nonsingular rigid-motion frame required')
    t=t-u@(u.T@t);norm=np.linalg.norm(t)
    if norm<1e-10:raise ValueError('IRC tangent has no internal component')
    t/=norm;constraints=np.column_stack([u,t])
    raw=ref-constraints@(constraints.T@ref)
    ev,v=np.linalg.eigh(raw.T@raw)
    if ev.min()<1e-10:raise ValueError('reference bath cannot define this transverse frame')
    result=raw@((v/np.sqrt(ev))@v.T)
    return result,dict(minimum_frame_gram_eigenvalue=float(ev.min()),
                       constraint_residual=float(np.max(abs(constraints.T@result))))


class SparseTransverseHessians:
    """Interpolate matrix logs in a continuous geometry-defined bath gauge.

    Piecewise linear matrix-log interpolation avoids spline overshoot.
    Values are squared wavenumbers, not frequencies. All knot matrices must
    be positive definite. Extrapolation is forbidden. The TS can be an extra
    knot already supplied by its quartic endpoint field.
    """
    def __init__(self, positions, matrices_cm2):
        x=np.asarray(positions,float);k=np.asarray(matrices_cm2,float)
        if (x.ndim!=1 or len(x)<2 or k.ndim!=3 or k.shape[0]!=len(x)
                or k.shape[1]!=k.shape[2] or np.any(np.diff(x)<=0)
                or not np.all(np.isfinite(x)) or not np.all(np.isfinite(k))
                or not np.allclose(k,k.swapaxes(1,2),atol=1e-7,rtol=1e-12)):
            raise ValueError('ordered positions and finite symmetric Hessian matrices required')
        logs=[]
        for matrix in k:
            ev,v=np.linalg.eigh(matrix)
            if ev.min()<=0:raise ValueError('positive transverse Hessians required; no frequency repair')
            logs.append((v*np.log(ev))@v.T)
        self.positions=x.copy();self._interpolator=interp1d(x,np.array(logs),axis=0,kind="linear",bounds_error=True)

    def matrix(self, position):
        if not np.isfinite(position) or not self.positions[0]<=position<=self.positions[-1]:
            raise ValueError('outside the sparse Hessian interpolation interval')
        log=self._interpolator(float(position));ev,v=np.linalg.eigh((log+log.T)/2)
        result=(v*np.exp(ev))@v.T
        if not np.all(np.isfinite(result)):raise ValueError('nonfinite interpolated transverse Hessian')
        return result


def _symmetric_frechet(matrix, direction, function):
    """Frechet derivative of log/exp, including repeated eigenvalues."""
    ev,u=np.linalg.eigh((matrix+matrix.T)/2)
    if function=='log':
        if ev.min()<=0:raise ValueError('positive matrix required for logarithmic slope')
        values=np.log(ev);diagonal=1/ev
    elif function=='exp':values=np.exp(ev);diagonal=values
    else:raise ValueError('unsupported matrix function')
    delta=ev[:,None]-ev[None,:];difference=values[:,None]-values[None,:]
    close=abs(delta)<=1e-9*np.maximum(1.,np.maximum(abs(ev[:,None]),abs(ev[None,:])))
    divided=np.empty_like(delta)
    np.divide(difference,delta,out=divided,where=~close)
    if function=='log':divided[close]=(2/(ev[:,None]+ev[None,:]))[close]
    else:divided[close]=np.exp((ev[:,None]+ev[None,:])/2)[close]
    np.fill_diagonal(divided,diagonal)
    return u@(divided*(u.T@direction@u))@u.T


class SmoothSparseTransverseHessians(SparseTransverseHessians):
    """C1 positive interpolation, with optional exact anchor Hessian slopes.

    Slopes are dK/ds in the supplied moving bath gauge. A TS slope may be
    supplied by its cubic potential AND kinetic-metric field. Unprescribed
    slopes use nonuniform central differences of matrix logs and a scalar
    Frobenius bound; this bound is invariant under constant bath rotations.
    Only unprescribed slopes are limited. Corners are rounded in a fixed
    fraction of adjacent intervals; elsewhere interpolation is log-linear.
    No frequencies are clipped.
    """
    def __init__(self,positions,matrices_cm2,*,anchor_derivatives_cm2=None,corner_fraction=.1):
        super().__init__(positions,matrices_cm2)
        from scipy.interpolate import CubicHermiteSpline
        x=self.positions;k=np.asarray(matrices_cm2,float);logs=[]
        for matrix in k:
            ev,u=np.linalg.eigh(matrix);logs.append((u*np.log(ev))@u.T)
        logs=np.array(logs);h=np.diff(x);sec=np.diff(logs,axis=0)/h[:,None,None]
        slopes=np.empty_like(logs);slopes[0]=sec[0];slopes[-1]=sec[-1]
        for j in range(1,len(x)-1):
            slopes[j]=(h[j]*sec[j-1]+h[j-1]*sec[j])/(h[j-1]+h[j])
            bound=3*min(np.linalg.norm(sec[j-1]),np.linalg.norm(sec[j]))
            norm=np.linalg.norm(slopes[j])
            if norm>bound:slopes[j]*=bound/norm
        self.anchor_derivatives_cm2={}
        for position,derivative in (anchor_derivatives_cm2 or {}).items():
            match=np.flatnonzero(np.isclose(x,float(position),rtol=0,atol=1e-12))
            d=np.asarray(derivative,float)
            if (len(match)!=1 or d.shape!=k.shape[1:] or not np.all(np.isfinite(d))
                    or not np.allclose(d,d.T,rtol=1e-12,atol=1e-8)):
                raise ValueError('finite symmetric derivative at an acquired anchor required')
            j=int(match[0]);slopes[j]=_symmetric_frechet(k[j],d,'log')
            self.anchor_derivatives_cm2[float(x[j])]=d.copy()
        if not np.isfinite(corner_fraction) or not 0<corner_fraction<=.25:
            raise ValueError('corner fraction must be positive and at most one quarter')
        # Round the log-linear corners locally. Broad Hermite intervals can
        # create new extrema between sparse observations; outside these fixed
        # neighborhoods the certified log-linear interpolant is unchanged.
        self.corner_fraction=float(corner_fraction);pieces=[]
        for j in range(len(x)):
            width=corner_fraction*min(h[max(j-1,0)],h[min(j,len(h)-1)])
            if j>0:
                left=x[j]-width
                pieces.append((left,x[j],CubicHermiteSpline([left,x[j]],
                    [logs[j]-width*sec[j-1],logs[j]],[sec[j-1],slopes[j]],axis=0)))
            if j<len(x)-1:
                right=x[j]+width
                pieces.append((x[j],right,CubicHermiteSpline([x[j],right],
                    [logs[j],logs[j]+width*sec[j]],[slopes[j],sec[j]],axis=0)))
        def evaluate(position,nu=0):
            if not np.isfinite(position) or not x[0]<=position<=x[-1]:
                raise ValueError('outside the smooth sparse Hessian interval')
            for left,right,polynomial in pieces:
                if left<=position<=right:return polynomial(position,nu)
            j=min(int(np.searchsorted(x,position,side='right'))-1,len(x)-2)
            if nu==0:return logs[j]+(position-x[j])*sec[j]
            if nu==1:return sec[j]
            raise ValueError('only values and first directional derivatives are supported')
        self._log_spline=evaluate
        self._interpolator=evaluate

    def derivative(self,position,*,connection=None):
        """Covariant derivative of the projected dynamical Hessian.

        A = B.T dB/ds is the skew moving-frame connection. The returned
        derivative is K' + [A,K]. It includes projected-path/metric effects;
        it is not the bare potential cubic tensor with three bath indices.
        """
        k=self.matrix(position);log=self._log_spline(float(position))
        derivative=_symmetric_frechet(log,self._log_spline(float(position),1),'exp')
        if connection is not None:
            a=np.asarray(connection,float)
            if (a.shape!=k.shape or not np.all(np.isfinite(a))
                    or not np.allclose(a,-a.T,rtol=0,atol=1e-8)):
                raise ValueError('finite skew moving-frame connection required')
            derivative+=a@k-k@a
        return (derivative+derivative.T)/2

    def effective_reactive_cubic(self,position,*,reaction_length,connection=None):
        """Reactive/bath/bath effective coupling from a directional Hessian.

        Normal-coordinate scaling is reaction_length / sqrt(w_i*w_j).
        This determines one reactive index, not pure transverse cubics or
        mixed quartics. The full kinetic/constraint convention is retained.
        """
        if not np.isfinite(reaction_length) or reaction_length<=0:
            raise ValueError('positive reaction oscillator length required')
        ev,u=np.linalg.eigh(self.matrix(position));w=np.sqrt(ev)
        d=u.T@self.derivative(position,connection=connection)@u
        return reaction_length*d/np.sqrt(w[:,None]*w[None,:]),w,u


def modal_dynamical_hessian_slope(frequencies_cm, harmonic_cm, cubic_cm,
                                  metric_first_cm, direction):
    """Directional slope of sqrt(G) F sqrt(G) in an anchor modal gauge.

    At the anchor G=diag(abs(w)), F is the signed harmonic potential block.
    direction is dq_anchor/ds. Both potential cubic and kinetic-metric first
    derivatives are required. A saddle reaction axis is allowed here; it is
    removed when projecting the result into a statistical transverse bath.
    """
    w=np.asarray(frequencies_cm,float);h=np.asarray(harmonic_cm,float)
    c=np.asarray(cubic_cm,float);g=np.asarray(metric_first_cm,float);d=np.asarray(direction,float)
    n=len(w)
    if (w.ndim!=1 or np.any(w<=0) or h.shape!=(n,n) or c.shape!=(n,n,n)
            or g.shape!=(n,n,n) or d.shape!=(n,)
            or not all(np.all(np.isfinite(a)) for a in (w,h,c,g,d))
            or not np.allclose(h,h.T) or not np.allclose(g,g.swapaxes(0,1))):
        raise ValueError('finite consistent modal harmonic/cubic/metric field and direction required')
    root=np.diag(np.sqrt(w));dh=np.einsum('ijk,k->ij',c,d);dg=np.einsum('ijk,k->ij',g,d)
    droot=dg/(np.sqrt(w[:,None])+np.sqrt(w[None,:]))
    result=droot@h@root+root@dh@root+root@h@droot
    return (result+result.T)/2
