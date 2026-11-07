"""Finite energy-only inverse-distance stretch completion.

The positive squared bond-order form follows the Morse/long-range family
(Le Roy and Henderson, DOI: 10.1080/00268970701241656).
The expansion variable y=(R-Re)/(R+Re) is rational in inverse distance.
Local quadratic/cubic/quartic derivatives and the leading physical tail are
constraints; only two or three higher coefficients use additional energies.
Sampling and fitting have no electronic backend or adaptive acquisition.
"""
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
import numpy as np
from numpy.polynomial import Polynomial
from .radial import RadialDerivatives


@dataclass(frozen=True)
class SparseRadialPlan:
    """Immutable additional-energy budget; indices are zero based."""
    r0: float
    curvature: float
    depth: float
    left: float
    right: float
    sampling_power: int
    sample_count: int
    radii: tuple[float, ...]
    fit_indices: tuple[int, ...]
    check_indices: tuple[int, ...]

    def __post_init__(self):
        if type(self.sample_count) is not int or self.sample_count not in (3,5):
            raise ValueError('three or five observations required')
        if type(self.sampling_power) is not int or self.sampling_power not in (4,6):
            raise ValueError('sampling power four or six required')
        scalars=(self.r0,self.curvature,self.depth,self.left,self.right)
        if not all(isfinite(x) and x>0 for x in scalars) or not self.r0<self.left<self.right:
            raise ValueError('positive local inputs and ordered outer interval required')
        if (tuple(self.fit_indices)!=tuple(range(0,self.sample_count,2)) or
                tuple(self.check_indices)!=tuple(range(1,self.sample_count,2))):
            raise ValueError('fixed disjoint fitting and checking assignments required')
        radii=tuple(float(r) for r in self.radii)
        if len(radii)!=self.sample_count or not all(isfinite(r) and self.left<r<self.right for r in radii):
            raise ValueError('declared point count and finite interval radii required')
        xa,xb=(self.r0/self.left)**self.sampling_power,(self.r0/self.right)**self.sampling_power
        expected=[(1-j/(self.sample_count+1))*xa+j/(self.sample_count+1)*xb for j in range(1,self.sample_count+1)]
        if not np.allclose([(self.r0/r)**self.sampling_power for r in radii],expected,rtol=1e-12,atol=0):
            raise ValueError('points must be equispaced in the declared inverse power')
        object.__setattr__(self,'radii',radii)
        object.__setattr__(self,'fit_indices',tuple(self.fit_indices))
        object.__setattr__(self,'check_indices',tuple(self.check_indices))


def plan_sparse_inverse_radial(r0, curvature, depth, *, sampling_power=6,
                               sample_count=5, interval_scales=(.5, 4.)):
    """Choose all points before observing any extra energy.

    ell=sqrt(2D/k2), interval=Re+[a,b]*ell. Dimensionless a,b are declared
    inputs (default .5,4); points are uniform in (Re/R)^n inside the interval.
    N=3: two fitting energies and one check; N=5: three fits and two checks.
    """
    values=tuple(float(x) for x in (r0,curvature,depth,*interval_scales))
    if not all(isfinite(x) for x in values) or min(values[:3])<=0:
        raise ValueError('finite positive reference, curvature and depth required')
    if len(interval_scales)!=2 or not 0<interval_scales[0]<interval_scales[1]:
        raise ValueError('two ordered positive interval scales required')
    if type(sample_count) is not int or sample_count not in (3,5):
        raise ValueError('the fixed additional energy budget must be three or five')
    if type(sampling_power) is not int or sampling_power not in (4,6):
        raise ValueError('sampling power must be four or six')
    r0,k2,depth=values[:3];ell=sqrt(2*depth/k2)
    left,right=(r0+float(a)*ell for a in interval_scales)
    xa,xb=(r0/left)**sampling_power,(r0/right)**sampling_power
    fractions=np.arange(1,sample_count+1)/(sample_count+1)
    radii=tuple(float(r0/((1-t)*xa+t*xb)**(1/sampling_power)) for t in fractions)
    return SparseRadialPlan(r0,k2,depth,left,right,sampling_power,sample_count,radii,
        tuple(range(0,sample_count,2)),tuple(range(1,sample_count,2)))


@dataclass(frozen=True)
class SparseInverseDistancePotential:
    """MLR form with a single inverse-power attractive tail.

    V=D[1-b(R)]^2, b=(Re/R)^m exp[-gamma(y)], y=(R-Re)/(R+Re).
    gamma=y[beta_inf*y+(1-y)sum beta_i*y^i].  gamma(1)=beta_inf
    enforces V=D-C_m/R^m+o(R^-m). No continuum states are supplied.
    """
    depth: float
    r0: float
    tail_power: int
    tail_coefficient: float
    beta_coefficients: tuple[float, ...]

    def __post_init__(self):
        if not all(isfinite(x) and x>0 for x in (self.depth,self.r0,self.tail_coefficient)):
            raise ValueError('positive finite depth, reference and tail coefficient required')
        if type(self.tail_power) is not int or self.tail_power not in (4,6):
            raise ValueError('physical tail power must be four or six')
        object.__setattr__(self,'beta_coefficients',tuple(float(x) for x in self.beta_coefficients))
        if not self.beta_coefficients or not all(isfinite(x) for x in self.beta_coefficients):
            raise ValueError('finite exponent coefficients required')

    @property
    def beta_infinity(self):
        return log(2*self.depth*self.r0**self.tail_power/self.tail_coefficient)

    @property
    def gamma(self):
        y=Polynomial([0.,1.])
        return y*(self.beta_infinity*y+(1-y)*Polynomial(self.beta_coefficients))

    def derivatives(self, distance):
        r=float(distance)
        if not isfinite(r) or r<=0:raise ValueError('positive finite radial distance required')
        re=self.r0;y=(r-re)/(r+re);g=self.gamma
        y1,y2,y3=2*re/(r+re)**2,-4*re/(r+re)**3,12*re/(r+re)**4
        m=self.tail_power;l=-m*log(r/re)-g(y)
        b=exp(l);l1=-m/r-g.deriv(1)(y)*y1
        l2=m/r**2-g.deriv(2)(y)*y1*y1-g.deriv(1)(y)*y2
        l3=-2*m/r**3-g.deriv(3)(y)*y1**3-3*g.deriv(2)(y)*y1*y2-g.deriv(1)(y)*y3
        b1=b*l1;b2=b*(l1*l1+l2);b3=b*(l1**3+3*l1*l2+l3)
        s=1-b;d=self.depth
        return RadialDerivatives(d*s*s,-2*d*s*b1,
            2*d*(b1*b1-s*b2),2*d*(3*b1*b2-s*b3))

    def monotonicity_certificate(self):
        """Polynomial test for strictly decreasing b on all R>0.

        d(log b)/dy=-S(y)/(1-y^2). S>0 on (-1,1) implies a unique
        potential minimum at Re, with repulsive inner and attractive outer
        branches. Evaluate endpoints and every real stationary point of S.
        """
        y=Polynomial([0.,1.]);s=2*self.tail_power+(1-y*y)*self.gamma.deriv()
        candidates=[-1.,1.]
        candidates.extend(float(z.real) for z in s.deriv().roots()
            if abs(z.imag)<1e-8 and -1<z.real<1)
        minimum=min(float(s(t)) for t in candidates)
        return dict(passed=minimum>1e-8,minimum_polynomial_value=minimum,
                    candidate_y=candidates,scope='all R>0; floating polynomial root evaluation')


def fit_sparse_inverse_radial(plan, energies, *, depth, curvature, cubic,
                              quartic, tail_coefficient, tail_power=6):
    """Fit only the predeclared fitting indices; never use check energies.

    Energies are relative to the minimum. Held-out energies are consulted
    only after coefficients are fixed. Physical asymptotes are external
    inputs with separate provenance; this routine never estimates them.
    """
    if not isinstance(plan,SparseRadialPlan):raise TypeError('a frozen sparse radial plan is required')
    e=np.asarray(energies,float)
    if e.shape!=(plan.sample_count,) or not np.all(np.isfinite(e)):
        raise ValueError('exactly the declared finite energy set is required')
    if not all(isfinite(x) for x in (depth,curvature,cubic,quartic,tail_coefficient)) or min(depth,curvature,tail_coefficient)<=0:
        raise ValueError('finite local derivatives and positive physical scales required')
    if not np.allclose([depth,curvature],[plan.depth,plan.curvature],rtol=1e-12,atol=0):
        raise ValueError('fit must retain the local scales used to freeze point positions')
    r0=plan.r0;m=tail_power
    s1=sqrt(curvature*r0*r0/(2*depth))
    s2=cubic*r0**3/(12*depth*s1)
    s3=(quartic*r0**4/(24*depth)-s2*s2)/(2*s1)
    l1=-s1;l2=-s2-.5*s1*s1;l3=-s3-s1*s2-s1**3/3
    bi=log(2*depth*r0**m/tail_coefficient)
    b0=-2*(l1+m);b1=2*m+2*b0-bi-4*l2
    b2=-8*m/3-3*b0+2*bi+3*b1-8*l3
    r=np.asarray(plan.radii)[list(plan.fit_indices)];v=e[list(plan.fit_indices)]
    if np.any(v<=0) or np.any(v>=depth):
        raise ValueError('fit energies must belong to the bound outer branch')
    y=(r-r0)/(r+r0);bond=1-np.sqrt(v/depth)
    target=(-np.log(bond)-m*np.log(r/r0))/y
    residual=(target-bi*y)/(1-y)-(b0+b1*y+b2*y*y)
    count=len(plan.fit_indices);A=np.column_stack([y**j for j in range(3,3+count)])
    condition=float(np.linalg.cond(A))
    if not isfinite(condition) or condition>1e10:raise ValueError('ill-conditioned fixed sample fit')
    higher=np.linalg.solve(A,residual)
    # Monotonicity is a physical fit constraint, never a posteriori clipping
    # of energies or removal of states. The constraint is linear in the
    # exponent coefficients. No held-out energy enters this optimization.
    from scipy.optimize import LinearConstraint,minimize
    ypoly=Polynomial([0.,1.])
    base_model=SparseInverseDistancePotential(float(depth),r0,tail_power,float(tail_coefficient),(b0,b1,b2))
    nodes=np.cos(np.linspace(0,np.pi,257))
    base_s=2*m+(1-ypoly*ypoly)*base_model.gamma.deriv()
    shape_matrix=np.column_stack([((1-ypoly*ypoly)*(ypoly**(j+1)*(1-ypoly)).deriv())(nodes)
        for j in range(3,3+count)])
    inverse=np.linalg.inv(A)
    constraint=LinearConstraint(shape_matrix@inverse,.05-base_s(nodes),np.inf)
    def objective(z):return .5*float(np.sum((z-residual)**2))
    def gradient(z):return z-residual
    fit_status='unconstrained solution satisfies shape'
    if np.any(shape_matrix@higher+base_s(nodes)<.05):
        solved=minimize(objective,A@higher,jac=gradient,constraints=[constraint],
            method='SLSQP',options=dict(ftol=1e-12,maxiter=1000))
        if not solved.success:raise ValueError('shape-constrained fit did not converge: '+solved.message)
        higher=inverse@solved.x
        fit_status='shape-constrained least squares'
    model=SparseInverseDistancePotential(float(depth),r0,tail_power,float(tail_coefficient),tuple([b0,b1,b2,*higher]))
    prediction=np.array([model.derivatives(r).energy for r in plan.radii])
    return model,dict(fit_indices=plan.fit_indices,check_indices=plan.check_indices,
        residuals=tuple(float(x) for x in prediction-e),matrix_condition=condition,fit_status=fit_status,
        maximum_held_out_error=float(max(abs(prediction[i]-e[i]) for i in plan.check_indices)),
        shape_certificate=model.monotonicity_certificate())
