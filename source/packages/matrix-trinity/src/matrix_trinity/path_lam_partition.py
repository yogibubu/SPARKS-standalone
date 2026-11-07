"""Periodic LAM/SAM/rotation free-energy surfaces from finite path anchors.

Existing analytic fitting and periodic DVR owners are reused. This is an
adiabatic scalar conditional trace, not coupled-channel tunnelling dynamics.
No electronic or PES evaluator is accepted by this interface.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.special import logsumexp
from matrix_dvr.analytic_surface import fit_analytic_surface
from matrix_dvr.msr import periodic_rotor_levels


class PeriodicLAMPath:
    def __init__(self,positions,angles_rad,potential_cm,inertias_amu_A2,*,
                 temperatures_K,log_sam_partition=None,log_rotational_weight=None,
                 symmetry_number=1,degree=6,held_out_indices=(),max_fit_error_cm=10.):
        s=np.asarray(positions,float);phi=np.asarray(angles_rad,float);v=np.asarray(potential_cm,float)
        inertia=np.asarray(inertias_amu_A2,float);t=np.asarray(temperatures_K,float)
        if (s.ndim!=1 or len(s)<2 or np.any(np.diff(s)<=0) or phi.ndim!=1 or np.any(np.diff(phi)<=0)
                or v.shape!=(len(s),len(phi)) or inertia.shape!=v.shape or np.any(inertia<=0)
                or t.ndim!=1 or not len(t) or np.any(t<=0) or symmetry_number<1
                or not all(np.all(np.isfinite(a)) for a in (s,phi,v,inertia,t))):
            raise ValueError('ordered path anchors and finite positive periodic metric/temperatures required')
        shape=v.shape+(len(t),)
        sam=np.zeros(shape) if log_sam_partition is None else np.asarray(log_sam_partition,float)
        rot=np.zeros(v.shape) if log_rotational_weight is None else np.asarray(log_rotational_weight,float)
        if sam.shape!=shape or rot.shape!=v.shape or not np.all(np.isfinite(sam)) or not np.all(np.isfinite(rot)):
            raise ValueError('matching conditional SAM and rotational log weights required')
        self.positions=s.copy();self.temperatures=t.copy();self.beta=1.4387768775039338/t
        self.symmetry_number=int(symmetry_number);self.fits=[];self.metric_fits=[];self.fit_controls=[]
        mask=np.ones(len(phi),bool);mask[list(held_out_indices)]=False
        if not np.any(mask) or np.count_nonzero(mask)<2*degree+1:raise ValueError('too few prescribed samples for fit validation')
        for a in range(len(s)):
            effective=v[a,:,None]-(sam[a]-sam[a,0])/self.beta-(rot[a]-rot[a,0])[:,None]/self.beta
            fits=[];errors=[]
            for j in range(len(t)):
                trial=fit_analytic_surface(phi[mask],effective[mask,j],degree=degree,family='fourier')
                error=float(max(abs(trial.evaluate(phi[~mask])-effective[~mask,j]),default=0.))
                if error>max_fit_error_cm:raise ValueError('finite conditional potential fit exceeds its prescribed tolerance')
                fits.append(fit_analytic_surface(phi,effective[:,j],degree=degree,family='fourier'));errors.append(error)
            self.fits.append(fits);self.fit_controls.append(errors)
            self.metric_fits.append(fit_analytic_surface(phi,np.log(inertia[a]),degree=degree,family='fourier'))

    def _interval(self,position):
        if not np.isfinite(position) or not self.positions[0]<=position<=self.positions[-1]:
            raise ValueError('LAM path extrapolation is not qualified')
        j=min(max(int(np.searchsorted(self.positions,position,side='right'))-1,0),len(self.positions)-2)
        t=(position-self.positions[j])/(self.positions[j+1]-self.positions[j])
        return j,float(t)

    def partition(self,position,*,basis_size=121,analytic_grid_size=2048):
        j,t=self._interval(position);phi=np.linspace(0.,2*np.pi,analytic_grid_size+1)
        logI=(1-t)*self.metric_fits[j].evaluate(phi)+t*self.metric_fits[j+1].evaluate(phi)
        inertia=np.exp(logI)
        arc=cumulative_trapezoid(np.sqrt(inertia),phi,initial=0.)
        length=float(arc[-1]);moment=(length/(2*np.pi))**2
        theta=np.linspace(0.,2*np.pi,analytic_grid_size,endpoint=False)
        source_phi=PchipInterpolator(2*np.pi*arc/length,phi)(theta)
        levels=[];logs=[]
        for k,beta in enumerate(self.beta):
            value=(1-t)*self.fits[j][k].evaluate(source_phi)+t*self.fits[j+1][k].evaluate(source_phi)
            # The native rotor uses a zero-minimum potential internally.
            # Restore that known offset so absolute LAM ZPE is counted once.
            e=periodic_rotor_levels(moment,1,float(np.ptp(value)),phi_rad=theta,potential_cm=value,basis_size=basis_size)+float(np.min(value))
            levels.append(e);logs.append(float(logsumexp(-beta*e)-np.log(self.symmetry_number)))
        return dict(log_partition=np.array(logs),levels_cm=np.array(levels),
            effective_moment_amu_A2=moment,variable_metric_arclength=True,
            absolute_potential_offset_restored=True,symmetry_divisor_applied_once=self.symmetry_number,
            scope='scalar adiabatic conditional SAM trace with periodic LAM DVR and supplied rotational weights')
