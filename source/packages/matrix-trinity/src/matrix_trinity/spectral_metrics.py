"""Shape distances for nonnegative spectra on a declared physical axis.

Each profile is interpreted as a piecewise-linear density and normalized
to unit area on the supplied window. No shift or scale parameter is fitted.
Wasserstein distance has the axis units and is coordinate-dependent. Signed
ECD or interference spectra must not be passed as probability densities.
"""
import numpy as np


def spectral_profile_distances(axis, reference, candidate):
    """Return overlap coefficient and exact piecewise-linear 1D Wasserstein.

    Inputs share a finite, strictly increasing axis with at least two points.
    Absolute areas are returned so shape normalization remains explicit.
    """
    x, a, b = (np.asarray(v, dtype=float) for v in (axis, reference, candidate))
    if (x.ndim != 1 or len(x) < 2 or a.shape != x.shape or b.shape != x.shape
            or not all(np.isfinite(v).all() for v in (x, a, b))
            or np.any(np.diff(x) <= 0) or np.any(a < 0) or np.any(b < 0)):
        raise ValueError('Finite nonnegative profiles on one strictly increasing axis required')
    h = np.diff(x)
    areas = [float(np.sum(h*(v[:-1]+v[1:])*.5)) for v in (a, b)]
    if not all(np.isfinite(v) and v > 0 for v in areas):
        raise ValueError('Both profile areas must be finite and positive')
    d = a/areas[0]-b/areas[1]
    left, right = d[:-1], d[1:]
    # Exact integral of a linear absolute difference, including its crossings.
    magnitude = .5*h*(np.abs(left)+np.abs(right))
    cross = (left*right < 0)
    magnitude[cross] = (.5*h[cross]*(left[cross]**2+right[cross]**2)
                        /(np.abs(left[cross])+np.abs(right[cross])))
    overlap = float(np.clip(1.-.5*np.sum(magnitude), 0., 1.))
    cdf = np.r_[0., np.cumsum(.5*h*(left+right))]
    distance = 0.
    for width, dl, dr, initial in zip(h, left, right, cdf[:-1]):
        # CDF difference in u=(x-x_i)/h is c0+c1*u+c2*u**2.
        c0, c1, c2 = float(initial), float(width*dl), float(.5*width*(dr-dl))
        coefficients = [c2, c1, c0] if c2 else [c1, c0]
        roots = np.roots(coefficients) if c2 or c1 else []
        cuts = [0., *sorted(float(z.real) for z in roots
                           if abs(z.imag) < 1e-12 and 0. < z.real < 1.), 1.]
        primitive = lambda u: c0*u+.5*c1*u*u+(c2/3.)*u*u*u
        distance += width*sum(abs(primitive(v)-primitive(u))
                              for u, v in zip(cuts[:-1], cuts[1:]))
    return dict(overlap_coefficient=overlap, wasserstein_1=float(distance),
                reference_area=areas[0], candidate_area=areas[1],
                axis_window=[float(x[0]), float(x[-1])],
                normalization='Unit area on the declared axis/window; no alignment fitted',
                interpolation='Piecewise-linear density; exact absolute CDF integration')


def spectral_window_convergence(axis, reference, candidate, *, position_tolerance=3.,
                                band_strength_tolerance=.05, profile_tolerance=.02,
                                minimum_band_prominence=.01):
    """Observable stopping test on one fixed, positive, broadened window.

    No spectral shift or intensity rescaling is fitted. Reference maxima and
    intervening minima define fixed groups, so intensity exchange inside an
    unresolved resonance is tested collectively. This compares two supplied
    spectra; it supplies no omitted-state or physical-model error bound.
    """
    from scipy.signal import find_peaks
    x,a,b=(np.asarray(v,float) for v in (axis,reference,candidate))
    distances=spectral_profile_distances(x,a,b)
    for value in (position_tolerance,band_strength_tolerance,profile_tolerance,minimum_band_prominence):
        if not np.isfinite(value) or value<=0:
            raise ValueError('positive finite observable tolerances required')
    area=distances['reference_area']
    integrate=lambda y,z: float(np.sum(np.diff(z)*(y[:-1]+y[1:])*.5))
    l1=integrate(abs(a-b),x)/area
    strength_change=abs(distances['candidate_area']-area)/area
    peaks=find_peaks(a,prominence=minimum_band_prominence*a.max())[0].tolist()
    if a[0]>=a[1] and a[0]>=minimum_band_prominence*a.max():peaks.insert(0,0)
    if a[-1]>=a[-2] and a[-1]>=minimum_band_prominence*a.max():peaks.append(len(a)-1)
    if not peaks:peaks=[int(np.argmax(a))]
    cuts=[0,*[left+int(np.argmin(a[left:right+1])) for left,right in zip(peaks[:-1],peaks[1:])],len(x)-1]
    bands=[]
    for left,right in zip(cuts[:-1],cuts[1:]):
        if right<=left:continue
        take=slice(left,right+1)
        ar=integrate(a[take],x[take]);br=integrate(b[take],x[take])
        # Test every resolved reference group with at least one percent of the
        # window strength; weak groups remain in the absolute profile test.
        if ar<minimum_band_prominence*area:continue
        pa=float(x[left+int(np.argmax(a[take]))]);pb=float(x[left+int(np.argmax(b[take]))])
        ds=abs(br-ar)/ar
        bands.append(dict(window=[float(x[left]),float(x[right])],reference_peak=pa,
            candidate_peak=pb,peak_change=abs(pb-pa),relative_strength_change=ds,
            passed=abs(pb-pa)<=position_tolerance and ds<=band_strength_tolerance))
    passed=(l1<=profile_tolerance and strength_change<=band_strength_tolerance
            and bool(bands) and all(row['passed'] for row in bands))
    return dict(passed=bool(passed),relative_absolute_profile_L1=l1,
        relative_window_strength_change=strength_change,bands=bands,
        tolerances=dict(position=position_tolerance,band_strength=band_strength_tolerance,
                        absolute_profile_L1=profile_tolerance),
        shape_distances=distances,energy_shift_fitted=False,intensity_rescaled=False,
        scope='Stability of the supplied positive broadened spectrum on its fixed window; no completeness bound')
