"""Preserve angular labels in the resident positive statistical convolution.

Supplied internal spectra exclude overall rotation. Rotor weights already
contain magnetic multiplicities and symmetry divisors; no factors are added.
"""
import numpy as np
from .statistical_convolution import convolve_discrete_measure


def convolve_resolved_rotation(internal_counts, bin_width_cm, energies_cm,
                              weights, angular_momenta, *, maximum_bins=None):
    e=np.asarray(energies_cm,float);w=np.asarray(weights,float);j=np.asarray(angular_momenta,float)
    if (j.shape!=e.shape or w.shape!=e.shape or j.ndim!=1 or not len(j)
            or not np.all(np.isfinite(j)) or np.any(j<0)
            or not np.allclose(j*2,np.rint(j*2),atol=1e-12,rtol=0)):
        raise ValueError('one explicit physical J label per unsummed rotational level required')
    labels=np.unique(j)
    rows=[]
    for label in labels:
        selected=j==label
        row=convolve_discrete_measure(internal_counts,bin_width_cm,e[selected],w[selected],algorithm='positive_direct')
        rows.append(row if maximum_bins is None else row[:maximum_bins])
    length=max(map(len,rows)) if maximum_bins is None else maximum_bins
    result=np.zeros((len(labels),length))
    for index,row in enumerate(rows):result[index,:len(row)]=row
    return labels,result


def assemble_internal_counts(energies_cm, weights, *, bin_width_cm, maximum_energy_cm,
                             tail_anchor_energies_cm=(), tail_gaps_cm=None,
                             extra_spectra=()):
    """Existing finite states, positive harmonic tails and discrete LAM spectra.

    Each harmonic tail is a supplied disjoint occupation cone. Its anchor is
    excluded from the cone, since it is already included among finite states.
    No high-order polynomial is extrapolated to generate missing levels.
    """
    if not np.isfinite(maximum_energy_cm) or maximum_energy_cm<=0:
        raise ValueError('positive finite energy domain required')
    count=int(maximum_energy_cm/bin_width_cm)+1
    if count>10_000_000:raise MemoryError('state grid exceeds ten million bins')
    finite=convolve_discrete_measure(np.array([1.]),bin_width_cm,energies_cm,weights,algorithm='positive_direct')
    density=np.zeros(count);density[:min(count,len(finite))]=finite[:count]
    anchors=np.asarray(tail_anchor_energies_cm,float)
    if anchors.size:
        gaps=np.asarray(tail_gaps_cm,float)
        if (anchors.ndim!=1 or gaps.ndim!=2 or len(gaps)!=len(anchors)
                or not np.all(np.isfinite(anchors)) or np.any(anchors<0)
                or np.any(np.isnan(gaps)) or np.any(gaps<=0)):
            raise ValueError('nonnegative anchors and positive supplied tail gaps required')
        for anchor,gaplist in zip(anchors,gaps):
            if anchor>maximum_energy_cm:continue
            kernel=np.array([1.])
            for gap in gaplist[np.isfinite(gaplist)]:
                levels=gap*np.arange(int((maximum_energy_cm-anchor)/gap)+1)
                kernel=convolve_discrete_measure(kernel,bin_width_cm,levels,np.ones(len(levels)),algorithm='positive_direct')[:count]
            kernel[0]-=1.
            kernel=np.maximum(kernel,0.)
            shifted=convolve_discrete_measure(kernel,bin_width_cm,np.array([anchor]),np.ones(1),algorithm='positive_direct')
            density[:min(len(shifted),count)]+=shifted[:count]
    for spectrum,weight in extra_spectra:
        spectrum=np.asarray(spectrum,float);weight=np.asarray(weight,float)
        active=spectrum<=maximum_energy_cm
        density=convolve_discrete_measure(density,bin_width_cm,spectrum[active],weight[active],algorithm='positive_direct')[:count]
    return density
