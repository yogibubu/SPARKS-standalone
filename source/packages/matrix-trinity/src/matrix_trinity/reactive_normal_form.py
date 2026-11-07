"""Reaction-only normal form for explicit quantization of the stable sector."""
from dataclasses import replace
import numpy as np
from .transition_state import TransitionStateField, solve_transition_state_normal_form


def reaction_only_normal_form(field, **resources):
    """Eliminate reactive imbalance while retaining every stable coupling.

    The existing Weyl engine is used unchanged. For polynomials of degree at
    most four, 4*max(|omega|) bounds every stable homological frequency. The
    corresponding finite resonance window therefore retains all monomials
    commuting with reactive action. No stable-mode VPT2 elimination precedes
    a subsequent DVR/variational treatment of those same couplings.
    """
    if not isinstance(field, TransitionStateField):
        raise TypeError('TransitionStateField required')
    window = float(np.nextafter(4*np.max(abs(field.signed_frequencies_cm)), np.inf))
    if 'resonance_window_cm' in resources:
        raise ValueError('reaction-only partition fixes its window by the polynomial degree')
    result = solve_transition_state_normal_form(field, resonance_window_cm=window, **resources)
    r = field.reaction_mode
    if any(k[2*r] != k[2*r+1] for k in result.polynomial):
        raise ArithmeticError('reaction-only normal form failed its action commutator')
    return replace(result, diagnostics=dict(result.diagnostics,
        partition='reactive_imbalance_only', all_stable_couplings_retained=True,
        stable_couplings_already_perturbatively_eliminated=False))
