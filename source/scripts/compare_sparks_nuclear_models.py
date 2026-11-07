"""Matched harmonic/selected-VCI FC and FC+HT spectra at fixed photon energy.

Reuse portable nuclear moments from evaluate_sparks_variational_window.py.
The continuum table interpolates complex amplitudes, before squaring, and
every transition has its own electron kinetic energy. No energy or intensity
fit is applied. Continuum, nuclear support and interpolation controls remain
separate numerical checks.
"""
import argparse
import json
from pathlib import Path
import socket
from time import perf_counter
import numpy as np
from matrix_sparks.execution import require_compute_host
from matrix_sparks import harmonic_photoionization_spectrum, cached_harmonic_overlap_matrix
from matrix_sparks.photoelectron import ContinuumDipoleTable, fixed_photon_photoelectron_lines, gaussian_line_profiles
from matrix_trinity.vibronic_harmonic import HarmonicVibronicModel
from matrix_gf.harmonic import CM_PER_HARTREE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nuclear-window", type=Path, required=True)
    p.add_argument("--amplitudes", type=Path, required=True)
    p.add_argument("--equilibrium-amplitudes", type=Path, required=True)
    p.add_argument("--photon-energy-ev", type=float, required=True)
    p.add_argument("--channel-multiplicity", type=float, required=True)
    p.add_argument("--sigma-cm", type=float, default=50.)
    p.add_argument("--target-fraction", type=float, default=.99)
    p.add_argument("--maximum-transitions", type=int, default=200000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--nuclear-overlap-cache", type=Path,
                   help="Reusable nuclear overlaps; default: inside the saved nuclear window")
    p.add_argument("--maximum-kernel-entries", type=int, default=1048576)
    a = p.parse_args()
    host = require_compute_host()
    if a.output.exists():
        raise FileExistsError(a.output)
    if not np.isfinite(a.sigma_cm) or a.sigma_cm <= 0:
        raise ValueError("Positive finite Gaussian linewidth required")
    with np.load(a.nuclear_window / "spectrum.npz") as data:
        saved = {key: data[key] for key in data.files}
    wi, wf = saved["initial_frequencies_cm"], saved["final_frequencies_cm"]
    n = len(wi)
    model = HarmonicVibronicModel(wi, wf, saved["mode_map"],
        saved["final_shift_dimensionless"], float(saved["electronic_gap_cm"]),
        np.zeros(3), np.zeros((3, n)))
    with np.load(a.equilibrium_amplitudes / "center.npz") as data:
        constants, energies = data["amplitudes"], data["energies"]
    gradients = []
    for mode in range(n):
        with np.load(a.amplitudes / f"derivative-mode{mode+1:02d}.npz") as data:
            if not np.array_equal(data["energies"], energies):
                raise ValueError("Continuum energy grids differ")
            gradients.append(data["first"])
    gradients = np.einsum("escm,mn->escn", np.stack(gradients, axis=-1),
                          saved["normal_transport"])
    if gradients.shape != constants.shape + (n,):
        raise ValueError("Equilibrium and derivative continuum channel spaces differ")
    table = ContinuumDipoleTable(energies,
        np.concatenate((constants[..., None], gradients), axis=-1))
    ti = harmonic_photoionization_spectrum(model, constants, gradients,
        electron_energies_hartree=energies, method="TI", selection="adaptive",
        target_fraction=a.target_fraction, maximum_states=a.maximum_transitions,
        maximum_transitions=a.maximum_transitions, maximum_integrals=4000000,
        maximum_final_quanta=32, workers=1)
    quanta = np.asarray(ti["final_quanta"], int)
    binding = np.asarray(ti["wavenumbers_cm"]) / CM_PER_HARTREE
    axis = saved["axis_relative_harmonic_origin_cm"]
    keep = (binding * CM_PER_HARTREE - model.origin_cm <= axis[-1] + 6*a.sigma_cm)
    quanta, binding = quanta[keep], binding[keep]
    initial = np.vstack((np.zeros((1, n), int), np.eye(n, dtype=int)))
    tick = perf_counter()
    overlaps, overlap_reuse = cached_harmonic_overlap_matrix(model, initial, quanta,
        cache_directory=a.nuclear_overlap_cache or a.nuclear_window / 'harmonic-overlaps',
        maximum_grid_entries=128000000, maximum_harmonic_pairs=32000000)
    overlap_seconds = perf_counter() - tick
    moments = overlaps.copy()
    moments[:, 1:] /= np.sqrt(2.)
    moment_errors = []
    for index, (c, h) in enumerate(zip(constants, gradients)):
        amplitudes = (moments[:, 0, None, None] * c[None]
                      + np.einsum("lm,scm->lsc", moments[:, 1:], h))
        reconstructed = np.sum(abs(amplitudes)**2, axis=(1, 2))
        reference = np.asarray(ti["strengths"])[index, 3, keep]
        error = float(np.max(abs(reconstructed-reference)) / max(float(np.max(reference)), 1e-300))
        if error > 1e-9:
            raise ValueError("Exported harmonic moments and native TI intensities disagree")
        moment_errors.append(error)
    photon = np.array([a.photon_energy_ev / 27.211386245988])
    outputs = {}
    profile_seconds = {}
    for name, b, m in (("harmonic", binding, moments),
                       ("vci", saved["binding_energies_hartree"], saved["nuclear_moments"])):
        result = fixed_photon_photoelectron_lines(photon, b, m, table,
            channel_multiplicity=a.channel_multiplicity)
        lines = b*CM_PER_HARTREE - model.origin_cm
        tick = perf_counter()
        profiles = gaussian_line_profiles(axis, lines, result["cross_sections_Mb"][0],
            sigma=a.sigma_cm, maximum_kernel_entries=a.maximum_kernel_entries)
        profile_seconds[name] = perf_counter() - tick
        outputs[name] = dict(lines_relative_harmonic_origin_cm=lines,
            cross_sections_Mb=result["cross_sections_Mb"][0],
            profiles_Mb_per_cm=profiles)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / "spectra.npz", axis_relative_harmonic_origin_cm=axis,
        harmonic_final_quanta=quanta, harmonic_moments=moments,
        **{name+"_"+key: value for name, row in outputs.items() for key, value in row.items()})
    assignments = []
    for i, line in enumerate(outputs["vci"]["lines_relative_harmonic_origin_cm"]):
        weights = abs(saved["final_vectors"][:, i])**2
        dominant = np.argsort(weights)[-3:][::-1]
        assignments.append(dict(relative_cm=float(line),
            cross_sections_Mb=outputs["vci"]["cross_sections_Mb"][:, i].tolist(),
            leading_components=[dict(quanta=saved["final_basis"][k].tolist(),
                                     weight=float(weights[k])) for k in dominant]))
    rules = np.asarray(ti["sum_rules"])
    fractions = np.asarray(ti["strengths"])[:, 3].sum(axis=-1) / rules[:, 3]
    report = dict(host=host, photon_energy_ev=a.photon_energy_ev,
        channel_multiplicity=a.channel_multiplicity, sigma_cm=a.sigma_cm,
        harmonic_origin_cm=model.origin_cm,
        vci_origin_cm=float(saved["binding_energies_hartree"][0]*CM_PER_HARTREE),
        harmonic_global_selected_transitions=len(ti["wavenumbers_cm"]),
        harmonic_conditional_captured_fractions=fractions.tolist(),
        harmonic_moment_TI_relative_errors=moment_errors,
        nuclear_overlap_reuse=overlap_reuse,
        nuclear_overlap_seconds=overlap_seconds,
        profile_seconds=profile_seconds,
        maximum_profile_kernel_entries=a.maximum_kernel_entries,
        conditional_electron_energies_hartree=energies.tolist(),
        vci_scope="saved selected-VCI low-energy window, not a global intensity completeness claim",
        energy_alignment="calculated origins; no shift or intensity rescaling",
        assignments=assignments)
    (a.output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "assignments"}, indent=2))


if __name__ == "__main__":
    main()
