"""Plot matched fixed-photon nuclear-model profiles and an optional reference.

Only display normalization is applied: each calculated FC/FC+HT pair shares
its FC+HT origin-peak scale. Energies are shown above each calculated origin;
the underlying spectra archive retains absolute origins and cross sections.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--experiment-csv", type=Path)
    p.add_argument("--detail-experiment", action="store_true",
                   help="Also show the supplied measured trace in the detail panel")
    p.add_argument("--reference-lines-cm", nargs="*", type=float, default=[])
    p.add_argument("--detail-window-cm", nargs=2, type=float, default=[300., 650.])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with np.load(a.comparison / "spectra.npz") as source:
        data = {key: source[key] for key in source.files}
    report = json.loads((a.comparison / "result.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.9), constrained_layout=True)
    styles = {"harmonic": ("Harmonic", "#2467a6"), "vci": ("VCI", "#16866a")}
    normalized = {}
    for model, (label, color) in styles.items():
        origin = 0. if model == "harmonic" else report["vci_origin_cm"]-report["harmonic_origin_cm"]
        x = data["axis_relative_harmonic_origin_cm"] - origin
        profiles = data[model+"_profiles_Mb_per_cm"]
        scale = np.max(profiles[3, abs(x) <= 100.])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Positive resolved origin required for display normalization")
        normalized[model] = (x, profiles / scale)
        for component, suffix, linestyle in ((0, "FC", "--"), (3, "FC+HT", "-")):
            axes[0].plot(x, profiles[component] / scale, color=color, ls=linestyle,
                         lw=1.4, label=f"{label} {suffix}")
    if a.experiment_csv:
        observed = np.genfromtxt(a.experiment_csv, delimiter=",", names=True)
        axes[0].plot(observed["relative_wavenumber_cm"], observed["intensity_relative_origin"],
                     color="0.25", lw=.85, alpha=.8, label="Measured trace")
    x, profiles = normalized["vci"]
    for component, label, color, linestyle in (
        (0, "VCI FC", "#aa7733", "--"), (3, "VCI FC+HT", "#16866a", "-")):
        axes[1].plot(x, profiles[component], label=label, color=color, ls=linestyle, lw=1.6)
    mask = (x >= a.detail_window_cm[0]) & (x <= a.detail_window_cm[1])
    if not mask.any():
        raise ValueError("Detail window must contain at least one calculated spectral point")
    detail_max = float(profiles[[0, 3]][:, mask].max())
    if a.detail_experiment and a.experiment_csv:
        ex = observed["relative_wavenumber_cm"]
        ey = observed["intensity_relative_origin"]
        axes[1].plot(ex, ey, color="0.25", lw=.85, alpha=.8, label="Measured trace")
        selected = (ex >= a.detail_window_cm[0]) & (ex <= a.detail_window_cm[1])
        if selected.any():
            detail_max = max(detail_max, float(ey[selected].max()))
    axes[0].set_xlim(float(data["axis_relative_harmonic_origin_cm"][0]),
                     float(data["axis_relative_harmonic_origin_cm"][-1]))
    axes[0].set_ylim(bottom=0)
    axes[1].set_xlim(*a.detail_window_cm)
    axes[1].set_ylim(0, max(1e-5, 1.25*detail_max))
    for ax, tag in zip(axes, ("(a)", "(b)")):
        for line in a.reference_lines_cm:
            if ax.get_xlim()[0] <= line <= ax.get_xlim()[1]:
                ax.axvline(line, color="0.65", lw=.8, ls=":", zorder=0)
        ax.text(-.065, 1.02, tag, transform=ax.transAxes, va="bottom", fontweight="bold")
        ax.set_xlabel("Vibrational energy above each origin / cm$^{-1}$")
        ax.set_ylabel("Relative profile")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output.with_suffix(".pdf"))
    fig.savefig(a.output.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
