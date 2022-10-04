import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
import scipy.stats as ss

import sys

sys.path.append("..")
import panco2 as p2

mcmc_params = {
    "n_chains": 30,
    "max_steps": 1e5,
    "n_threads": 4,
    "n_check": 1e3,
    "max_delta_tau": 0.02,
    "min_autocorr_times": 20,
    "n_burn": 500,
    "discard": 20,
    "clip_percent": 20.0,
}
instruments = {
    "Planck": {
        "name": "Planck",
        "beam": 600.0,
        "map_size": 240.0,
        "conv": (1.0, 0.05),
        "zero": (0.0, 1e-6),
        "cbar_fact": 1e5,
        "cbar_label": "Compton $y \\times 10^5$",
        "cmap": p2.utils.get_planck_cmap(),
    },
    "SPT": {
        "name": "SPT",
        "beam": 75.0,
        "map_size": 60.0,
        "conv": (1.0, 0.05),
        "zero": (0.0, 1e-6),
        "cbar_fact": 1e5,
        "cbar_label": "Compton $y \\times 10^5$",
        "cmap": "twilight_shifted",
    },
    "NIKA2": {
        "name": "NIKA2",
        "beam": 18.0,
        "map_size": 6.5,
        "conv": (-12.0, 0.9),
        "zero": (0.0, 1e-5),
        "cbar_fact": 1e3,
        "cbar_label": "NIKA2 150 GHz [mJy/beam]",
        "cmap": "RdBu_r",
    },
}
clusters = {
    "C1": {"name": "C1", "z": 0.05, "M_500": 9.0},
    "C2": {"name": "C2", "z": 0.5, "M_500": 6.0},
    "C3": {"name": "C3", "z": 1.0, "M_500": 3.0},
}

instrument, cluster = instruments["Planck"], clusters["C1"]


def get_binning(ppf, n_bins, beam_fwhm):
    pix_kpc = ppf.cluster.arcsec2kpc(ppf.pix_size)
    half_map_kpc = ppf.cluster.arcsec2kpc(ppf.map_size * 60 / 2)
    beam_kpc = ppf.cluster.arcsec2kpc(beam_fwhm)
    r_bins = np.concatenate(
        (
            [pix_kpc],
            np.logspace(
                np.log10(beam_kpc),
                np.log10(1.1 * half_map_kpc),
                n_bins - 1,
            ),
        )
    )
    return r_bins


def run_valid(cluster, instrument, n_bins_P):

    path = f"./results/{cluster['name']}/{instrument['name']}"
    ppf = p2.PressureProfileFitter(
        f"{path}/input_map.fits",
        1,
        5,
        cluster["z"],
        cluster["M_500"] * 1e14,
        map_size=instrument["map_size"],
        coords_center=SkyCoord("12h00m00s +00d00m00s"),
    )

    r_bins = get_binning(ppf, n_bins_P, instrument["beam"])
    ppf.define_model("binned", r_bins)
    P_bins = p2.utils.gNFW(r_bins, *ppf.cluster.A10_params)

    ppf.add_filtering(beam_fwhm=instrument["beam"])

    ppf.define_priors(
        P_bins=[ss.loguniform(0.01 * P, 100.0 * P) for P in P_bins],
        conv=ss.norm(*instrument["conv"]),
        zero=ss.norm(instrument["zero"]),
    )
    ppf.dump_to_file(f"{path}/ppf.panco2")

    _ = ppf.run_mcmc(
        mcmc_params["n_chains"],
        mcmc_params["max_steps"],
        mcmc_params["n_threads"],
        n_check=mcmc_params["n_check"],
        max_delta_tau=mcmc_params["max_delta_tau"],
        min_autocorr_times=mcmc_params["min_autocorr_times"],
        out_chains_file=f"{path}/raw_chains.npz",
        plot_convergence=f"{path}/mcmc_convergence.pdf",
    )
    chains_clean = p2.results.load_chains(
        f"{path}/raw_chains.npz",
        mcmc_params["n_burn"],
        mcmc_params["discard"],
        clip_percent=mcmc_params["clip_percent"],
        verbose=True,
    )

    plt.close("all")
    _ = p2.results.mcmc_trace_plot(
        chains_clean, filename=f"{path}/mcmc_trace.png"
    )
    _ = p2.results.mcmc_corner_plot(
        chains_clean, ppf=ppf, filename=f"{path}/mcmc_corner.pdf"
    )

    meds = dict(chains_clean.median())
    p2.results.plot_data_model_residuals(
        ppf,
        par_dic=meds,
        smooth=0.5,
        cbar_fact=instrument["cbar_fact"],
        lims=None if instrument["name"] == "SPT" else "sym",
        cbar_label=instrument["cbar_label"],
        filename=f"{path}/data_model_residuals_maps.pdf",
        cmap=instrument["cmap"],
    )
    p2.results.plot_data_model_residuals_1d(
        ppf,
        chains_clean=chains_clean,
        y_fact=instrument["cbar_fact"],
        plot_beam=True,
        y_label=instrument["cbar_label"],
        filename=f"{path}/data_model_residuals_profiles.pdf",
        x_log=True,
    )

    r_range = np.logspace(
        np.log10(ppf.cluster.arcsec2kpc(ppf.pix_size / 2)),
        np.log10(ppf.cluster.arcsec2kpc(ppf.map_size * 60 / np.sqrt(2))),
        100,
    )
    fig, ax = p2.results.plot_profile(
        chains_clean, ppf, r_range=r_range, label="panco2"
    )
    ax.plot(
        r_range,
        p2.utils.gNFW(r_range, *ppf.cluster.A10_params),
        "k--",
        label="Truth",
    )
    ax.legend(frameon=False)
    fig.savefig(f"{path}/pressure_profile.pdf")


# =========================================================================== #

if __name__ == "__main__":
    n_bins_P = 5
    run_valid(clusters["C1"], instruments["Planck"], n_bins_P)
    run_valid(clusters["C1"], instruments["SPT"], n_bins_P)
    run_valid(clusters["C2"], instruments["SPT"], n_bins_P)
    run_valid(clusters["C2"], instruments["NIKA2"], n_bins_P)
    run_valid(clusters["C3"], instruments["NIKA2"], n_bins_P)
