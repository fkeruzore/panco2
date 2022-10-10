# import os
#
# os.environ["OPENBLAS_NUM_THREADS"] = "48"
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.table import Table
import scipy.stats as ss

import sys

sys.path.append("..")
import panco2 as p2


mcmc_params = {
    "n_chains": 30,
    "max_steps": 1e5,
    "n_threads": 10,
    "n_check": 1e3,
    "max_delta_tau": 0.05,
    "min_autocorr_times": 50,
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
        "map_size": 30.0,
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
    "C2_corrnoise": {"name": "C2_corrnoise", "z": 0.5, "M_500": 6.0},
    "C2_ptsources": {"name": "C2_ptsources", "z": 0.5, "M_500": 6.0},
    "C2_Y500const": {"name": "C2_Y500const", "z": 0.5, "M_500": 6.0},
}


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


def run_valid(cluster, instrument, n_bins_P, restore=False):
    print(f"===>  CLUSTER {cluster['name']}, {instrument['name']} VIEW  <===")
    path = f"./results/{cluster['name']}/{instrument['name']}"
    if restore:
        ppf = p2.PressureProfileFitter.load_from_file(f"{path}/ppf.panco2")
    else:
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
        ppf.define_model(r_bins)
        P_bins = p2.utils.gNFW(r_bins, *ppf.cluster.A10_params)

        # ========  <INDIVIDUAL OPTIONS>  ======== #
        # FILTERING
        if instrument["name"] == "NIKA2":
            tf = Table.read("./example_data/NIKA2/nk2_tf.fits")
            ppf.add_filtering(
                beam_fwhm=18.0,
                k=tf["k"].to("arcsec-1").value,
                tf=tf["tf_2mm"].value,
                pad=20,
            )
        else:
            ppf.add_filtering(beam_fwhm=instrument["beam"])

        # CORRELATED NOISE
        if cluster["name"] == "C2_corrnoise":
            noise = Table.read(
                "./example_data/SPT/noise_powspec.csv", format="csv"
            )
            ell, c_ell = noise["ell"].value, noise["c_ell"].value
            covs = p2.noise_covariance.covmat_from_powspec(
                ell,
                c_ell,
                ppf.sz_map.shape[0],
                ppf.pix_size,
                n_maps=1000,
                method="lw",
                return_maps=False,
            )
            ppf.add_covmat(covmat=covs[0], inv_covmat=covs[1])
            np.savez_compressed(
                f"{path}/covmats.npz", covmat=covs[0], inv_covmat=covs[1]
            )

        # POINT SOURCES
        elif cluster["name"] == "C2_ptsources":
            ps_pos = [
                SkyCoord("12h00m00s +00d00m30s"),
                SkyCoord("12h00m05s +00d00m10s"),
            ]
            ps_fluxes_priors = [ss.norm(1e-3, 2e-4), ss.uniform(0.0, 2e-3)]
            ppf.add_point_sources(ps_pos, instrument["beam"])

        # INTEGRATED SZ
        elif cluster["name"] == "C2_Y500const":
            ppf.add_integ_Y(75.46, 7.55, ppf.cluster.R_500)

        # ========  </INDIVIDUAL OPTIONS>  ======== #
        ppf.define_priors(
            P_bins=[ss.loguniform(0.01 * P, 100.0 * P) for P in P_bins],
            conv=ss.norm(*instrument["conv"]),
            zero=ss.norm(*instrument["zero"]),
            ps_fluxes=(
                ps_fluxes_priors if cluster["name"] == "C2_ptsources" else []
            ),
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
    _ = p2.results.mcmc_matrices_plot(
        chains_clean, ppf, filename=f"{path}/mcmc_matrices.pdf"
    )

    meds = dict(chains_clean.median())
    p2.results.plot_data_model_residuals(
        ppf,
        par_dic=meds,
        smooth=1.0,
        cbar_fact=instrument["cbar_fact"],
        lims=None if instrument["name"] == "SPT" else "sym",
        cbar_label=instrument["cbar_label"],
        filename=f"{path}/data_model_residuals_maps.pdf",
        cmap=instrument["cmap"],
        separate_ps_model=(cluster["name"] == "C2_ptsources"),
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
    # run_valid(clusters["C1"], instruments["Planck"], n_bins_P)
    # run_valid(clusters["C1"], instruments["SPT"], n_bins_P)
    # run_valid(clusters["C2"], instruments["SPT"], n_bins_P)
    # run_valid(clusters["C2"], instruments["NIKA2"], n_bins_P)
    # run_valid(clusters["C3"], instruments["NIKA2"], n_bins_P)
    # run_valid(clusters["C2_corrnoise"], instruments["SPT"], n_bins_P)
    # run_valid(clusters["C2_ptsources"], instruments["NIKA2"], n_bins_P)
    run_valid(clusters["C2_Y500const"], instruments["NIKA2"], n_bins_P)
