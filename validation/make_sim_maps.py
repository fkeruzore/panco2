import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import sys

sys.path.append("..")

import panco2 as p2
import mapview

cmap_planck = mpl.colors.ListedColormap(
    np.loadtxt("./example_data/Planck/cmap.txt") / 255.0
)
cmap_spt = "twilight_shifted"
cmap_nika2 = "RdBu_r"


def make_sim_map_nika2(
    cluster,
    file_out,
    map_size=6.1,
    conv=-12.0,
    zero=0.0,
    fact_noise=1.0,
    ps_pos=[],
    ps_fluxes=[],
    plot_map=True,
):
    z, M_500 = cluster["z"], cluster["M_500"]
    ppf = p2.PressureProfileFitter(
        "./example_data/NIKA2/empty.fits",
        1,
        2,
        z,
        M_500=M_500 * 1e14,
        map_size=map_size,
    )
    ppf.sz_rms *= fact_noise
    r_bins = np.logspace(
        np.log10(ppf.cluster.arcsec2kpc(ppf.pix_size)),
        np.log10(ppf.cluster.arcsec2kpc(ppf.map_size * 30)),
        50,
    )
    ppf.define_model(r_bins)
    ppf.add_point_sources(ps_pos, 18.0)

    tf = Table.read("./example_data/NIKA2/nk2_tf.fits")
    ppf.add_filtering(
        beam_fwhm=18.0,
        k=tf["k"].to("arcsec-1").value,
        tf=tf["tf_2mm"].value,
        pad=20,
    )

    P_bins = p2.utils.gNFW(ppf.model.r_bins, *ppf.cluster.A10_params)
    par_vec = np.append(P_bins, [conv, zero, *ps_fluxes])

    ppf.write_sim_map(par_vec, file_out, filter_noise=False)
    if plot_map:
        fig, ax = mapview.fitsview(
            file_out,
            1,
            fwhm=18 * u.arcsec,
            cmap=cmap_nika2,
            imrange="sym",
            smooth=1,
            scale=1e3,
            cbar_label="NIKA2 150 GHz surface brightness [mJy/beam]",
        )
        ax.set_xlabel(r"Right ascension [J2000]")
        ax.set_ylabel(r"Declination [J2000]")
        fig.suptitle(
            f"SPT view of {cluster['name']}: $z={z},"
            + f" M_{{{500}}}={M_500} \\times 10^{{{14}}} \, M_\odot$"
        )
        return fig, ax


def make_sim_map_spt(
    cluster,
    file_out,
    map_size=60.0,
    conv=1.0,
    zero=0.0,
    fact_noise=1.0,
    ps_pos=[],
    ps_fluxes=[],
    corr_noise=False,
    plot_map=True,
):
    z, M_500 = cluster["z"], cluster["M_500"]
    ppf = p2.PressureProfileFitter(
        "./example_data/SPT/empty.fits",
        1,
        2,
        z,
        M_500=M_500 * 1e14,
        map_size=map_size,
    )
    ppf.sz_rms *= fact_noise
    r_bins = np.logspace(
        np.log10(ppf.cluster.arcsec2kpc(ppf.pix_size)),
        np.log10(ppf.cluster.arcsec2kpc(ppf.map_size * 30)),
        50,
    )
    ppf.define_model(r_bins)
    ppf.add_point_sources(ps_pos, 75.0)

    P_bins = p2.utils.gNFW(ppf.model.r_bins, *ppf.cluster.A10_params)
    par_vec = np.append(P_bins, [conv, zero, *ps_fluxes])

    ppf.add_filtering(beam_fwhm=75.0)
    if corr_noise:
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
            return_maps=True,
        )
        # return covs
        ppf.add_covmat(covmat=covs[0], inv_covmat=covs[1])

    ppf.write_sim_map(
        par_vec, file_out, filter_noise=False, corr_noise=corr_noise
    )
    if plot_map:
        fig, ax = mapview.fitsview(
            file_out,
            1,
            fwhm=1.25 * u.arcmin,
            cmap=cmap_spt,
            smooth=1,
            scale=1e5,
            cbar_label="Compton $y \\times 10^5$",
        )
        ax.set_xlabel(r"Right ascension [J2000]")
        ax.set_ylabel(r"Declination [J2000]")
        fig.suptitle(
            f"SPT view of {cluster['name']}: $z={z},"
            + f" M_{{{500}}}={M_500} \\times 10^{{{14}}} \, M_\odot$"
        )
        return fig, ax


def make_sim_map_planck(
    cluster,
    file_out,
    map_size=300.0,
    conv=1.0,
    zero=0.0,
    fact_noise=1.0,
    ps_pos=[],
    ps_fluxes=[],
    plot_map=True,
):
    z, M_500 = cluster["z"], cluster["M_500"]
    ppf = p2.PressureProfileFitter(
        "./example_data/Planck/empty.fits",
        1,
        2,
        z,
        M_500=M_500 * 1e14,
        map_size=map_size,
    )
    ppf.sz_rms *= fact_noise
    r_bins = np.logspace(
        np.log10(ppf.cluster.arcsec2kpc(ppf.pix_size)),
        np.log10(ppf.cluster.arcsec2kpc(ppf.map_size * 30)),
        50,
    )
    ppf.define_model(r_bins)
    ppf.add_point_sources(ps_pos, 600.0)

    P_bins = p2.utils.gNFW(ppf.model.r_bins, *ppf.cluster.A10_params)
    par_vec = np.append(P_bins, [conv, zero, *ps_fluxes])

    ppf.add_filtering(beam_fwhm=600.0)

    ppf.write_sim_map(par_vec, file_out, filter_noise=False)
    if plot_map:
        fig, ax = mapview.fitsview(
            file_out,
            1,
            fwhm=10.0 * u.arcmin,
            cmap=cmap_planck,
            smooth=1,
            scale=1e5,
            imrange="sym",
            cbar_label="Compton $y \\times 10^5$",
        )
        ax.set_xlabel(r"Right ascension [J2000]")
        ax.set_ylabel(r"Declination [J2000]")
        fig.suptitle(
            f"Planck view of {cluster['name']}: $z={z},"
            + f" M_{{{500}}}={M_500} \\times 10^{{{14}}} \, M_\odot$"
        )
        return fig, ax


if __name__ == "__main__":
    C1 = {"name": "C1", "z": 0.05, "M_500": 9.0}
    C2 = {"name": "C2", "z": 0.5, "M_500": 6.0}
    C3 = {"name": "C3", "z": 1.0, "M_500": 3.0}

    C2_corrnoise = {"name": "C2 with correlated noise", "z": 0.5, "M_500": 6.0}
    C2_2d_filter = {"name": "C2 with anisotropic TF", "z": 0.5, "M_500": 6.0}
    C2_ptsources = {"name": "C2 with point sources", "z": 0.5, "M_500": 6.0}

    with PdfPages("sim_maps.pdf") as pdf:

        # =================================================================== #
        print("==> Planck map of C1...")
        np.random.seed(42)
        fig, ax = make_sim_map_planck(
            C1,
            "./results/C1/Planck/input_map.fits",
            map_size=301.0,
        )
        pdf.savefig(fig)

        # =================================================================== #
        print("==> SPT map of C1...")
        np.random.seed(43)
        fig, ax = make_sim_map_spt(
            C1,
            "./results/C1/SPT/input_map.fits",
            map_size=61.0,
        )
        pdf.savefig(fig)

        # =================================================================== #
        print("==> SPT map of C2...")
        np.random.seed(44)
        fig, ax = make_sim_map_spt(
            C2,
            "./results/C2/SPT/input_map.fits",
            map_size=61.0,
        )
        pdf.savefig(fig)

        # =================================================================== #
        print("==> NIKA2 map of C2...")
        np.random.seed(45)
        fig, ax = make_sim_map_nika2(
            C2,
            "./results/C2/NIKA2/input_map.fits",
            map_size=6.6,
            fact_noise=1.1,
        )
        pdf.savefig(fig)

        # =================================================================== #
        print("==> NIKA2 map of C3...")
        np.random.seed(46)
        fig, ax = make_sim_map_nika2(
            C3,
            "./results/C3/NIKA2/input_map.fits",
            map_size=6.6,
            fact_noise=0.9,
        )
        pdf.savefig(fig)

        # =================================================================== #
        print("==> SPT map of C2 with correlated noise...")
        np.random.seed(47)
        fig, ax = make_sim_map_spt(
            C2_corrnoise,
            "./results/C2_corrnoise/SPT/input_map.fits",
            map_size=21.0,
            corr_noise=True,
        )
        pdf.savefig(fig)

        # =================================================================== #
        print("==> SPT map of C2 with 2D filtering ...")
        np.random.seed(48)
        fig, ax = make_sim_map_spt(
            C2_2d_filter,
            "./results/C2_2d_filter/SPT/input_map.fits",
            map_size=21.0,
        )
        pdf.savefig(fig)

        # =================================================================== #
        print("==> NIKA2 map of C2 with point sources...")
        np.random.seed(49)
        fig, ax = make_sim_map_nika2(
            C2_ptsources,
            "./results/C2_ptsources/NIKA2/input_map.fits",
            map_size=6.6,
            fact_noise=1.1,
            ps_pos=[
                SkyCoord("12h00m00s +00d00m30s"),
                SkyCoord("12h00m05s +00d00m10s"),
            ],
            ps_fluxes=[1e-3, 1e-3],
        )
        pdf.savefig(fig)
