import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as ss
import pandas as pd
import seaborn as sns
import cmocean
import os
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter
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
    z,
    M_500,
    file_out,
    map_size=6.1,
    conv=-12.0,
    zero=0.0,
    fact_noise=1.0,
    ps_pos=[],
    ps_fluxes=[],
):
    ppf = p2.PressureProfileFitter(
        "./example_data/NIKA2/empty.fits",
        1,
        2,
        z,
        M_500=M_500,
        map_size=map_size,
    )
    ppf.sz_rms *= fact_noise
    r_bins = np.logspace(
        np.log10(ppf.cluster.arcsec2kpc(ppf.pix_size)),
        np.log10(ppf.cluster.arcsec2kpc(ppf.map_size * 30)),
        50,
    )
    ppf.define_model("binned", r_bins)
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


def make_sim_map_spt(
    z,
    M_500,
    file_out,
    map_size=60.0,
    conv=1.0,
    zero=0.0,
    fact_noise=1.0,
    ps_pos=[],
    ps_fluxes=[],
):
    ppf = p2.PressureProfileFitter(
        "./example_data/SPT/empty.fits",
        1,
        2,
        z,
        M_500=M_500,
        map_size=map_size,
    )
    ppf.sz_rms *= fact_noise
    r_bins = np.logspace(
        np.log10(ppf.cluster.arcsec2kpc(ppf.pix_size)),
        np.log10(ppf.cluster.arcsec2kpc(ppf.map_size * 30)),
        50,
    )
    ppf.define_model("binned", r_bins)
    ppf.add_point_sources(ps_pos, 75.0)

    P_bins = p2.utils.gNFW(ppf.model.r_bins, *ppf.cluster.A10_params)
    par_vec = np.append(P_bins, [conv, zero, *ps_fluxes])

    ppf.add_filtering(beam_fwhm=75.0)

    ppf.write_sim_map(par_vec, file_out, filter_noise=True)


def make_sim_map_planck(
    z,
    M_500,
    file_out,
    map_size=300.0,
    conv=1.0,
    zero=0.0,
    fact_noise=1.0,
    ps_pos=[],
    ps_fluxes=[],
):
    ppf = p2.PressureProfileFitter(
        "./example_data/Planck/empty.fits",
        1,
        2,
        z,
        M_500=M_500,
        map_size=map_size,
    )
    ppf.sz_rms *= fact_noise
    r_bins = np.logspace(
        np.log10(ppf.cluster.arcsec2kpc(ppf.pix_size)),
        np.log10(ppf.cluster.arcsec2kpc(ppf.map_size * 30)),
        50,
    )
    ppf.define_model("binned", r_bins)
    ppf.add_point_sources(ps_pos, 600.0)

    P_bins = p2.utils.gNFW(ppf.model.r_bins, *ppf.cluster.A10_params)
    par_vec = np.append(P_bins, [conv, zero, *ps_fluxes])

    ppf.add_filtering(beam_fwhm=600.0)

    ppf.write_sim_map(par_vec, file_out, filter_noise=True)


if __name__ == "__main__":
    C1 = {"z": 0.05, "M500": 9.0}
    C2 = {"z": 0.5, "M500": 6.0}
    C3 = {"z": 1.0, "M500": 3.0}

    with PdfPages("sim_maps.pdf") as pdf:

        # ==========================================
        # C1 ---------------------------------------

        # ------------------------------------------
        # Planck ...................................

        print("==> Planck map of C1...")
        np.random.seed(42)
        make_sim_map_planck(
            C1["z"],
            C1["M500"] * 1e14,
            "./results/C1/Planck/input_map.fits",
            map_size=301.0,
        )
        fig, ax = mapview.fitsview(
            "./results/C1/Planck/input_map.fits",
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
            f"Planck view of C1: $z={C1['z']},"
            + f" ={C1['M500']} \\times 10^{{{14}}} \, M_\odot$"
        )
        pdf.savefig(fig)

        # ------------------------------------------
        # SPT ......................................

        print("==> SPT map of C1...")
        np.random.seed(43)
        make_sim_map_spt(
            C1["z"],
            C1["M500"] * 1e14,
            "./results/C1/SPT/input_map.fits",
            map_size=61.0,
        )
        fig, ax = mapview.fitsview(
            "./results/C1/SPT/input_map.fits",
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
            f"SPT view of C1: $z={C1['z']},"
            + f" ={C1['M500']} \\times 10^{{{14}}} \, M_\odot$"
        )
        pdf.savefig(fig)

        # ==========================================
        # C2 ---------------------------------------

        # ------------------------------------------
        # SPT ......................................

        print("==> SPT map of C2...")
        np.random.seed(44)
        make_sim_map_spt(
            C2["z"],
            C2["M500"] * 1e14,
            "./results/C2/SPT/input_map.fits",
            map_size=61.0,
        )
        fig, ax = mapview.fitsview(
            "./results/C2/SPT/input_map.fits",
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
            f"SPT view of C2: $z={C2['z']},"
            + f" ={C2['M500']} \\times 10^{{{14}}} \, M_\odot$"
        )
        pdf.savefig(fig)

        # ------------------------------------------
        # NIKA2 ....................................

        print("==> NIKA2 map of C2...")
        np.random.seed(45)
        make_sim_map_nika2(
            C2["z"],
            C2["M500"] * 1e14,
            "./results/C2/NIKA2/input_map.fits",
            map_size=6.6,
            fact_noise=1.1,
        )
        fig, ax = mapview.fitsview(
            "./results/C2/NIKA2/input_map.fits",
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
            f"NIKA2 view of C2: $z={C2['z']},"
            + f" ={C2['M500']} \\times 10^{{{14}}} \, M_\odot$"
        )
        pdf.savefig(fig)

        # ==========================================
        # C3 ---------------------------------------

        # ------------------------------------------
        # NIKA2 ....................................

        np.random.seed(46)
        print("==> NIKA2 map of C3...")
        make_sim_map_nika2(
            C3["z"],
            C3["M500"] * 1e14,
            "./results/C3/NIKA2/input_map.fits",
            map_size=6.6,
            fact_noise=0.9,
        )
        fig, ax = mapview.fitsview(
            "./results/C3/NIKA2/input_map.fits",
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
        ax.set_xlabel(r"Right ascension [J2000]")
        ax.set_ylabel(r"Declination [J2000]")
        fig.suptitle(
            f"NIKA2 view of C3: $z={C3['z']},"
            + f" ={C3['M500']} \\times 10^{{{14}}} \, M_\odot$"
        )
        pdf.savefig(fig)
