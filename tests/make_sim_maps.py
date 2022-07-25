import numpy as np
import matplotlib.pyplot as plt
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

sys.path.append("../panco2")

from panco2 import PressureProfileFitter, utils, results


def make_sim_map_nika2(z, M_500, file_out, map_size=6.1, conv=-12.0, zero=0.0):
    ppf = PressureProfileFitter(
        "./example_data/NIKA2/nk2_actj0215_15.fits",
        4,
        5,
        z,
        M_500=M_500,
        map_size=map_size,
    )
    r_bins = np.logspace(
        np.log10(ppf.cluster.arcsec2kpc(ppf.pix_size)),
        np.log10(ppf.cluster.arcsec2kpc(ppf.map_size * 30)),
        50,
    )
    ppf.define_model("binned", r_bins)
    P_bins = utils.gNFW(ppf.model.r_bins, *ppf.cluster.A10_params)
    par_vec = np.append(P_bins, [conv, zero])

    tf = Table.read("./example_data/NIKA2/nk2_tf.fits")
    ppf.add_filtering(
        beam_fwhm=18.0, k=tf["k"].to("arcmin-1").value, tf_k=tf["tf_2mm"].value
    )

    ppf.write_sim_map(par_vec, file_out, filter_noise=False)
