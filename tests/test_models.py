import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table
from astropy.io import fits
import scipy.stats as ss

import sys
sys.path.append("..")

from panco2 import PressureProfileFitter, results, utils

def test1():
    #hdulist = fits.open("data/data1.fits")

    ppf = PressureProfileFitter(
        "./data/data1.fits",
        1,
        1,
        0.5,
        M_500=5e14,
        map_size=6.0
    )

    r_bins = np.logspace(
        np.log10(ppf.cluster.arcsec2kpc(ppf.pix_size)),
        np.log10(ppf.cluster.arcsec2kpc(ppf.map_size * 30)),
        50,
    )
    ppf.define_model("binned", r_bins)
    P_bins = utils.gNFW(ppf.model.r_bins, *ppf.cluster.A10_params)
    par_vec = np.append(P_bins, [-2.0, -2e-4])

    tf = Table.read("./data/tf_1d.fits")
    ppf.add_filtering(
        beam_fwhm=18.0, k=tf["k"].to("arcmin-1").value, tf_k=tf["tf"].value
    )

    true_mod = ppf.model.sz_map(par_vec) + ppf.model.ps_map(par_vec)
    diff = np.abs((ppf.sz_map - true_mod) / np.mean(ppf.sz_map))

    assert np.all(diff < 1e-2), "Model differs from data by more than 1%"

if __name__ == "__main__":
    test1()
