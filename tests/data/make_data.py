import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
import sys

sys.path.append("../..")

import panco2 as p2

def make_test1_map():
    file_out = "./data1.fits"
    ppf = p2.PressureProfileFitter(
        "../../science_tests/example_data/NIKA2/nk2_actj0215_15.fits",
        4,  # SZ HDU
        5,  # RMS HDU
        0.5,  # z
        M_500=5e14,  # Msun
        map_size=6.1,  # arcmin
    )
    r_bins = np.logspace(
        np.log10(ppf.cluster.arcsec2kpc(ppf.pix_size)),
        np.log10(ppf.cluster.arcsec2kpc(ppf.map_size * 30)),
        50,
    )
    ppf.define_model("binned", r_bins)
    P_bins = p2.utils.gNFW(ppf.model.r_bins, *ppf.cluster.A10_params)
    par_vec = np.append(P_bins, [-2.0, -2e-4])  # [conv, zero]

    tf = Table.read("./tf_1d.fits")
    ppf.add_filtering(
        beam_fwhm=18.0, k=tf["k"].to("arcmin-1").value, tf_k=tf["tf"].value
    )

    ppf.write_sim_map(par_vec, file_out, filter_noise=False)
    hdulist = fits.open(file_out)
    hdulist[1].data = hdulist[2].data + hdulist[3].data
    for i in [5, 4, 3, 2]:
        hdulist.pop(i)
    hdulist.writeto(file_out, overwrite=True)



if __name__ == "__main__":
    make_test1_map()
