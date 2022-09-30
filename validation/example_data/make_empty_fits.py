#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
import astropy.units as u

# # NIKA2-like empty file
#
# Header and WCS copied from NIKA2 ACTJ0215 map.
# The RMS map is also the real one from NIKA2 ACTJ0215 observations.

h_in = fits.open(
    "./NIKA2/nk2_actj0215_15.fits"
)  # only to get the RMS map of ACTJ0215

my_wcs = wcs.WCS(
    {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -3.0 / 3600,
        "CDELT2": 3.0 / 3600,
        "CRPIX1": 167,
        "CRPIX2": 167,
        "PC1_1": 1.0,
        "PC1_2": 0.0,
        "PC2_1": 0.0,
        "PC2_2": 1.0,
        "NAXIS": 2,
        "NAXIS1": 333,
        "NAXIS2": 333,
    }
)
my_head = my_wcs.to_header()
my_head["UNIT"] = "JY/BEAM"

h_out = fits.HDUList(
    [
        fits.PrimaryHDU(header=my_head),
        fits.ImageHDU(
            data=np.zeros((333, 333)), header=my_head, name="SZ_MAP"
        ),
        fits.ImageHDU(data=h_in[5].data, header=my_head, name="SZ_RMS"),
    ]
)
h_out.writeto("./NIKA2/empty.fits", overwrite=True)

# # SPT-like empty file
#
# Header and WCS copied from an SPT y-map of El Gordo.
# RMS was computed from the standard deviation of this map with sources masked.

my_wcs = wcs.WCS(
    {
        "CTYPE1": "RA---SFL",
        "CTYPE2": "DEC--SFL",
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -15.0 / 3600,
        "CD2_2": 15.0 / 3600,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CRPIX1": 601,
        "CRPIX2": 601,
        "NAXIS": 2,
        "NAXIS1": 1201,
        "NAXIS2": 1201,
    }
)
my_head = my_wcs.to_header()
my_head["UNIT"] = "YSZ"

h_out = fits.HDUList(
    [
        fits.PrimaryHDU(header=my_head),
        fits.ImageHDU(
            data=np.zeros((1201, 1201)), header=my_head, name="SZ_MAP"
        ),
        fits.ImageHDU(
            data=np.ones((1201, 1201)) * 9.78e-6, header=my_head, name="SZ_RMS"
        ),
    ]
)
h_out.writeto("./SPT/empty.fits", overwrite=True)

# # Planck-like empty file
#
# Header and WCS improvised, as the Planck y-maps are in Healpix.
# RMS is the standard deviation of the noise in the Planck NILC maps, evaluated graphically from the left panel of figure 13 of Planck 2015 XXII (arXiv:1502.01596)

my_wcs = wcs.WCS(
    {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -2.0 / 60,
        "CDELT2": 2.0 / 60,
        "CRPIX1": 91,
        "CRPIX2": 91,
        "PC1_1": 1.0,
        "PC1_2": 0.0,
        "PC2_1": 0.0,
        "PC2_2": 1.0,
        "NAXIS": 2,
        "NAXIS1": 181,
        "NAXIS2": 181,
    }
)
my_head = my_wcs.to_header()
my_head["UNIT"] = "YSZ"

h_out = fits.HDUList(
    [
        fits.PrimaryHDU(header=my_head),
        fits.ImageHDU(data=np.zeros((181, 181)), header=my_head, name="SZ_MAP"),
        fits.ImageHDU(
            data=np.ones((181, 181)) * 4.12e-6, header=my_head, name="SZ_RMS"
        ),
    ]
)
h_out.writeto("./Planck/empty.fits", overwrite=True)
