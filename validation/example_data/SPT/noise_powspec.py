from random import gauss
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
import sys

sys.path.append("../../..")

import panco2.noise_covariance as nc

hdulist = fits.open("./SPT-CLJ0102-4915_ymap_and_rms.fits")

ymap = hdulist[1].data
mask = hdulist[3].data.astype(bool)

half_npix = ymap.shape[0] // 2
theta_map = np.hypot(
    *np.meshgrid(
        np.arange(-half_npix, half_npix), np.arange(-half_npix, half_npix)
    )
)
mask &= theta_map > 40
ymap_masked = ymap * mask

# fig, ax = plt.subplots()
# im = ax.imshow(ymap_masked)
# fig.colorbar(im, ax=ax)
# plt.show()

pk, k = nc.powspec(ymap_masked, 15.0, n_bins=half_npix)
ell = 180.0 * 3600 * k
c_ell = gaussian_filter1d(pk, 5)

fig, ax = plt.subplots()
ax.loglog(ell, pk, ".")
ax.loglog(ell, c_ell, "-", lw=2)
plt.show()

t = Table({"ell": ell, "c_ell": c_ell, "d_ell": ell * (ell + 1) * c_ell})
t.write("./noise_powspec.csv", format="csv", overwrite=True)
