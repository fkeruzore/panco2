import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from scipy.ndimage import gaussian_filter1d, gaussian_filter
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
mask_apod = gaussian_filter(mask.astype(float), 3.0)
ymap_masked = 1e-6 * ymap * mask_apod

# fig, ax = plt.subplots()
# im = ax.imshow(mask_apod)
# fig.colorbar(im, ax=ax)
# plt.show()

pk, k = nc.powspec(ymap_masked, 15.0, n_bins=half_npix)
pk /= mask.sum() / mask.size
# pk *= (5.0 * u.deg.to("rad")) ** 2
ell = 2.0 * 180.0 * 3600 * k
C_ell = pk * u.Unit("arcsec2").to("rad2")
C_ell = gaussian_filter1d(C_ell, 2)
D_ell = ell * (ell + 1) * C_ell / (2.0 * np.pi)

fig, ax = plt.subplots()
ax.semilogy(ell, 1e12 * D_ell, "-", lw=2)
ax.set_xlabel("$\\ell$")
ax.set_ylabel("$D_\\ell \\times 10^{12}$")
ax.set_xlim(0, 6000)
# ax.set_ylim(1e0, 2e2)
plt.show()

t = Table({"ell": ell, "c_ell": C_ell})
t.write("./noise_powspec.csv", format="csv", overwrite=True)
