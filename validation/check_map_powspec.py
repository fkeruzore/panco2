import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import sys

sys.path.append("..")
from panco2 import noise_covariance as nc
import mapview

powspec = Table.read("./example_data/SPT/noise_powspec.csv", format="csv")

mapview.fitsview(
    "results/C2_corrnoise/SPT/input_map.fits", 1, smooth=1, scale=1e5
)
plt.show()
mapview.fitsview(
    "results/C2_corrnoise/SPT/input_map.fits", 4, smooth=1, scale=1e5
)
plt.show()

m = fits.getdata("results/C2_corrnoise/SPT/input_map.fits", 4)
pk, k = nc.powspec(m, 15.0)
fig, ax = plt.subplots()
ax.loglog((180 * 3600 * k), pk, ".-")
ax.plot(powspec["ell"], powspec["c_ell"], "-")
plt.show()
