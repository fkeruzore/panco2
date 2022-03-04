import numpy as np
import matplotlib.pyplot as plt
from _cluster import Cluster
from panco2 import PressureProfile, Dataset
from _model_gnfw import gNFW
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

plt.ion()

coords_center = SkyCoord(33.868157 * u.deg, 0.5088958 * u.deg)
cluster_kwargs = {
    "z": 0.865,
    "Y_500": 1.8e-4 * u.arcmin ** 2,
    "err_Y_500": 0.5e-4 * u.arcmin ** 2,
    "theta_500": None,
}

cluster = Cluster(**cluster_kwargs)
pp = PressureProfile(cluster, "./Results/")

tf_nk2 = Table.read("./Demo/Data/transfer_function.fits")
pp.add_dataset(
    Dataset(
        "./Demo/Data/map_input_sim.fits",
        4,
        5,
        18 * u.arcsec,
        crop=6.5*u.arcmin,
        coords_center=coords_center,
        tf_k=tf_nk2["k"],
        tf_filtering=tf_nk2["tf_2mm"],
        name="NIKA2",
        conversion=(-12.0, 1.2),
        d_a=cluster.d_a,
    )
)
pp.add_dataset(
    Dataset(
        "./Demo/Data/map_input_sim.fits",
        4,
        5,
        60 * u.arcsec,
        crop=15 * u.arcmin,
        coords_center=coords_center,
        tf_k=1.0 / np.arange()
        tf_filtering=tf_nk2["tf_2mm"],
        name="NIKA3",
        conversion=(-12.0, 1.2),
        d_a=cluster.d_a,
    )
)
pp.add_integrated_compton(38.0 * u.kpc ** 2, 0.8 * u.kpc ** 2, 800.0 * u.kpc)
pp.init_radial_bins()

press_test = gNFW(pp.radius_tab.to('kpc').value, *cluster.A10_params)
m = pp.datasets[0].compute_compton_map(press_test)
