#!/usr/bin/env python3
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import os
import pdb
import json
from _cluster import Cluster
import _utils
import _model_gnfw

"""
Script to read panco_params.py and create a simulated cluster map.
"""

# =============================================================================== #
# --------------------------   ARGUMENTS AND OPTIONS   -------------------------- #
# =============================================================================== #

# Factor by which to multiply the noise map added to the sim.
# If fact_noise = 1.0, will be white noise with RMS = the NIKA2 rms in your input fits
fact_noise = 1.0 


# ===== Load option file ===== #
from panco_params import *

if not os.path.isdir(path_to_results):
    os.makedirs(path_to_results)

path_to_plots = path_to_results + "Plots/"
if not os.path.isdir(path_to_plots):
    os.makedirs(path_to_plots)

if not os.path.isfile(file_nk2):
    raise Exception("NIKA2 map nor found. Aborting.")

# =============================================================================== #
# --------------------------   INITIALIZE ALL THINGS   -------------------------- #
# =============================================================================== #

cluster = Cluster(**cluster_kwargs)
model = _model_gnfw.ModelGNFW(cluster, do_ps=do_ps, fit_zl=fit_zl)

# ===== NIKA2 radii maps ===== #
with fits.open(file_nk2) as hdulist:

    head = hdulist[hdu_data].header

    wcs_nk2 = WCS(head)
    reso_nk2 = (np.abs(head["CDELT1"]) * u.deg).to("arcsec")
    reso_nk2_arcsec = reso_nk2.to("arcsec").value

    # ===== If asked, crop the NIKA2 map in a smaller FoV ===== #
    if crop is not None:

        if coords_center is None:
            coords_center = SkyCoord(head["CRVAL1"] * u.deg, head["CRVAL2"] * u.deg)

        # Ensure the number of pixels will be odd after cropping
        new_npix = int(_utils.adim(crop / reso_nk2))
        if new_npix % 2 == 0.0:
            crop += reso_nk2
            new_npix += 1

        new_map = Cutout2D(hdulist[hdu_data].data, coords_center, crop, wcs=wcs_nk2)
        data_nk2 = new_map.data
        new_rms = Cutout2D(hdulist[hdu_rms].data, coords_center, crop, wcs=wcs_nk2)
        wcs_nk2 = new_map.wcs
        rms_nk2 = new_rms.data

    else:
        data_nk2 = hdulist[hdu_data].data
        rms_nk2 = hdulist[hdu_rms].data

    npix = data_nk2.shape[0]
    cpix = int((npix - 1) / 2)
    model.init_profiles_radii(
        reso=reso_nk2_arcsec,
        npix=npix,
        center=(0.0, 0.0),
        r_min_z=1e-3,
        r_max_z=5 * cluster.R_500_kpc,
        nbins_z=100,
        mode=integ_mode,
    )

    # ===== Point sources ===== #
    if do_ps:
        data_nk2, ps_fluxes_sed = model.init_point_sources(
            path=path_ps,
            data=data_nk2,
            wcs=wcs_nk2,
            reso=reso_nk2,
            beam=17.6 * u.arcsec,
            ps_prior_type="pdf",
            # fixed_error=5e-5,
            which_ps=which_ps,
        )
        nps = model.init_ps["nps"]

# ===== Initialize transfer function ===== #
side_tf = 9 * u.arcmin  # TODO: this should NOT have to be hardcoded
pad_tf = int(0.5 * _utils.adim(side_tf / reso_nk2))
model.init_transfer_function(
    file_tf, data_nk2.shape[0], pad_tf, reso_nk2.to("arcsec").value
)

# ===== Finish up model initialization ===== #
model.init_param_indices()

# ===== Do simulation ===== #
par_sim = truth
# param_sim = np.load("/archeops/keruzore/diff_model_pancos.npz")["params"]
# par_sim = model.params_to_dict(param_sim)

zero = par_sim["zero"] if fit_zl else 0.0

model_map, model_Y500 = model(par_sim)
model_Y500_arcmin2 = (model_Y500 * u.kpc ** 2 * u.rad ** 2 / cluster.d_a ** 2).to(
    "arcmin2"
)
print(
    f"Input Y_{integ_mode} = {model_Y500:.2f} kpc2",
    f"= {1e4 * model_Y500_arcmin2.value:.2f} e-4 arcmin2",
)
model_map += fact_noise * np.random.normal(0.0, rms_nk2)

# ===== Write map ===== #
hdulist = fits.open(file_nk2)
if crop is not None:
    bbox = np.array(new_map.bbox_original)
    hdulist[hdu_data].data = hdulist[hdu_data].data * 0.0 + zero
    hdulist[hdu_data].data[
        bbox[0, 0] : bbox[0, 1] + 1, bbox[1, 0] : bbox[1, 1] + 1
    ] = model_map
else:
    hdulist[hdu_data].data = model_map

hdulist.writeto(file_nk2_sim, overwrite=True)
hdulist.close()
print("Wrote simu map:", file_nk2_sim)

# ===== Write parameters ===== #
if do_ps and isinstance(par_sim["ps_fluxes"], np.ndarray):
    par_sim["ps_fluxes"] = par_sim["ps_fluxes"].tolist()
with open(path_to_results + "sim_truth.json", "w") as handle:
    json.dump(par_sim, handle)
print("Wrote simu params:", path_to_results + "sim_truth.json")
