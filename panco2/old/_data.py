#!/usr/bin/env python3
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import covar
import _utils

np.set_printoptions(precision=3)
_utils.ignore_astropy_warnings()

"""
"""


class Dataset:
    """
    A class to store information about a dataset (maps, covariance matrices,
    resolution, etc).

    Args:
        in_fits_file (str): path to a fits file;
        hdu_data (int): extension containing the map to be fitted;
        hdu_rms (int): extension containing the noise RMS map;
        reso_fwhm (Quantity): the FWHM of the beam of the instrument,
            in angle units;
        crop (Quantity or None): the size of the map to be fitted,
            in angle units;
        coords_center (SkyCoord or None): the center of the map to be
            fitted in equatorial coordinates;
        inv_covmat (array or None): inverse of the noise covariance matrix,
            in the same units as the input map to the power of -2;
        file_noise_simus (str or None): path to a fits file where each
            extension is a correlated noise realization normalized to the
            noise RMS, to be used to compute the covariace matrix if it is
            not provided.
    """

    def __init__(
        self,
        in_fits_file,
        hdu_data,
        hdu_rms,
        reso_fwhm,
        crop=None,
        coords_center=None,
        inv_covmat=None,
        file_noise_simus=None,
        name=None,
    ):
        self.reso_fwhm = reso_fwhm
        self.name = name if (name is not None) else "Dataset"
        self.map, self.rms, _, self.inv_covmat, self.wcs, self.pix_size = read_data(
            in_fits_file,
            hdu_data,
            hdu_rms,
            crop=crop,
            coords_center=coords_center,
            inv_covmat=inv_covmat,
            file_noise_simus=file_noise_simus,
        )
        self.map_size = self.map.shape[0] * self.pix_size


def read_data(
    file_nk2,
    hdu_data,
    hdu_rms,
    crop=None,
    coords_center=None,
    inv_covmat=None,
    file_noise_simus=None,
):
    """
    Reads input SZ data and formats it for the MCMC.

    Args:
        file_nk2 (str): path to a fits file;
        hdu_data (int): extension containing the map to be fitted;
        hdu_rms (int): extension containing the noise RMS map;
        crop (Quantity or None): the size of the map to be fitted,
            in angle units;
        coords_center (SkyCoord or None): the center of the map to be
            fitted in equatorial coordinates;
        inv_covmat (array or None): inverse of the noise covariance matrix,
            in the same units as the input map to the power of -2;
        file_noise_simus (str or None): path to a fits file where each
            extension is a correlated noise realization normalized to the
            noise RMS, to be used to compute the covariace matrix if it is
            not provided.

    Returns:
        (tuple): tuple containing:

        * (array) the SZ map correctly cropped and centered;
        * (array) the noise RMS map correctly cropped and centered;
        * (array or None) the noise covariance matrix;
        * (array or None) the inverse of the noise covariance matrix;
        * (WCS) the world coordinate system associated to the maps;
        * (Quantity) the pixel size of the maps in angle units.

    """

    # ===== NIKA2 data map and RMS map ===== #
    hdulist = fits.open(file_nk2)

    head = hdulist[hdu_data].header

    wcs_nk2 = WCS(head)
    reso_nk2 = (np.abs(head["CDELT1"]) * u.deg).to("arcsec")
    # reso_nk2_arcsec = reso_nk2.to("arcsec").value

    # If asked, crop the NIKA2 map in a smaller FoV
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

    hdulist.close()

    # ===== Noise covariance matrix ===== #
    if (inv_covmat is None) and (file_noise_simus is None):
        covmat = None
        print("    No covariance or noise simulations: considering white noise")

    elif (inv_covmat is None) and (file_noise_simus is not None):
        print("    Noise covariance matrix computation...")
        # Read correlated noise maps
        hdulist_noise = fits.open(file_noise_simus)
        n_maps = len(hdulist_noise)
        if crop is not None:
            noise_maps = np.array(
                [
                    Cutout2D(hdu.data, coords_center, crop, wcs=wcs_nk2).data
                    for hdu in hdulist_noise
                ]
            )
        else:
            noise_maps = np.array([hdu.data * rms_nk2 for hdu in hdulist_noise])

        if hdulist_noise[1].header["UNIT"] != "Jy/beam":
            print(
                "    Noise maps were not in Jy/beam? Multiplied them by the noise RMS"
            )
            noise_maps = noise_maps * rms_nk2[np.newaxis, :, :]
        hdulist_noise.close()

        # Compute covariance matrix
        noise_vecs = noise_maps.reshape(n_maps, -1)
        covmat = np.cov(noise_vecs, rowvar=False)

        # numpy needs help for large covariace matrices
        # see https://pythonhosted.org/covar/
        covmat, shrink = covar.cov_shrink_rblw(covmat, n_maps)
        print(f"    Covariance shrinkage: {shrink}")
        inv_covmat = np.linalg.inv(covmat)

    elif inv_covmat is not None:
        print("    Loading inverse covariance matrix...")
        covmat = None

    return data_nk2, rms_nk2, covmat, inv_covmat, wcs_nk2, reso_nk2
