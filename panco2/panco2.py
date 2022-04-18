#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import emcee
from iminuit import Minuit
import covar
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import sys
import os
import shutil
import json
import time
import argparse
from multiprocessing import Pool
import pdb
from _cluster import Cluster
import _data
import _xray
import _probability
import _utils
import _model_gnfw, _model_nonparam
import _fit_gnfw_on_non_param
import _results


class PressureProfileFitter:
    """
    Blah, blah, blah

    Parameters
    ----------
    sz_map_file : str
        Path to a FITS file containing the SZ map and noise RMS
    hdu_data : int
        Index of the FITS extension in which the SZ map is stored
    hdu_rms : int
        Index of the FITS extension in which the RMS map is stored
    z : float
        Cluster's redshift
    Y_500 : float [arcmin2], optional
        Integrated Compton parameter within R_500.
        This is used to build the starting point of the MCMC.
        Either one of Y_500 or M_500 need to be specified.
    M_500 : float [Msun], optional
        A guess of the cluster's mass.
        This is used to build the starting point of the MCMC.
        Either one of Y_500 or M_500 need to be specified.
    beam_fwhm : float [arcsec], optional
        The FWHM of your gaussian beam.
        Can be set to 0 if you don't want beam confolution
            (e.g. if the beam is already in the transfer function).
    coords_center : SkyCoord, optional
        Coordinate to consider as the center of the map
    map_size : float [arcmin], optional
        The size of the map to be considered
    """

    def __init__(
        self,
        sz_map_file,
        hdu_data,
        hdu_rms,
        z,
        Y_500=None,
        M_500=None,
        beam_fwhm=0.0,
        coords_center=None,
        map_size=None,
    ):

        self.cluster = Cluster(z, Y_500=Y_500, M_500=M_500)

        # ===== Read and formats the data ===== #
        hdulist = fits.open(sz_map_file)
        head = hdulist[hdu_data].header

        wcs = WCS(head)
        pix_size = (
            (np.abs(head["CDELT1"]) * u.deg).to("arcsec").value
        )  # arcsec

        # Coordinates of the enter if not specified
        if coords_center is None:
            coords_center = SkyCoord(
                head["CRVAL1"] * u.deg, head["CRVAL2"] * u.deg
            )

        # If asked, map_size the NIKA2 map in a smaller FoV
        if map_size is not None:
            # Ensure the number of pixels will be odd after map_sizeping
            new_npix = int(map_size * 60 / pix_size)
            if new_npix % 2 == 0.0:
                map_size += pix_size
                new_npix += 1

            map_sizeped_map = Cutout2D(
                hdulist[hdu_data].data, coords_center, map_size, wcs=wcs
            )
            self.sz_map = map_sizeped_map.data
            map_sizeped_rms = Cutout2D(
                hdulist[hdu_rms].data, coords_center, map_size, wcs=wcs
            )
            self.sz_rms = map_sizeped_rms.data
            self.wcs = map_sizeped_map.wcs

        else:
            self.sz_map = hdulist[hdu_data].data
            self.sz_rms = hdulist[hdu_rms].data
            self.wcs = wcs

        hdulist.close()
        self.pix_size = pix_size  # arcsec
        self.coords_center = coords_center
        self.map_size = map_size  # arcmin

        # By default no transfer function convolution
        def convolve_tf(x):
            return x

        self.__convolve_tf = convolve_tf

        # Beam convolution
        if beam_fwhm == 0.0:

            def convolve_beam(x):
                return x

        else:
            beam_sigma_pix = (
                beam_fwhm / (2 * np.sqrt(2 * np.log(2))) / pix_size
            )

            def convolve_beam(x):
                return gaussian_filter(x, beam_sigma_pix)

        self.__convolve_beam = convolve_beam

        """
        model_type = model_type.lower()
        if model_type == "gnfw":
            self.model = _model_gnfw.ModelGNFW(self.cluster, zero_level=fit_zero)
        elif model_type == "binned":
            self.model = _model_nonparam.ModelNonParam(
                self.cluster, zero_level=fit_zero
            )
        else:
            raise Exception("Unrecognized `model_type`: " + model_type)

        n_bins = 5  # TODO
        self.model.init_profiles_radii(
            reso=pix_size,
            npix=sz_map.shape[0],
            r_min_z=1e-3,
            r_max_z=5 * self.cluster.R_500_kpc,
            nbins_z=100,
            mode="500",  # TODO
            radius_tab=radial_bins,
            n_bins=n_bins,
        )

        side_tf = 9 * u.arcmin  # TODO: this should NOT have to be hardcoded
        pad_tf = int(0.5 * _utils.adim(side_tf / pix_size))
        self.model.init_transfer_function(
            file_tf,
            sz_map.shape[0],
            pad_tf,
            pix_size.to("arcsec").value,
            beam_fwhm / (2 * np.sqrt(2 * np.log(2))) / pix_size,
        )

        # TODO add point sources here?

        self.model.init_param_indices()
        """

    # ---------------------------------------------------------------------- #

    def add_covmat(self, file_noise_simus=None, inv_covmat_file=None):
        """
        Loads or computes a covariance matrix for the likelihood.

        Parameters
        ----------
        file_noise_simus : str
            Path to a `fits` file in which each extension is a noise map.
            The covariance will be computed as the pixel-to-pixel
                covariance of these maps.
            The first extension must have a header that can be used to
                create a `WCS` object to ensure the noise and data
                pixels are the same.
        inv_covmat : str
            Path to a `npy` file containing an inverse covariance matrix.
            The matrix must be the same size and units as the data.

        """

        if (inv_covmat_file is None) and (file_noise_simus is None):
            covmat = None
            print(
                "No covariance or noise simulations: considering white noise"
            )

        elif (inv_covmat_file is None) and (file_noise_simus is not None):
            print("Noise covariance matrix computation...")
            # Read correlated noise maps
            hdulist_noise = fits.open(file_noise_simus)
            n_maps = len(hdulist_noise)
            wcs = WCS(hdulist_noise[0].header)
            noise_maps = np.array(
                [
                    Cutout2D(
                        hdu.data, self.coords_center, self.map_size, wcs=wcs
                    ).data
                    for hdu in hdulist_noise
                ]
            )
            hdulist_noise.close()

            # Compute covariance matrix
            noise_vecs = noise_maps.reshape(n_maps, -1)
            covmat = np.cov(noise_vecs, rowvar=False)

            # numpy needs help for large covariace matrices
            # see https://pythonhosted.org/covar/
            covmat, shrink = covar.cov_shrink_rblw(covmat, n_maps)
            print(f"    Covariance shrinkage: {shrink}")
            self.inv_covmat = np.linalg.inv(covmat)

        elif inv_covmat_file is not None:
            print("    Loading inverse covariance matrix...")
            self.inv_covmat = np.load(inv_covmat_file)

    # ---------------------------------------------------------------------- #

    def add_transfer_function(self, tf_k, k, pad=60.0):
        """
        Initialize a transfer function convolution in the
        model computation.

        Parameters
        ----------
        tf_k : array
            Filtering measurement.
        k : array [arcmin-1]
            Angular frequencies at which the filtering was measured.
        pad : float [arcsec]
            Padding to be added to the sides of the map before convolution.

        Notes
        -----
        The convention used for `k` is the same as the `numpy` one,
            i.e. the largest 1D mode is 1/(pixel size).
        """

        # Compute the modes covered in the map
        npix = self.sz_map.shape[0]
        k_vec = np.fft.fftfreq(npix + 2 * pad, self.pix_size)
        karr = np.hypot(*np.meshgrid(k_vec, k_vec))

        interp = interp1d(k, tf_k, bounds_error=False, fill_value=tf_k[-1])
        tf_arr = interp(karr)

        self.transfer_function = {"k": karr, "tf_k": tf_arr}
        self.pad_pix = int(pad / self.pix_size)

        def convolve_tf(in_map):
            in_map_pad = np.pad(
                in_map, self.pad_pix, mode="constant", constant_values=0.0
            )
            in_map_fourier = np.fft.fft2(in_map_pad)
            conv_in_map = np.real(np.fft.ifft2(in_map_fourier * tf_arr))
            return conv_in_map[pad:-pad, pad:-pad]

        self.__convolve_tf = convolve_tf

    # ---------------------------------------------------------------------- #
    def convolve_beam(self, in_map):
        return self.__convolve_beam(in_map)

    def convolve_tf(self, in_map):
        return self.__convolve_tf(in_map)

    # ---------------------------------------------------------------------- #

    def define_model(
        self,
        model_type,
        r_bins,
        zero_level=True,
        integ_Y=None,
    ):

        self.model_type = model_type.lower()
        d_a = self.cluster.d_a

        # ===== Radii arrays ===== #
        # 1D radius in the sky plane, only half the map
        theta_x = (
            np.arange(0, int(npix / 2) + 1) * reso_rad
        )  # angle, in radians
        r_x = d_a * np.tan(theta_x)  # distance, in kpc

        # 2D (x, y) radius in the sky plane to compute compton map
        r_xy = np.hypot(
            *np.meshgrid(
                np.concatenate((-np.flip(r_x[1:]), r_x)),
                np.concatenate((-np.flip(r_x[1:]), r_x)),
            )
        )

        self.__radii = {"r_x": r_x, "r_xy": r_xy, "theta_x": theta_x}

        # If gNFW, we need to define the line of sight
        if self.model_type == "gnfw":
            # 1D LoS radius
            r_z = np.logspace(-3.0, np.log10(5 * self.cluster.R_500), 500)

            # 2D (x, z) radius plane to compute 1D compton profile
            r_xx, r_zz = np.meshgrid(r_x, r_z)
            r_xz = np.hypot(r_xx, r_zz)

            self.__radii["r_z"] = r_z
            self.__radii["r_zz"] = r_zz
            self.__radii["r_xz"] = r_xz

    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
