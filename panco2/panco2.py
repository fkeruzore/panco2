#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import emcee
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
import scipy.stats as ss
import sys
import os
import dill
import time
from multiprocessing import Pool
from . import utils
from . import model
from .cluster import Cluster
from .filtering import Filter


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
    M_500 : float [Msun]
        A guess of the cluster's mass.
        This is used to build the starting point of the MCMC.
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
        M_500,
        coords_center=None,
        map_size=None,
    ):

        self.cluster = Cluster(z, M_500=M_500)

        # ===== Read and formats the data ===== #
        hdulist = fits.open(sz_map_file)
        head = hdulist[hdu_data].header

        wcs = WCS(head)
        pix_size = (
            (np.abs(head["CDELT1"]) * u.deg).to("arcsec").value
        )  # arcsec

        # Coordinates of the enter if not specified
        if coords_center is None:
            pix_center = np.array([head[f"NAXIS{ii}"] // 2 for ii in (1, 2)])
            ra, dec = wcs.all_pix2world([pix_center], 1)[0]
            coords_center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            # coords_center = SkyCoord(
            #     head["CRVAL1"] * u.deg, head["CRVAL2"] * u.deg
            # )

        # If asked, map_size the NIKA2 map in a smaller FoV
        if map_size is not None:
            self.map_size = map_size  # arcmin
            # Ensure the number of pixels will be odd after map_sizeping
            new_npix = int(map_size * 60 / pix_size)
            if new_npix % 2 == 0.0:
                map_size += pix_size / 60.0
                new_npix += 1

            cropped_map = Cutout2D(
                hdulist[hdu_data].data, coords_center, new_npix, wcs=wcs
            )
            self.sz_map = cropped_map.data
            cropped_rms = Cutout2D(
                hdulist[hdu_rms].data, coords_center, new_npix, wcs=wcs
            )
            self.sz_rms = cropped_rms.data
            self.wcs = cropped_map.wcs

        else:
            self.sz_map = hdulist[hdu_data].data
            self.sz_rms = hdulist[hdu_rms].data
            self.wcs = wcs
            self.map_size = self.sz_map.shape[0] * pix_size / 60

        hdulist.close()
        self.pix_size = pix_size  # arcsec
        self.coords_center = coords_center
        self.inv_covmat = None
        self.has_covmat = False
        self.has_integ_Y = False

    # ---------------------------------------------------------------------- #

    @classmethod
    def load_from_file(cls, file_name):
        with open(file_name, "rb") as f:
            inst = dill.load(f)
        return inst

    def dump_to_file(self, file_name):
        with open(file_name, "wb") as f:
            f.write(dill.dumps(self))

    # ---------------------------------------------------------------------- #

    def default_radial_binning(self, bin2_arcsec):
        pix_kpc = self.cluster.arcsec2kpc(self.pix_size)
        map_kpc = self.cluster.arcsec2kpc(self.map_size * 60 / 2)
        beam_kpc = self.cluster.arcsec2kpc(bin2_arcsec)
        beam_pix = beam_kpc / pix_kpc
        r_bins = np.array([pix_kpc * (beam_pix**i) for i in range(100)])
        r_bins = r_bins[: int(np.max(np.where(r_bins < map_kpc)) + 2)]
        return r_bins

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
            # covmat, shrink = covar.cov_shrink_rblw(covmat, n_maps)
            # print(f"    Covariance shrinkage: {shrink}")
            self.inv_covmat = np.linalg.inv(covmat)

        elif inv_covmat_file is not None:
            print("    Loading inverse covariance matrix...")
            self.inv_covmat = np.load(inv_covmat_file)
        self.has_covmat = True

    # ---------------------------------------------------------------------- #

    def add_filtering(self, beam_fwhm=0.0, tf_k=None, k=None, pad=60.0):
        """
        Initialize convolution by a beam and
        transfer function in the model computation.

        Parameters
        ----------
        beam_fwhm : float [arcsec], optional
            The FWHM of your gaussian beam.
            Can be set to 0 if you don't want beam confolution
            (e.g. if the beam is already in the transfer function).
        tf_k : array
            Filtering measurement.
        k : array [arcmin-1]
            Angular frequencies at which the filtering was measured.
        pad : float [arcsec]
            Padding to be added to the sides of the map before convolution.

        Notes
        =====
        The convention used for `k` is the same as the `numpy` one,
        i.e. the largest 1D mode is 1/(pixel size).
        """

        self.beam_fwhm = beam_fwhm
        beam_sigma_pix = (
            beam_fwhm / (2 * np.sqrt(2 * np.log(2))) / self.pix_size
        )
        self.model.filter = Filter(
            self.sz_map.shape[0],
            self.pix_size,
            beam_sigma_pix=beam_sigma_pix,
            tf_k=tf_k,
            k=k,
            pad=pad,
        )

    # ---------------------------------------------------------------------- #

    def define_model(
        self,
        model_type,
        r_bins,
        zero_level=True,
        integ_Y=None,
    ):

        model_type = model_type.lower()
        d_a = self.cluster.d_a
        npix = self.sz_map.shape[0]

        # ===== Radii arrays ===== #
        # 1D radius in the sky plane, only half the map
        theta_x = (
            np.arange(0, int(npix / 2) + 1)
            * self.pix_size
            * u.arcsec.to("rad")
        )  # angle, in radians
        r_x = d_a * np.tan(theta_x)  # distance, in kpc

        # 2D (x, y) radius in the sky plane to compute compton map
        r_xy = np.hypot(
            *np.meshgrid(
                np.concatenate((-np.flip(r_x[1:]), r_x)),
                np.concatenate((-np.flip(r_x[1:]), r_x)),
            )
        )

        self.radii = {"r_x": r_x, "r_xy": r_xy, "theta_x": theta_x}

        # If gNFW, we need to define the line of sight
        if model_type == "gnfw":
            # 1D LoS radius
            r_z = np.logspace(-3.0, np.log10(5 * self.cluster.R_500), 500)

            # 2D (x, z) radius plane to compute 1D compton profile
            r_xx, r_zz = np.meshgrid(r_x, r_z)
            r_xz = np.hypot(r_xx, r_zz)

            self.radii["r_z"] = r_z
            self.radii["r_zz"] = r_zz
            self.radii["r_xz"] = r_xz

        self.model = model.ModelBinned(
            r_bins, self.radii, zero_level=zero_level
        )

    # ---------------------------------------------------------------------- #

    def add_point_sources(self, coords, beam_fwhm):
        beam_sigma_pix = (
            beam_fwhm / (2 * np.sqrt(2 * np.log(2))) / self.pix_size
        )
        self.model.add_point_sources(coords, self.wcs, beam_sigma_pix)

    # ---------------------------------------------------------------------- #

    def add_integ_Y(self, Y, dY, r):
        """
        Add a constraint on the integrated Compton parameter
        to the fit.

        Parameters
        ----------
        Y : float [kpc2]
            Integrated Compton parameter value
        dY : float [kpc2]
            Uncertainty on Y
        r : float [kpc]
            Radius within which the Compton parameter was integrated
        """
        self.integ_Y = (Y, dY)
        self.r_integ_Y = r
        self.model.init_integ_Y(r)

        self.has_integ_Y = True

    # ---------------------------------------------------------------------- #

    def define_priors(
        self,
        P_bins=None,
        gNFW_params=None,
        conv=ss.norm(0.0, 0.1),
        zero=ss.norm(0.0, 0.1),
        ps_fluxes=[],
    ):
        """

        Parameters
        ----------
        P_bins : ss.distributions or list of ss.distributions
            Priors on the pressure bins in binned profile mode.
            If only one distribution, all bins will have the same
            prior. If a list, the length should be the number of bins.
        gNFW_params : #TODO
        conv : ss.distributions instance
            Prior on the conversion coefficient.
            Defaults to N(0, 1).
        zero : ss.distributions instance
            Prior on the zero level of the map.
            Defaults to N(0, 1).
        ps_fluxes : list of ss.distributions
            Priors on the flux of each point source, in map units.

        Raises
        ======
        Exception
            If something happens #TODO

        Notes
        =====
        To create distributions, use the `scipy.stats` module.

        Examples
        ========
        - Normal distribution with mean 0 and spread 1:

            import scipy.stats as ss

            prior_on_parameter = ss.norm(0.0, 1.0)


        - Uniform distribution between 0 and 1:

            import scipy.stats as ss

            prior_on_parameter = ss.uniform(0.0, 1.0)

        """
        priors = {}

        # Pressure profile parameters
        if self.model.type == "binned":
            if isinstance(P_bins, (list, tuple, np.ndarray)):
                if len(P_bins) == self.model.n_bins:
                    for i, P in enumerate(P_bins):
                        priors[f"P_{i}"] = P
                else:
                    raise Exception(
                        "`P_bins` is a list but has the wrong length"
                    )
            elif isinstance(P_bins, ss.distributions.rv_frozen):
                for i in range(self.model.n_bins):
                    priors[f"P_{i}"] = P_bins
            else:
                raise Exception(
                    f"Invalid priors for `P_bins` in a binned model: {P_bins}"
                )

        elif self.model.type == "gnfw":
            pass  # TODO

        # Conversion coefficient
        priors["conv"] = conv

        # Zero level
        priors["zero"] = zero

        # Point sources
        if len(ps_fluxes) == self.model.n_ps:
            for i, F in enumerate(ps_fluxes):
                priors[f"F_{i+1}"] = F  #TODO this is 1-indexed, is this evil?
        else:
            raise Exception("`ps_fluxes` is a list but has the wrong length")

        self.model.priors = priors

    # ---------------------------------------------------------------------- #

    def log_lhood(self, par_vec):
        mod = self.model.sz_map(par_vec) + self.model.ps_map(par_vec)
        sqrtll = (self.sz_map - mod) / self.sz_rms
        ll = -0.5 * np.sum(sqrtll**2)

        if self.has_integ_Y:
            Y = self.model.integ_Y(par_vec)
            ll_integY = -0.5 * ((Y - self.integ_Y[0]) / self.integ_Y[1]) ** 2
            ll += ll_integY

        if np.isfinite(ll):
            return ll
        else:
            return -np.inf

    # ---------------------------------------------------------------------- #

    def log_lhood_covmat(self, par_vec):
        mod = self.model.sz_map(par_vec) + self.model.ps_map(par_vec)
        diff = (self.sz_map - mod).flatten()
        m2ll = diff @ self.inv_covmat @ diff
        ll = -0.5 * np.sum(m2ll)
        if np.isfinite(ll):
            return ll
        else:
            return -np.inf

    # ---------------------------------------------------------------------- #

    def write_sim_map(self, par_vec, out_file, filter_noise=True):
        """
        Write a FITS map with given parameter values

        Parameters
        ----------
        par_vec : list or array
            Vector in the parameter space.
        out_file : str
            Path to a FITS file to which the map will be written.
            If the file already exists, it will be overwritten.
        filter_noise : bool
            If True, convolve the noise realization by your
            filtering kernel.

        Notes
        =====
        The created FITS file contains the following extensions:

        - HDU 0: primary, contains header and no data.

        - HDU 1: SZMAP, contains the model map and header.

        - HDU 2: RMS, contains the noise RMS map and header.

        All maps are cropped identically to the data used for the
        fit, and the headers are adjusted accordingly.
        """
        mod_map = self.model.sz_map(par_vec) + self.model.ps_map(par_vec)
        noise = np.random.normal(np.zeros_like(self.sz_map), self.sz_rms)
        if filter_noise:
            noise = self.model.filter(noise)

        header = self.wcs.to_header()
        hdu0 = fits.PrimaryHDU(header=header)
        hdu1 = fits.ImageHDU(data=mod_map + noise, header=header, name="SZMAP")
        hdu2 = fits.ImageHDU(data=self.sz_rms, header=header, name="RMS")

        hdulist = fits.HDUList(hdus=[hdu0, hdu1, hdu2])
        hdulist.writeto(out_file, overwrite=True)

    # ---------------------------------------------------------------------- #

    def fastfit(self):

        # Starting point for pressure = A10 universal pressure profile
        A10 = utils.gNFW(self.model.r_bins, *self.cluster.A10_params)
        P_start = np.random.lognormal(np.log(A10), 0.1)

        # Starting point for the rest = mean of the priors
        nonP_start = [np.mean(self.model.priors["conv"].rvs(1000))]
        if self.model.zero_level:
            nonP_start.append(np.mean(self.model.priors["zero"].rvs(1000)))
        for i in range(self.model.n_ps):
            nonP_start.append(np.mean(self.model.priors[f"F_{i+1}"].rvs(1000)))
        start = np.append(P_start, nonP_start)

        # Fast fit by minimizing -log posterior
        def tomin(par_vec):
            post, _, _ = log_post(
                par_vec, self.log_lhood, self.model.log_prior
            )
            return -2.0 * post

        f = minimize(tomin, x0=start)

        return f["x"]

    # ---------------------------------------------------------------------- #

    def run_mcmc(
        self,
        n_chains,
        max_steps,
        n_threads,
        n_check=1e3,
        max_delta_tau=1e-2,
        min_autocorr_times=100,
        out_chains_file="./chains.npz",
    ):
        """
        Runs MCMC sampling of the posterior distribution.

        Parameters
        ----------
        n_chains : int
            Number of emcee walkers.
        max_steps : int
            Maximum number of steps in the Markov chains.
            The final number of points can be lower if
            convergence is accepted before `max_steps` is
            reached -- see Notes.
        n_threads : int
            Number of parallel threads to use.
        n_check : int
            Number of steps between two convergence checks.
        max_delta_tau : float
            Maximum relative difference of the autocorrelation
            length between two convergence checks to end sampling.
        min_autocorr_times : int
            Minimum ratio between the chains length and the
            autocorrelation length to end samlling.
        out_chains_file : str
            Path to a `.npz` file in which the chains
            will be stored.

        Returns
        =======
        chains : dict
            Markov chains. Each key is a parameter, and the
            values are 2D arrays of shape (n_chains, n_steps).

        Notes
        =====
        - The convergence check are performed every `n_check` steps.
          Convergence is accepted if:

          * In the last two checks, the mean autocorrelation time
            had a relative variation smaller than `max_delta_tau`
            compared to the previous step.

          * The chain is longer than `min_autocorr_times` times
            the current mean autocorrelation time.

        """
        n_chains = int(n_chains)
        max_steps = int(max_steps)
        n_threads = int(n_threads)
        n_check = int(n_check)
        min_autocorr_times = int(min_autocorr_times)

        # Starting point from fast fit, with some shaking
        start = self.fastfit()
        starts = [
            np.random.normal(start, 0.2 * np.abs(start))
            for _ in range(n_chains)
        ]

        # Crash now if you want to crash
        log_lhood = (
            self.log_lhood_covmat if self.has_covmat else self.log_lhood
        )
        _ = log_post(starts[0], log_lhood, self.model.log_prior)

        # ==== MCMC sampling ==== #
        np.seterr(all="ignore")
        old_tau = 0.0
        tau_was_stable = False
        all_taus = [[], []]
        print(
            f"I'll check convergence every {n_check} steps, and "
            + f"stop when the autocorrelation length `tau` has changed by "
            + f"less than {100*max_delta_tau:.1f}% twice in a row, and the "
            + f"chain is longer than {min_autocorr_times}*tau"
        )
        with Pool(processes=n_threads) as pool:
            ti = time.time()
            sampler = emcee.EnsembleSampler(
                n_chains,
                len(starts[0]),
                log_post,
                pool=pool,
                moves=emcee.moves.DEMove(),
                args=[log_lhood, self.model.log_prior],
            )

            for sample in sampler.sample(
                starts, iterations=max_steps, progress=True
            ):

                it = sampler.iteration
                if it % n_check != 0.0:
                    continue
                # The following is only executed if it = n * ncheck
                tau = sampler.get_autocorr_time(tol=0)
                mean_tau = np.mean(tau)
                all_taus[0].append(it)
                all_taus[1].append(mean_tau)
                dtau = np.abs(old_tau - mean_tau) / mean_tau
                tau_is_stable = dtau < max_delta_tau
                chain_is_long = it > (mean_tau * min_autocorr_times)

                print(tau_is_stable, tau_was_stable, chain_is_long)
                print(
                    f"    {it} iterations = {it / mean_tau:.1f}*tau",
                    f"(tau = {mean_tau:.1f} -> dtau/tau = {dtau:.4f})",
                    end="\n",
                )

                if tau_is_stable and tau_was_stable and chain_is_long:
                    print("    -> Convergence achieved")
                    break

                old_tau = mean_tau
                tau_was_stable = tau_is_stable

            rt = time.time() - ti
            rt_h = int(rt / 3600)
            rt_m = int((rt - (3600 * rt_h)) / 60)
            rt_s = int(rt - (3600 * rt_h) - (60 * rt_m))
            print(f"Running time: {rt_h:02}h{rt_m:02}m{rt_s:02}s")

            self.autocorr = np.array(all_taus)

            blobs = sampler.get_blobs()
            ch = sampler.chain
            chains = {}
            for p, i in self.model.indices.items():
                chains[p] = ch[:, :, i]
            del ch
            chains["lnprior"] = blobs[:, :, 0].T
            chains["lnlike"] = blobs[:, :, 1].T

        np.seterr(all="warn")
        np.savez(out_chains_file, **chains)

        return chains

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


def log_post(par_vec, log_lhood, log_prior):
    lprior = log_prior(par_vec)
    if not np.isfinite(lprior):
        llhood = -np.inf
    else:
        llhood = log_lhood(par_vec)
    return llhood + lprior, lprior, llhood
