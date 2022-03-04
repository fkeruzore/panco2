import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import sigma_T, c, m_e
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import time
from threadpoolctl import threadpool_limits
import pdb


# ====================== #
# ----- Likelihood ----- #
# ====================== #


def log_lhood_nocovmat(par, model, data, error):
    """
    Log-likelihood computation with no covariance matrix inclusion.

    Args:
        par (dict): the model parameter values at which to evaluate
            the likelihood.

        model (_model.Model child): a ``Model`` instance to be used to
            compute the model map and integrated signal.

        data (array): your input map.

        error (array): your 1sigma error map.

    Returns:
        float: the log-likelihood
    """

    model_map, Y_integ = model(par)

    ll = (
        np.sum(((data - model_map) / error) ** 2)
        + ((model.init_prof["integ_Y"] - Y_integ) / model.init_prof["err_integ_Y"]) ** 2
    )
    return -0.5 * ll


def log_lhood_covmat(par, model, data, inv_covmat):
    """
    Log-likelihood computation with covariance matrix inclusion.

    Args:
        par (dict): the model parameter values at which to evaluate
            the likelihood.

        model (_model.Model child): a ``Model`` instance to be used to
            compute the model map and integrated signal.

        data (array): your input map.

        inv_covmat (array): the inverse of your pixel-to-pixel noise
            covariance matrix.

    Returns:
        float: the log-likelihood
    """

    model_map, Y_integ = model(par)

    dmm = (data - model_map).flatten()  # data minus model vector
    with threadpool_limits(limits=1, user_api="blas"):
        ll = (
            dmm @ inv_covmat @ dmm  # `@` is the matrix product
            + ((model.init_prof["integ_Y"] - Y_integ) / model.init_prof["err_integ_Y"])
            ** 2
        )
    return -0.5 * ll


# ====================== #
# -----   Priors   ----- #
# ====================== #


class Prior:
    """
    Class to compute the priors on your MCMC parameters.

    Args:
        model (_model.Model child):

        ref (None): name of your favorite universal pressure
            profile (implemented: "A10", "Planck")

        gNFW_values ((array) or None): if you don't want to use a
            universal pressure profile, the central values of
            P0, rp, a, b, c for your priors.

        sigma (float): Dispersion around the parameters of the gNFW,
            in fraction of the central value (1.0 = 100%).

        flat (bool): wether you want to use flat priors on the
            gNFW parameters.

        calib (tuple): the central value and dispersion for your
            y-to-Jy/beam conversion coefficient.

        a, b, c (tuple or None): override the priors for the slopes
            of the gNFW.

        nonparam(bool) : answer to "Are you fitting a non-parametric
            pressure profile?"
    """

    def __init__(
        self,
        model,
        ref="A10",
        gNFW_values=None,
        sigma=0.5,
        flat=False,
        calib=(-12.0, 1.2),
        a=None,
        b=None,
        c=None,
        nonparam=False,
    ):
        self.fit_zl = model.fit_zl
        self.do_ps = model.do_ps
        if self.do_ps:
            self.interp_prior_ps = model.init_ps["interp_prior_ps"]
            self.nps = model.init_ps["nps"]
        self.calib = calib

        self.nonparam = nonparam

        if not self.nonparam:
            if ref == "A10":
                self.P0 = (
                    8.403 * model.cluster.P_500.value,
                    sigma * 8.403 * model.cluster.P_500.value,
                )
                self.rp = (
                    model.cluster.R_500.value / 1.177,
                    sigma * model.cluster.R_500.value / 1.177,
                )
                self.a = (1.0510, sigma * 1.0510)
                self.b = (5.4905, sigma * 5.4905)
                self.c = (0.3081, sigma * 0.3081)
            elif ref == "Planck":
                self.P0 = (
                    6.41 * model.cluster.P_500.value,
                    sigma * 6.41 * model.cluster.P_500.value,
                )
                self.rp = (
                    model.cluster.R_500.value / 1.81,
                    sigma * model.cluster.R_500.value / 1.81,
                )
                self.a = (1.33, sigma * 1.33)
                self.b = (4.13, sigma * 4.13)
                self.c = (0.31, sigma * 0.31)
            elif ref is None and gNFW_values is not None:
                self.P0 = (gNFW_values[0], sigma * gNFW_values[0])
                self.rp = (gNFW_values[1], sigma * gNFW_values[1])
                self.a = (gNFW_values[2], sigma * gNFW_values[2])
                self.b = (gNFW_values[3], sigma * gNFW_values[3])
                self.c = (gNFW_values[4], sigma * gNFW_values[4])
            else:
                raise Exception("Unrecognized priors.")

            # ===== Possibly overwrite ===== #
            if a is not None:
                self.a = a
            if b is not None:
                self.b = b
            if c is not None:
                self.c = c

            # ===== Flat ? ===== #
            self.flat = flat

        # ===== Creates the __compute_priors function ===== #
        if self.nonparam:
            self.__log_prior_SZ = self.__log_prior_SZ_nonparam
        elif self.flat:
            self.__log_prior_SZ = self.__log_prior_SZ_flat
        else:
            self.__log_prior_SZ = self.__log_prior_SZ_gauss

        if (self.fit_zl) and (self.do_ps):

            def compute_priors(par):
                return (
                    self.__log_prior_SZ(par)
                    + self.__log_prior_ps(**par)
                    - 0.5 * (par["zero"] / 5e-4) ** 2
                )

        elif (not self.fit_zl) and (self.do_ps):

            def compute_priors(par):
                return self.__log_prior_SZ(par) + self.__log_prior_ps(**par)

        elif (self.fit_zl) and (not self.do_ps):

            def compute_priors(par):
                return self.__log_prior_SZ(par) - 0.5 * (par["zero"] / 5e-4) ** 2

        elif (not self.fit_zl) and (not self.do_ps):

            def compute_priors(par):
                return self.__log_prior_SZ(par)

        self.__compute_priors = compute_priors

    # -------------------------------------------------------------------- #

    def __log_prior_SZ_gauss(self, par):
        lp = [
            positive_norm(par["P0"], *self.P0),
            positive_norm(par["rp"], *self.rp),
            positive_norm(par["a"], *self.a),
            positive_norm(par["b"], *self.b),
            positive_norm(par["c"], *self.c),
            -0.5 * ((par["calib"] - self.calib[0]) / self.calib[1]) ** 2,
        ]

        return np.sum(lp)

    # -------------------------------------------------------------------- #

    def __log_prior_SZ_flat(self, par):
        lp = [
            0.0 if 0.0 < par["P0"] < 5.0 * self.P0[0] else -np.inf,
            0.0 if 0.0 < par["rp"] < 5.0 * self.rp[0] else -np.inf,
            0.0 if 0.0 < par["a"] < 5.0 * self.a[0] else -np.inf,
            0.0 if 0.0 < par["b"] < 5.0 * self.b[0] else -np.inf,
            0.0 if 0.0 < par["c"] < 5.0 * self.c[0] else -np.inf,
            -0.5 * ((par["calib"] - self.calib[0]) / self.calib[1]) ** 2,
        ]

        return np.sum(lp)

    # -------------------------------------------------------------------- #

    def __log_prior_SZ_nonparam(self, par):
        is_pos = [par[k] >= 0.0 for k in par.keys() if k.startswith("P")]

        lp = [
            0.0 if np.all(is_pos) else -np.inf,
            -0.5 * ((par["calib"] - self.calib[0]) / self.calib[1]) ** 2,
        ]

        return np.sum(lp)

    # -------------------------------------------------------------------- #

    def __log_prior_ps(self, ps_fluxes=[0], **kwargs):
        return np.sum([self.interp_prior_ps[i](ps_fluxes[i]) for i in range(self.nps)])

    # -------------------------------------------------------------------- #

    def __call__(self, par):
        """
        Compute the log-prior on yout model parameters.

        Args:
            par (dict): the model parameter values at which to evaluate
                the likelihood.

        Returns:
            float: the log-prior
        """

        return self.__compute_priors(par)

    # -------------------------------------------------------------------- #


def positive_norm(x, mu, sigma):
    return -np.inf if x <= 0.0 else -0.5 * ((x - mu) / sigma) ** 2
