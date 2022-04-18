import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.constants import sigma_T, c, m_e
from shell_pl import shell_pl
import _utils
import pdb

sz_fact = (sigma_T / (m_e * c**2)).to(u.cm**3 / u.keV / u.kpc).value
del sigma_T, c, m_e


class ModelNonParam(Model):
    """
    ``_model.Model`` in the case you want to perform a
    non-parametric fit on your data.
    """

    # ---------------------------------------------------------------------- #
    # ====  INITIALIZATIONS  =============================================== #
    # ---------------------------------------------------------------------- #

    def __init__(self,
        reso=3.0,
        npix=101,
        r_bins=None,
        n_bins=6,
        **kwargs
    ):
        """
        Initialize the radii arrays necessary to compute profiles.

        Args:
        reso : float 
            pixel size in arcsec,
        npix : int 
            number of pixels,
        center : tuple 
            pixel offset of the point at which you want
                your profiles to be computed (not to be confused with
                coords_center from ``panco_params.py``)
        r_bins : array or None)
            the radial binning of the
                profile you want to fit if you already defined it.
        """

        super().__init__(cluster, **kwargs)

        d_a = self.cluster.d_a.value
        R_500 = self.cluster.R_500_kpc
        reso_rad = reso * u.arcsec.to("rad")
        self.reso_kpc = d_a * np.tan(reso_rad)

        # ===== Init pressure bins ===== #
        rad_tab_min = (
            0.5 * d_a * np.tan(18.0 * u.arcsec.to("rad"))
        )  # 1st bin: NIKA2 2mm HWHM

        if r_bins is None:
            # HWHM, then log-spaced from 3HWHM to R_500 with n bins
            self.r_bins = np.concatenate(
                (
                    [rad_tab_min],
                    np.logspace(
                        np.log10(3 * rad_tab_min), np.log10(R_500), n_bins
                    ),
                )
            )
            # Add (2*R_500, 5*R_500) at the end
            self.r_bins = np.append(self.r_bins, [2 * R_500])
        else:
            self.r_bins = r_bins

        self.r_bins_cm = self.r_bins * u.kpc.to("cm")
        self.nbins = self.r_bins.size

        # ===== Radii arrays ===== #
        # 1D radius in the sky plane, only half the map
        theta_x = (
            np.arange(0, int(npix / 2) + 1) * reso_rad
        )  # angle, in radians
        r_x = d_a * np.tan(theta_x)  # distance, in kpc

        # 2D (x, y) radius in the sky plane to compute compton map
        r_xy = np.hypot(
            *np.meshgrid(
                np.concatenate((-np.flip(r_x[1:]), r_x))
                - d_a * np.tan(center[0] * reso_rad),
                np.concatenate((-np.flip(r_x[1:]), r_x))
                - d_a * np.tan(center[1] * reso_rad),
            )
        )

        # ===== Integrated signal ===== #
        if mode == "500":
            integ_Y = self.cluster.Y_500_kpc2
            err_integ_Y = self.cluster.err_Y_500_kpc2
            r_max_integ_Y = R_500
        elif mode == "tot":
            integ_Y = 1.796 * self.cluster.Y_500_kpc2
            err_integ_Y = 1.796 * self.cluster.err_Y_500_kpc2
            r_max_integ_Y = 5.0 * R_500

        r_integ_Y = [0.0]
        r_integ_Y += [r for r in self.r_bins if r < (r_max_integ_Y - 1.0)]
        r_integ_Y += [r_max_integ_Y]
        r_integ_Y = np.array(r_integ_Y)

        self.init_prof = {
            "reso": reso,
            "r_x": r_x,
            "r_xy": r_xy,
            "theta_x": theta_x * u.rad.to("arcmin"),
            "r_integ_Y": r_integ_Y,
            "integ_Y": integ_Y,
            "err_integ_Y": err_integ_Y,
        }

    # ---------------------------------------------------------------------- #

    def __call__(self, par):
        return super().__call__(par)

    # ---------------------------------------------------------------------- #

    def dict_to_params(self, dic):
        """
        Given a dict describing a parameter vector, returns a vector.
        """
        params = [dic["P" + str(i)] for i in range(self.nbins)]
        params.append(dic["calib"])
        if self.zero_level:
            params.append(dic["zero"])
        if self.do_ps:
            for f in dic["ps_fluxes"]:
                params.append(f)
        return params

    # ---------------------------------------------------------------------- #

    def init_param_indices(self):
        """
        Generates the names of parameters of your model
        depending on the options you use.  This function makes
        ``self.params_to_dict`` and ``self.dict_to_params`` work.
        """
        self.param_names = ["$P_{" + str(i) + "}$" for i in range(self.nbins)]
        self.param_names.append("Calib")
        self.indices_press = slice(0, self.nbins)

        self.indices = {"P" + str(i): i for i in range(self.nbins)}
        self.indices["calib"] = np.max(list(self.indices.values())) + 1

        if self.zero_level:
            self.indices["zero"] = np.max(list(self.indices.values())) + 1
            self.param_names.append("Zero")
        i = np.max(list(self.indices.values()))  # Last atributed index

        if self.do_ps:
            self.indices["ps_fluxes"] = slice(i, None)
            for i in range(self.init_ps["nps"]):
                self.param_names.append("$F_{" + str(i + 1) + "}$")
        self.nparams = len(self.param_names)

    # ---------------------------------------------------------------------- #
    # ====  MODEL COMPUTATION RELATED FUNCTIONS  =========================== #
    # ---------------------------------------------------------------------- #

    def pressure_profile(self, r, par):
        """
        P(r) = P_i * (r / r_i) ^ -alpha
        """
        params = self.dict_to_params(par)
        press = np.array(params[self.indices_press])
        return _utils.interp_powerlaw(self.r_bins, press, r)

    # ---------------------------------------------------------------------- #

    def deriv_pressure_profile(self, r, par):
        """
        dP/dr = d/dr (P_i * (r / r_i) ^ -alpha)
              = - P_i * (R_i ^ alpha) * alpha * r ^ (-alpha - 1)
        """
        params = self.dict_to_params(par)

        # ===== Pressure and radius bins ===== #
        p_bins = np.array(params[self.indices_press])
        r_bins = self.r_bins
        lr_bins = np.log(r_bins)
        lp_bins = np.log(p_bins)

        # ===== The powers (slopes in log-log) ===== #
        alpha_bins = -np.ediff1d(lp_bins) / np.ediff1d(lr_bins)
        alpha_bins = np.concatenate(([alpha_bins[0]], alpha_bins))

        # ===== For each r, X_i is the previous bin, with X = (R, P, alpha) ===== #
        R_i = interp1d(
            r_bins,
            r_bins,
            bounds_error=False,
            fill_value="extrapolate",
            kind="previous",
        )(r)
        P_i = interp1d(
            r_bins,
            p_bins,
            bounds_error=False,
            fill_value="extrapolate",
            kind="previous",
        )(r)
        alpha_i = interp1d(
            r_bins,
            alpha_bins,
            bounds_error=False,
            fill_value="extrapolate",
            kind="previous",
        )(r)

        return -P_i * alpha_i * (R_i**alpha_i) * r ** (-alpha_i - 1)

    # ---------------------------------------------------------------------- #

    def check_mass(self, *args, **kwargs):
        return True

    # ---------------------------------------------------------------------- #

    def compute_model_SZ(self, par):
        params = self.dict_to_params(par)
        press = params[self.indices_press]

        alphas = compute_slopes(self.r_bins, press)
        y_prof = compton_prof(
            self.r_bins, press, self.init_prof["r_x"][1:], alphas
        )

        Y_500_model = 0.0
        r_integ = self.init_prof["r_integ_Y"]

        # You need another bin if your integration radius is outside r_bins
        if r_integ.max() > self.r_bins.max():
            press = np.concatenate(
                (
                    press,
                    [
                        press[-1]
                        * (r_integ.max() / self.r_bins.max())
                        ** (-alphas[-1])
                    ],
                )
            )
        # This loop could be optimized, TODO
        for i in range(len(r_integ) - 1):
            Y_500_model += (
                4.0
                * np.pi
                * press[i]
                * r_integ[i + 1] ** alphas[i]
                / (3.0 - alphas[i])
                * (
                    r_integ[i + 1] ** (3.0 - alphas[i])
                    - r_integ[i] ** (3.0 - alphas[i])
                )
            )
        Y_500_model *= sz_fact

        y_map = _utils.prof2map(
            y_prof, self.init_prof["r_x"][1:], self.init_prof["r_xy"]
        )
        y_map_filt = gaussian_filter(
            y_map, 17.6 * 0.4247 / self.init_prof["reso"]
        )

        SZ_map = par["calib"] * y_map_filt
        return SZ_map, Y_500_model

    # ---------------------------------------------------------------------- #

    def compute_integrated_SZ(self, par, r_max):
        params = self.dict_to_params(par)
        press = params[self.indices_press]

        alphas = compute_slopes(self.r_bins, press)

        Y_500_model = 0.0
        r_integ = self.init_prof["r_integ_Y"][
            self.init_prof["r_integ_Y"] < r_max
        ]
        r_integ = np.concatenate((r_integ, [r_max]))

        # You need another bin if your integration radius is outside r_bins
        if r_integ.max() > self.r_bins.max():
            press = np.concatenate(
                (
                    press,
                    [
                        press[-1]
                        * (r_integ.max() / self.r_bins.max())
                        ** (-alphas[-1])
                    ],
                )
            )
        # This loop could be optimized, TODO
        for i in range(len(r_integ) - 1):
            Y_500_model += (
                4.0
                * np.pi
                * press[i]
                * r_integ[i + 1] ** alphas[i]
                / (3.0 - alphas[i])
                * (
                    r_integ[i + 1] ** (3.0 - alphas[i])
                    - r_integ[i] ** (3.0 - alphas[i])
                )
            )
        Y_500_model *= sz_fact

        return Y_500_model


# =============================================================== #


def compute_slopes(r_bins, p_bins):
    """
    Slopes of a non-parametric profile.
    See Romero et al. 2018.
    Numpy-ed form of nk_icm_sz.cluster.power_index_nonparam
    """
    lr_bins = np.log(r_bins)
    lp_bins = np.log(p_bins)

    alphas = -np.ediff1d(lp_bins) / np.ediff1d(lr_bins)
    alphas = np.concatenate(
        ([alphas[0]], alphas, [alphas[-1]])
    )  # Fill the first value for extrapolation
    return alphas


# =============================================================== #


def compton_prof(r_bins, pressure_bins, radarr, alphas):
    """
    Analytical integral of a non-parametric profile.
    See Romero et al. 2018.
    Adapted from nk_icm_sz.cluster.Compton_analytic
    """

    integrals = np.zeros((r_bins.shape[0], radarr.shape[0]))

    r_bins_integ = np.concatenate(([0.0], r_bins, [-1.0]))

    # ===== Integrate
    for i in range(len(pressure_bins)):
        alpha_i = alphas[i]
        integrals[i] = shell_pl(
            pressure_bins[i],
            alpha_i,
            r_bins_integ[i],
            r_bins_integ[i + 1],
            radarr,
        )

    integrals = integrals * sz_fact
    totals = np.sum(integrals, axis=0)

    return totals
