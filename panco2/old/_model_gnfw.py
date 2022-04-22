import numpy as np
from scipy.ndimage import gaussian_filter
import astropy.units as u
from astropy.constants import sigma_T, c, m_e, G, m_p
from scipy.integrate import trapz
from _model import Model
import _utils
import pdb

# ===== Prefactors ===== #
sz_fact = (sigma_T / (m_e * c**2)).to(u.cm**3 / u.keV / u.kpc).value
dens_fact = G * 0.61 * m_p  # m3.s-2 ; mu=0.61, mean ICM molecular weight
del sigma_T, c, m_e, G, m_p


class ModelGNFW(Model):
    """
    ``_model.Model`` in the case you want to perform a
    gNFW fit on your data.
    """

    # ---------------------------------------------------------------------- #
    # ====  INITIALIZATIONS  =============================================== #
    # ---------------------------------------------------------------------- #

    def __init__(self, cluster, **kwargs):
        super().__init__(cluster, **kwargs)

    # ---------------------------------------------------------------------- #

    def __call__(self, par):
        return super().__call__(par)

    # ---------------------------------------------------------------------- #

    def init_profiles_radii(
        self,
        reso=3.0,
        npix=101,
        center=(0, 0),
        r_min_z=1e-3,
        r_max_z=5e3,
        nbins_z=500,
        mode="500",
        **kwargs
    ):
        """
        Initialize the radii arrays necessary to compute profiles.

        Args:
            reso (float): pixel size in arcsec,

            npix (int): number of pixels,

            center (tuple): pixel offset of the point at which you want
                your profiles to be computed (not to be confused with
                coords_center from ``panco_params.py``)

            r_min_z, r_max_z (floats): range of the line of sight considered for the
                analysis (tipycally 1 pc -> 5*R_500),

            nbins_z (int) : number of bins along the line of sight.

            mode (str) : "500" or "tot", which Y to use for large scale constraints
        """

        # Notes:
        #     Values indexed x (y) are along the RA (Dec) axes.
        #     Values indexed z are along the line of sight.
        #     Values indexed with two subscripts are in the plane subtended by
        #     the vectors described by each subscript, e.g.
        #     r_xz is in the (RA, LoS) plane
        #     r_xy is in the (RA, dec) plane

        d_a = self.cluster.d_a.value
        R_500 = self.cluster.R_500_kpc
        reso_rad = reso * u.arcsec.to("rad")
        self.reso_kpc = d_a * np.tan(reso_rad)

        # 1D radius in the sky plane, only half the map
        theta_x = (
            np.arange(0, int(npix / 2) + 1) * reso_rad
        )  # angle, in radians
        r_x = d_a * np.tan(theta_x)  # distance, in kpc

        # 1D LoS radius
        r_z = np.logspace(np.log10(r_min_z), np.log10(r_max_z), nbins_z)

        # 2D (x, z) radius plane to compute 1D compton profile
        r_xx, r_zz = np.meshgrid(r_x, r_z)
        r_xz = np.hypot(r_xx, r_zz)

        # 2D (x, y) radius in the sky plane to compute compton map
        r_xy = np.hypot(
            *np.meshgrid(
                np.concatenate((-np.flip(r_x[1:]), r_x))
                - d_a * np.tan(center[0] * reso_rad),
                np.concatenate((-np.flip(r_x[1:]), r_x))
                - d_a * np.tan(center[1] * reso_rad),
            )
        )

        # Integrated signal
        if mode == "500":
            integ_Y = self.cluster.Y_500_kpc2
            err_integ_Y = self.cluster.err_Y_500_kpc2
            r_integ_Y = np.logspace(np.log10(r_min_z), np.log10(R_500), 50)
        elif mode == "tot":
            integ_Y = 1.796 * self.cluster.Y_500_kpc2
            err_integ_Y = 1.796 * self.cluster.err_Y_500_kpc2
            r_integ_Y = np.logspace(
                np.log10(r_min_z), np.log10(5.0 * R_500), 50
            )

        self.init_prof = {
            "reso": reso,
            "r_x": r_x,
            "r_xz": r_xz,
            "r_zz": r_zz,
            "r_xy": r_xy,
            "theta_x": theta_x * u.rad.to("arcmin"),
            "r_integ_Y": r_integ_Y,
            "integ_Y": integ_Y,
            "err_integ_Y": err_integ_Y,
        }

    # ---------------------------------------------------------------------- #

    def init_param_indices(self, nocalib=False):
        """
        Generates the names of parameters of your model
        depending on the options you use.  This function makes
        ``self.params_to_dict`` and ``self.dict_to_params`` work.
        """
        self.param_names = ["$P_0$", "$r_p$", "$a$", "$b$", "$c$"]
        self.indices = {"P0": 0, "rp": 1, "a": 2, "b": 3, "c": 4}
        if nocalib:
            self.indices_gNFW = slice(0, 4)
        else:
            self.param_names.append("Calib")
            self.indices["calib"] = 5
            self.indices_gNFW = slice(0, 5)

        i = len(self.param_names)
        if self.zero_level:
            self.indices["zero"] = i + 1
            self.param_names.append("Zero")
        i = len(self.param_names)  # Last atributed index

        if self.do_ps:
            self.indices["ps_fluxes"] = slice(i, None)
            for i in range(self.init_ps["nps"]):
                self.param_names.append("$F_{" + str(i + 1) + "}$")
        self.nparams = len(self.param_names)

    # ---------------------------------------------------------------------- #

    def dict_to_params(self, dic):
        """
        Given a dict describing a parameter vector, returns a vector.
        """
        params = [
            dic["P0"],
            dic["rp"],
            dic["a"],
            dic["b"],
            dic["c"],
            dic["calib"],
        ]
        if self.zero_level:
            params.append(dic["zero"])
        if self.do_ps:
            for f in dic["ps_fluxes"]:
                params.append(f)
        return np.array(params)

    # ---------------------------------------------------------------------- #
    # ====  MODEL COMPUTATION RELATED FUNCTIONS  =========================== #
    # ---------------------------------------------------------------------- #

    def pressure_profile(self, r, par):
        return gNFW(
            r,
            par["P0"],
            par["rp"],
            par["a"],
            par["b"],
            par["c"],
        )

    # ---------------------------------------------------------------------- #

    def deriv_pressure_profile(self, r, par):
        return d_gNFW_d_r(
            r,
            par["P0"],
            par["rp"],
            par["a"],
            par["b"],
            par["c"],
        )

    # ---------------------------------------------------------------------- #

    def check_mass(self, par, x_profiles=None):
        """
        Given a pressure profile (par) and a density profile (in x_profiles),
        evaluates dM_HSE / dr for all density points, and returns wether or not
        the slope of the mass profile is always positive.
        """
        if x_profiles is None:
            return True
        else:
            r = x_profiles["rd"]
            mass_prof = HSE_mass(
                r,
                x_profiles["d"],
                par["P0"],
                par["rp"],
                par["a"],
                par["b"],
                par["c"],
            )
            d_mass_d_r = np.ediff1d(mass_prof) / np.ediff1d(r)
            return not np.any(d_mass_d_r < 0)

    # ---------------------------------------------------------------------- #

    def compute_model_SZ(self, par):
        """
        Model of the SZ map
        """
        y_prof = compton_prof(
            par["P0"],
            par["rp"],
            par["a"],
            par["b"],
            par["c"],
            self.init_prof["r_xz"],
            self.init_prof["r_zz"],
            # self.init_prof["dr_zz"],
        )

        Y_500_model = self.sz_fact * _utils.sph_integ_within(
            gNFW(
                self.init_prof["r_integ_Y"],
                par["P0"],
                par["rp"],
                par["a"],
                par["b"],
                par["c"],
            ),
            self.init_prof["r_integ_Y"],
        )

        y_map = _utils.prof2map(
            y_prof, self.init_prof["r_x"], self.init_prof["r_xy"]
        )
        y_map_filt = gaussian_filter(
            y_map, 17.6 * 0.4247 / self.init_prof["reso"]
        )

        SZ_map = par["calib"] * y_map_filt
        return SZ_map, Y_500_model

    # ---------------------------------------------------------------------- #

    def compute_integrated_SZ(self, par, r_max):
        r_integ = self.init_prof["r_integ_Y"][
            self.init_prof["r_integ_Y"] < r_max
        ]
        r_integ = np.concatenate((r_integ, [r_max]))
        Y_500_model = self.sz_fact * _utils.sph_integ_within(
            gNFW(
                r_integ,
                par["P0"],
                par["rp"],
                par["a"],
                par["b"],
                par["c"],
            ),
            r_integ,
        )
        return Y_500_model


# ---------------------------------------------------------------------- #
# ====  END OF CLASS, MISC FUNCTIONS FOR MODEL COMPUTATIONS  =========== #
# ---------------------------------------------------------------------- #


def gNFW(r, P0, rp, a, b, c):
    """
    gNFW pressure profile.

    Inputs :
    --------
    - r : radii (can be a scalar or a ndarray),
    - P0, rp, a, b, c : parameters

    Outputs :
    ---------
    - pressure evaluated at each radius (same shape as the input r)
    """

    x = r / rp
    return P0 / ((x**c) * (1.0 + x**a) ** ((b - c) / a))


# ---------------------------------------------------------------------- #


def integ_LoS(prof_rz, r_zz):
    """
    Integrate a profile along the line of sight.

    Inputs :
    --------
    - prof_rz : the profile evaluated in the (sky, LoS) plane,
    - r_zz : LoS radii in the LoS plane, in the same shape as
             prof_rz [kpc].
             Equivalent to the line of sight r_z repeated for
             each value of r_x,

    Outputs :
    ---------
    The pressure integrated along z - shape = shape of prof_rz along
        the r axis
    """
    return trapz(prof_rz, r_zz, axis=0)


# ---------------------------------------------------------------------- #


def compton_prof(P0, rp, a, b, c, r_xz, r_zz):
    """
    Compton parameter profile.
    See :
        gNFW(*args)
        integ_LoS(*args)
    for documentation.
    """
    P = gNFW(r_xz, P0, rp, a, b, c)
    return 2.0 * sz_fact * integ_LoS(P, r_zz)


# ---------------------------------------------------------------------- #


def compton_map(
    P0, rp, a, b, c, r_x=None, r_xz=None, r_zz=None, r_xy=None, **kwargs
):
    """
    Compton parameter map.
    See :
        compton_prof(*args)
        prof2map(*args)
    for documentation.
    """
    y_prof = compton_prof(P0, rp, a, b, c, r_xz, r_zz)
    y_map = prof2map(y_prof, r_x, r_xy)
    return y_map


# ---------------------------------------------------------------------- #


def d_gNFW_d_r(r, P0, rp, a, b, c):
    """
    Analytical derivative of the gNFW pressure profile

    Inputs :
    --------
    - r : radii (can be a scalar or a ndarray),
    - P0, rp, a, b, c : parameters

    Outputs :
    ---------
    - dp/dr evaluated at each radius (same shape as the input r)
    """

    x = r / rp
    return -P0 * c * x ** (-c) * (x**a + 1.0) ** (
        (-b + c) / a
    ) / r + P0 * x**a * x ** (-c) * (-b + c) * (x**a + 1.0) ** (
        (-b + c) / a
    ) / (
        r * (x**a + 1.0)
    )


# ---------------------------------------------------------------------- #


def HSE_mass(r, n, P0, rp, a, b, c):
    """
    Hydrostatic mass profile from gNFW parameters.
    """
    dpdr = d_gNFW_d_r(r, P0, rp, a, b, c)
    mass = (
        (r * u.kpc) ** 2
        * dpdr
        * u.Unit("keV cm-3 kpc-1")
        / (dens_fact * n * u.Unit("cm-3"))
    )
    return -1.0 * mass.to("Msun").value
