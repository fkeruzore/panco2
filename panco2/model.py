import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.constants import sigma_T, c, m_e
from astropy.wcs.utils import skycoord_to_pixel
from shell_pl import shell_pl
import utils
import pdb
from filtering import Filter

sz_fact = (sigma_T / (m_e * c**2)).to(u.cm**3 / u.keV / u.kpc).value
del sigma_T, c, m_e


class Model:
    def __init__(self, radii, zero_level=True):
        self.radii = radii
        self.zero_level = zero_level
        self.indices = {}
        self.n_ps = 0
        self._priors = {}
        self._type = "binned"
        self._filter = Filter(101, 1.0)

    # ---------------------------------------------------------------------- #

    @property
    def type(self):
        return self._type

    @property
    def params(self):
        return list(self.indices.keys())

    @property
    def priors(self):
        return self._priors

    @priors.setter
    def priors(self, value):
        self._priors = value

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        self._filter = value

    # ---------------------------------------------------------------------- #

    def par_vec2dic(self, vec):
        return {key: vec[self.indices[key]] for key in self.indices.keys()}

    def par_dic2vec(self, dic):
        vec = np.zeros(len(dic))
        for p, i in self.indices.items():
            vec[i] = dic[p]
        return vec

    # ---------------------------------------------------------------------- #

    def log_prior(self, par_vec):
        lp = [
            self.priors[k].logpdf(par_vec[i]) for k, i in self.indices.items()
        ]
        return np.sum(lp)

    # ---------------------------------------------------------------------- #

    def add_point_sources(self, coords, wcs, beam_sigma_pix):
        n_pix = wcs.array_shape[0]
        self.n_ps = len(coords)
        self.ps_pix_pos = []
        self.ps_xymaps = {"x": [], "y": []}

        old_n_params = len(self.indices)
        self.indices_ps = slice(old_n_params, old_n_params + self.n_ps)
        for i, c in enumerate(coords):
            pix_pos = skycoord_to_pixel(c, wcs)
            self.ps_pix_pos.append(pix_pos)
            xmap, ymap = np.meshgrid(
                np.arange(n_pix) - pix_pos[0], np.arange(n_pix) - pix_pos[1]
            )
            self.ps_xymaps["x"].append(xmap)
            self.ps_xymaps["y"].append(ymap)

            self.indices[f"F_{i + 1}"] = old_n_params + i

        self.ps_size = beam_sigma_pix

    # ---------------------------------------------------------------------- #

    def ps_map(self, par_vec):
        fluxes = par_vec[self.indices_ps]
        ps_map = np.zeros_like(self.radii["r_xy"])
        for f, x, y in zip(fluxes, self.ps_xymaps["x"], self.ps_xymaps["y"]):
            m = f * np.exp(
                -0.5 * ((x / self.ps_size) ** 2 + (y / self.ps_size) ** 2)
            )
            ps_map += m
        return ps_map


# -------------------------------------------------------------------------- #


class ModelBinned(Model):
    def __init__(self, r_bins, radii, zero_level=True):
        super().__init__(radii, zero_level=zero_level)
        self._type = "binned"
        self.r_bins = r_bins
        self.n_bins = len(r_bins)
        self.indices_press = slice(0, self.n_bins)
        for i in range(self.n_bins):
            self.indices[f"P_{i}"] = i
        self.indices["conv"] = len(self.indices.keys())
        if self.zero_level:
            self.indices["zero"] = len(self.indices.keys())

    def pressure_profile(self, r, par_vec):
        P_i = par_vec[self.indices_press]
        return utils.interp_powerlaw(self.r_bins, P_i, r)

    def compute_slopes(self, P_i):
        """
        Slopes of a non-parametric profile.
        See Romero et al. 2018.
        """
        lr_bins = np.log(self.r_bins)
        lp_bins = np.log(P_i)

        alphas = -np.ediff1d(lp_bins) / np.ediff1d(lr_bins)
        alphas = np.concatenate(
            ([alphas[0]], alphas, [alphas[-1]])
        )  # Fill the first value for extrapolation
        return alphas

    def compton_prof(self, P_i, radarr, alphas):
        """
        Analytical integral of a non-parametric profile.
        See Romero et al. 2018.
        """

        integrals = np.zeros((self.r_bins.shape[0], radarr.shape[0]))
        r_bins_integ = np.concatenate(([0.0], self.r_bins, [-1.0]))

        # Integrate
        for i in range(len(P_i)):
            alpha_i = alphas[i]
            integrals[i] = shell_pl(
                P_i[i],
                alpha_i,
                r_bins_integ[i],
                r_bins_integ[i + 1],
                radarr,
            )

        integrals = integrals * sz_fact
        totals = np.sum(integrals, axis=0)

        return totals

    def compton_map(self, par_vec):
        P_i = par_vec[self.indices_press]
        alphas = self.compute_slopes(P_i)
        y_prof = self.compton_prof(P_i, self.radii["r_x"][1:], alphas)
        y_map = utils.prof2map(
            y_prof, self.radii["r_x"][1:], self.radii["r_xy"]
        )
        return y_map

    def sz_map(self, par_vec):
        y_map = self.compton_map(par_vec)
        sz_map_filt = self.filter(y_map) * par_vec[self.indices["conv"]]
        return sz_map_filt + par_vec[self.indices["zero"]]
