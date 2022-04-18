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


class Model:
    def __init__(self, zero_level=True):
        self.zero_level = zero_level
        self.indices = {}

    def par_vec2dic(self, vec):
        return {key: vec[self.indices[key]] for key in self.indices.keys()}


class ModelBinned(Model):
    def __init__(self, r_bins, zero_level=True):
        super().__init__(zero_level=zero_level)
        self.r_bins = r_bins
        self.n_bins = len(r_bins)
        self.indices_press = range(self.n_bins)
        for i in range(self.n_bins):
            self.indices[f"P_{i}"] = i
        self.indices["conv"] = len(self.indices.keys())
        if self.zero_level:
            self.indices["zero"] = len(self.indices.keys())

    def pressure_profile(self, r, par_vec):
        P_i = par_vec[self.indices_press]
        return _utils.interp_powerlaw(self.r_bins, P_i, r)

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

    def compton_map(self, par_vec, radii):
        P_i = par_vec[self.indices_press]
        alphas = self.compute_slopes(P_i)
        y_prof = self.compton_prof(P_i, radii["r_x"][1:], alphas)
        y_map = _utils.prof2map(y_prof, radii["r_x"][1:], radii["r_xy"])
        return y_map
