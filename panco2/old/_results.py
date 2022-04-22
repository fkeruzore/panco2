import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from astropy.io import fits
import astropy.units as u
from astropy.constants import sigma_T, c, m_e, G, m_p
import emcee
from iminuit import Minuit
from joblib import Parallel, delayed
import _probability
import _model_gnfw, _model_nonparam
import _utils
from _utils import LogLogSpline, interp_powerlaw
from chainconsumer import ChainConsumer
import pdb
import os
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

_utils.ignore_astropy_warnings()
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{txfonts}")

errb_kwargs = {
    "ms": 5,
    "capsize": 2,
    "elinewidth": 1,
    "capthick": 1,
    "mew": 1,
}

cmap_plots = "RdBu_r"

sz_fact = (sigma_T / (m_e * c ** 2)).to(u.cm ** 3 / u.keV / u.kpc).value
dens_fact = G * 0.61 * m_p  # m3.s-2 ; mu=0.61, mean ICM molecular weight
del sigma_T, c, m_e, G, m_p

# =============================================================================== #


class Results:
    """
    A class managing the results of the MCMC, and their
    representation.

    Args:
        out_chains_file (str): the path to the latest save
            of your Markov chains.

        burn (int): the burn-in period.

        model (_model.Model child):

        path (str): where to save the results and plots.

        x_profiles (dict): a parprod-generated dict (see ``_xray``)

        truth (dict or None): if you ran the fit on a simulation,
            the true value of the parameters.

        dof (int): the number of degrees of freedom in your statistical
            model to compute the reduced chi-squared.
    """

    def __init__(
        self,
        out_chains_file,
        burn,
        model,
        path,
        x_profiles=None,
        truth=None,
        dof=1,
    ):
        self.out_chains_file = out_chains_file
        self.burn = burn
        self.path = path
        self.plot_path = os.path.join(self.path, "Plots/")
        if not os.path.isdir(self.plot_path):
            os.mkdir(self.plot_path)
        self.x_profiles = x_profiles

        # ===== What kind of model are we talking about? ===== #
        if isinstance(model, _model_nonparam.ModelNonParam):
            self.model_type = "nonparam"
        else:
            self.model_type = "gnfw"
        self.model = model

        # ===== Truth ===== #
        """
        Ok so truth is tricky.
        Even if you are doing a nonparam, your truth will most likely
        be a gNFW profile. So the two cases must be treated differently.
        """
        if self.model_type == "gnfw":
            if isinstance(truth, dict):
                self.truth = truth
                self.truth_as_list = model.dict_to_params(truth)
            else:
                self.truth = None
                self.truth_as_list = None

        elif self.model_type == "nonparam":
            if isinstance(truth, dict):
                self.truth = truth
                self.truth_as_list = [None for _ in range(model.nbins)]
                self.truth_as_list.append(truth["calib"])
                if model.fit_zl:
                    self.truth_as_list.append(truth["zero"])
                if model.do_ps:
                    self.truth_as_list.append([p for p in truth["ps_fluxes"]])
            else:
                self.truth = None
                self.truth_as_list = None

        self.dof = dof

    # ---------------------------------------------------------------------- #

    def clean_chains(
        self, burn_out=None, clip_at_sigma=None, clip_at_autocorr=0.0, thin_by=None
    ):
        """
        Chain management:
        remove a given burn-in time (and/or burn-out);
        if asked, clip problematic chains;
        if asked, isolate a subsample of points separated by the
        max autocorrelation length (faster for batch model computation).

        Args:
            clip_at_sigma (float): # of sigma on the posterior distribution
                beyond which you want to delete chains. Can be `None` for
                no clipping.

            clip_at_autocorr (float): min # of autocorrelation lengths to
                keep a chain. All chains that have walked shorter than that
                will be discarded.

            thin_by (str or int): only keep one point every ``thin_by``
                for each chain. Can be ``'autocorr'`` for the autocorrelation
                length of each chain.

            burn_out (int): the opposite of burn-in, discard chains after
                this length.

        Returns:
            (int): the number of chains remaining after your cuts were applied
        """
        chains_dirty = np.load(self.out_chains_file)
        self.nchains, npts, self.ndim = chains_dirty["chains"].shape

        # ===== 1) Burn-in cut ===== #
        if burn_out is None:
            burn_out = npts
        try:
            self.chains = {
                "chains": chains_dirty["chains"][:, self.burn : burn_out, :],
                "lnprob": chains_dirty["lnprob"][:, self.burn : burn_out],
                "lnlike": chains_dirty["lnlike"][:, self.burn : burn_out],
            }
        except:  # in case you are loading old results without lnlike stored
            self.chains = {
                "chains": chains_dirty["chains"][:, self.burn :, :],
                "lnprob": chains_dirty["lnprob"][:, self.burn :],
            }
        chains_dirty.close()

        # ===== 2) Remove problematic chains ===== #
        if clip_at_sigma is not None:
            avg_post_per_chain = np.average(self.chains["lnprob"], axis=1)
            median_of_that = np.median(avg_post_per_chain)
            std_of_that = np.std(avg_post_per_chain, ddof=1)
            is_kept = (
                (median_of_that - avg_post_per_chain) / std_of_that
            ) < clip_at_sigma
            self.chains["chains"] = self.chains["chains"][is_kept, :, :]
            self.chains["lnprob"] = self.chains["lnprob"][is_kept, :]
            try:
                self.chains["lnlike"] = self.chains["lnlike"][is_kept, :]
            except:
                pass
            self.nchains = np.sum(is_kept)
            print(
                "    Removed {:.0f} chains because of bad posterior values".format(
                    is_kept.size - is_kept.sum()
                )
            )

        # ===== 3) Autocorrelation analysis ===== #
        n_chains = self.chains["chains"].shape[0]
        n_steps = self.chains["chains"].shape[1]
        n_params = self.chains["chains"].shape[2]

        all_autocorr_times = [
            float(emcee.autocorr.integrated_time(self.chains["chains"][i, :, j], tol=0))
            for i in range(n_chains)
            for j in range(n_params)
        ]
        all_autocorr_times = np.array(all_autocorr_times).reshape(n_chains, n_params)
        max_autocorr_times = np.array(np.max(all_autocorr_times, axis=1), dtype=int)

        # 3.1) If asked, remove chains with too little autocorr lengths walked
        if clip_at_autocorr != 0.0:
            is_kept = (n_steps / max_autocorr_times) > 30
            max_autocorr_times = max_autocorr_times[is_kept]
            self.max_autocorr_times = max_autocorr_times
            self.chains["chains"] = self.chains["chains"][is_kept, :, :]
            self.chains["lnprob"] = self.chains["lnprob"][is_kept, :]
            try:
                self.chains["lnlike"] = self.chains["lnlike"][is_kept, :]
            except:
                pass
        self.nchains = np.sum(is_kept)
        print(
            "    Removed {:.0f} chains because of too long autocorrelation".format(
                is_kept.size - is_kept.sum()
            )
        )

        print(
            "    Max autocorrelation length per chain:",
            "{:.0f} +/- {:.0f}".format(
                np.average(max_autocorr_times), np.std(max_autocorr_times, ddof=1)
            ),
        )

        # 3.2) If asked, keep one point every n steps
        do_thinning = thin_by is not None
        if do_thinning and thin_by == "autocorr":
            print("    Keep one point every autocorrelation length")
            indices_uncorr = [
                np.arange(0, n_steps, autocorr) for autocorr in max_autocorr_times
            ]
        elif do_thinning and isinstance(thin_by, int):
            print(f"    Keep one point every {thin_by} steps")
            indices_uncorr = [
                np.arange(0, n_steps, thin_by) for _ in max_autocorr_times
            ]

        if do_thinning:
            subsample_uncorr = [
                self.chains["chains"][i, indices_uncorr[i], :]
                for i in range(self.nchains)
            ]
            self.subsample_uncorr = np.concatenate(subsample_uncorr, axis=0)
            print(
                "    -> Total number of accepted independent points: {:.0f}".format(
                    self.subsample_uncorr.shape[0]
                )
            )

        if do_thinning:
            self.chains_flat = self.subsample_uncorr
        else:
            self.chains_flat = self.chains["chains"].reshape(-1, self.ndim)

        return self.nchains

    # ---------------------------------------------------------------------- #

    def compute_solid_statistic(self, solid):
        """
        Define the estimator to be used for the central value
        when describing a PDF.

        Args:
            solid (str): which statistic to use.
                One of ``max-like`` (maximum likelihood),
                ``max-post`` (maximum posterior),
                or ``median``.
        """

        self.solid = "median" if solid == "median" else "bf"
        chains_flat = self.chains["chains"].reshape(-1, self.ndim)

        if solid == "max-post":
            i_bf = np.argmax(self.chains["lnprob"].reshape(-1))
            bf_params = chains_flat[i_bf, :]
            self.solid_params = bf_params
            print("    Best fit params:")
        elif solid == "max-like":
            i_bf = np.argmax(self.chains["lnlike"].reshape(-1))
            bf_params = chains_flat[i_bf, :]
            self.solid_params = bf_params
            print("    Best fit params:")
        else:
            self.solid_params = np.median(chains_flat, axis=0)
            print("    Median params:")
        self.solid_par = self.model.params_to_dict(self.solid_params)
        for k in self.solid_par.keys():
            print(f"       {k} = {self.solid_par[k]}")

    # ---------------------------------------------------------------------- #

    def gelman_rubin_stat(self):

        c = self.chains["chains"]  # dimensions = [walkers, steps, params]
        if c.shape[0] == 0:
            return np.inf

        n = c.shape[1]

        var = np.var(c, axis=1, ddof=1)  # [walkers, params]
        W = np.average(var, axis=0)  # [params]

        mean = np.average(c, axis=1)  # [walkers, params]
        B = np.var(mean, axis=0, ddof=1)  # [params]

        V = (1 - 1 / n) * W + B
        R = np.sqrt(V / W)

        return R  # One value per parameter

    # ---------------------------------------------------------------------- #

    def plot_mcmc_diagnostics(self, cleaned_chains=False, param_names=None):
        """
        Plots the walks for all chains and all parameters.
        The plots are created using ChainConsumer, doc :
        https://samreay.github.io/ChainConsumer/examples/index.html
        """

        if param_names is None:
            param_names = copy(self.model.param_names)
        chains = self.chains if cleaned_chains else np.load(self.out_chains_file)

        # ===== Create a ChainConsumer instance ===== #
        cc = ChainConsumer()
        for chain in chains["chains"]:
            cc.add_chain(chain, parameters=param_names, walkers=1)
        cc.configure(cmap="tab20")

        # ===== Walks plot ===== #
        fig = cc.plotter.plot_walks(truth=self.truth_as_list)
        fig.align_labels()
        # Color burn-in region in grey if asked
        if (self.burn is not None) and (not cleaned_chains):
            for ax in fig.get_axes():
                ax.fill_betweenx(
                    (0, 1),
                    0,
                    self.burn,
                    color="k",
                    alpha=0.3,
                    transform=ax.get_xaxis_transform(),
                    zorder=1000,
                )
        if cleaned_chains:
            fig.savefig(
                self.plot_path + "clean_chains.png", bbox_inches="tight", dpi=200
            )
        else:
            fig.savefig(self.plot_path + "chains.png", bbox_inches="tight", dpi=200)
        plt.close(fig)

        if not cleaned_chains:
            chains.close()

    # ---------------------------------------------------------------------- #

    def plot_distributions(
        self,
        color=None,
        param_names=None,
        alsoplot=[],
    ):
        """
        Plot the posterior distribution as a corner plot, as well as
        a marginalized distributions-only plot.
        The plots are created using ChainConsumer, doc :
        https://samreay.github.io/ChainConsumer/examples/index.html

        Args:
            color (str or None): whether to use the color of the points
                as an additional dimension.
                Can be "likelihood" or "posterior".

            param_names (list or None): the list of parameter names,
                if None, will try to figure it out using
                ``self.model.param_names``

            alsoplot (list): add some probability values as an additional
                dimension. Can contain "chi2", "lnlike", "lnprob", "lnprior".
        """

        # ===== Format the chains ===== #
        chains_toplot = copy(self.chains["chains"])
        if param_names is None:
            param_names = copy(self.model.param_names)
        n = 0
        for p in alsoplot:
            if p == "chi2":
                p = "$\chi^2$"
                new_chain = -2.0 * self.chains["lnlike"] / self.dof
            else:
                new_chain = self.chains[p]
            chains_toplot = np.concatenate(
                [chains_toplot, new_chain[:, :, np.newaxis]], axis=2
            )
            param_names.append(p)
            n += 1

        chains_flat = chains_toplot.reshape(-1, self.ndim + n)
        # chains_flat = self.subsample_uncorr

        cc = ChainConsumer()
        cc.configure_truth(linestyle="--", color="tab:red", alpha=0.5)

        if color == "likelihood":
            like_chain = self.chains["lnlike"].flatten()
            cc.add_chain(
                np.concatenate((chains_flat, like_chain[:, np.newaxis]), axis=1),
                parameters=param_names + ["Likelihood"],
                zorder=10,
                name="Posterior distribution",
                shade=True,
                shade_alpha=0.0,
                shade_gradient=0.0,
            )
            cc.configure(color_params="Likelihood", num_cloud=1e4)
        elif color == "posterior":
            post_chain = self.chains["lnprob"].flatten()
            cc.add_chain(
                np.concatenate((chains_flat, post_chain[:, np.newaxis]), axis=1),
                parameters=param_names + ["Posterior"],
                zorder=10,
                name="Posterior distribution",
                shade=True,
                shade_alpha=0.0,
                shade_gradient=0.0,
            )
            cc.configure(color_params="Posterior", num_cloud=1e4)
        else:
            cc.add_chain(
                chains_flat,
                parameters=param_names,
                zorder=10,
                name="Posterior distribution",
                color="#2178AE",  # matplotlib's tab:blue
                shade=True,
                shade_alpha=0.3,
                shade_gradient=0.0,
            )

        extents = None
        fig = cc.plotter.plot(truth=self.truth_as_list, extents=extents)
        fig.align_labels()
        fig.savefig(self.plot_path + "corner.pdf", bbox_inches="tight")
        plt.close(fig)

        fig = cc.plotter.plot_distributions(truth=self.truth_as_list, extents=extents)
        fig.align_labels()
        fig.subplots_adjust(hspace=1.0)
        fig.savefig(
            self.plot_path + "marginalized_distributions.pdf", bbox_inches="tight"
        )
        plt.close(fig)

        # ===== Bins correlation matrix if non-parametric ===== #
        if self.model_type == "nonparam":
            fig, ax = plt.subplots()
            corrmat = np.corrcoef(
                self.chains["chains"][:, :, self.model.indices_press]
                .reshape(-1, self.model.nbins)
                .T
            )
            im = ax.matshow(
                corrmat,
                vmin=-1.0,
                vmax=1.0,
                cmap="RdBu",
                interpolation=None,
            )
            cb = fig.colorbar(im)
            cb.set_label("Correlation coefficient")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.savefig(self.plot_path + "pressure_correlations.pdf")
            plt.close(fig)

    # ---------------------------------------------------------------------- #

    def chains2physics(self, radii, method="interp_powerlaw", nthreads=8):
        """
        Computes the physical properties from the samples of the
        posterior distribution and stores them in a dictionary
        (``self.thermo_profiles``).

        Args:
            radii (array): the radii at which to compute the
                thermodynamical properties.

            nthreads (int): the number of threads that can be used
                for the thermodynamical properties of the ICM,

            method (str): how to proceed for non-parametric profiles.
                Can be "interp_powerlaw", "interp_spline", or "gNFW"
        """

        thermo_profiles = {"r": radii}

        nsteps = self.chains_flat.shape[0]
        chains_flat = np.vstack((self.chains_flat, self.solid_params))
        press_chains = chains_flat[:, self.model.indices_press]
        press_std = np.std(press_chains, axis=0)

        if self.model_type == "gNFW":
            self.label_solid_profile = "gNFW fit"
        elif self.model_type == "nonparam" and method == "interp_powerlaw":
            self.label_solid_profile = "power law interp."
        elif self.model_type == "nonparam" and method == "interp_spline":
            self.label_solid_profile = "spline interp."
        elif self.model_type == "nonparam" and method == "gNFW":
            self.label_solid_profile = "gNFW fit"

        # ===== Initialization ===== #
        # ===== Case 1. "Simple" (pressure profile = what you fitted) ===== #
        if (self.model_type == "gnfw") or (
            self.model_type == "nonparam" and method == "interp_powerlaw"
        ):

            def pressure_profile(i):
                par = self.model.params_to_dict(chains_flat[i])
                return self.model.pressure_profile(radii, par)

            def mass_profile(i):
                par = self.model.params_to_dict(chains_flat[i])
                dpdr = self.model.deriv_pressure_profile(radii, par)
                return -(radii ** 2) * dpdr / thermo_profiles["d"]

        # ===== Case 2. Spline interpolation of a non-parametric profile ===== #
        elif self.model_type == "nonparam" and method == "interp_spline":

            all_splines = Parallel(n_jobs=nthreads)(
                delayed(LogLogSpline)(self.model.radius_tab, press_chains[i], k=2)
                for i in range(nsteps + 1)
            )

            def pressure_profile(i):
                return all_splines[i](radii)

            def mass_profile(i):
                dpdr = all_splines[i].differentiate(radii)
                return -(radii ** 2) * dpdr / thermo_profiles["d"]

        # ===== Case 3. gNFW fit to "interpolate" a non-parametric profile ===== #
        elif self.model_type == "nonparam" and method == "gNFW":

            def fit_gNFW(i):
                press = press_chains[i, :]
                r = self.model.radius_tab
                param_names = ["P_0", "r_p", "a", "b", "c"]
                init_pos = self.model.cluster.A10_params
                init_step = [0.1 * p for p in init_pos]
                limit = [(1e-3 * p, 10 * p) for p in init_pos]

                def chi2(params):
                    return np.sum(
                        ((press - _model_gnfw.gNFW(r, *params)) / press_std) ** 2
                    )

                m = Minuit.from_array_func(
                    chi2,
                    init_pos,
                    limit=limit,
                    error=init_step,
                    errordef=1,
                    name=param_names,
                )
                migrad = m.migrad()
                return np.array([p.value for p in migrad.params])

            all_gnfw_params = Parallel(n_jobs=nthreads)(
                delayed(fit_gNFW)(i) for i in range(nsteps + 1)
            )

            def pressure_profile(i):
                return _model_gnfw.gNFW(radii, *all_gnfw_params[i])

            def mass_profile(i):
                dpdr = _model_gnfw.d_gNFW_d_r(radii, *all_gnfw_params[i])
                return -(radii ** 2) * dpdr / thermo_profiles["d"]

        # ===== Case 4. None of the above ===== #
        else:
            raise Exception(
                f"`{method}` not in the authorized options. "
                + "Please give one of `interp_powerlaw`, `interp_spline`, or `gNFW`."
            )

        # ===== Thermodynamical profiles computation ===== #
        # ===== Pressure ------------------------------------------------------- #
        thermo_profiles["allp"] = np.array([pressure_profile(i) for i in range(nsteps)])

        if self.solid == "bf":  # symetric error bars = std
            thermo_profiles["p"] = pressure_profile(-1)
            errp = np.std(thermo_profiles["allp"], axis=0, ddof=1)
            thermo_profiles["errp"] = np.array(
                [thermo_profiles["p"] - i * errp for i in [-2, -1, 1, 2]]
            )
        else:  # asymmetric error bars = percentiles
            thermo_profiles["p"] = np.median(thermo_profiles["allp"], axis=0)
            thermo_profiles["errp"] = np.percentile(
                thermo_profiles["allp"], [2.5, 16, 84, 97.5], axis=0
            )
        thermo_profiles = thermo_profiles
        self.profiles_toplot = {
            "p": {
                "filename": "pressure.pdf",
                "ylabel": r"Pressure $P_e(r) \; [{\rm keV \cdot cm^{-3}}]$",
                "legend_loc": 1,
            }
        }

        # ===== Stop here if no X data ===== #
        if self.x_profiles is None:
            return thermo_profiles

        radii = thermo_profiles["r"]

        # ===== Density at same radii points as pressure ----------------------- #
        thermo_profiles["d"] = interp_powerlaw(
            self.x_profiles["rd"], self.x_profiles["d"], radii
        )

        # ===== Temperature ---------------------------------------------------- #
        thermo_profiles["allt"] = thermo_profiles["allp"] / thermo_profiles["d"]
        if self.solid == "bf":
            thermo_profiles["t"] = thermo_profiles["p"] / thermo_profiles["d"]
            errt = np.std(thermo_profiles["allt"], axis=0, ddof=1)
            thermo_profiles["errt"] = [
                thermo_profiles["t"] - i * errt for i in [-2, -1, 1, 2]
            ]
        else:
            thermo_profiles["t"] = np.median(thermo_profiles["allt"], axis=0)
            thermo_profiles["errt"] = np.percentile(
                thermo_profiles["allt"],
                [2.5, 16, 84, 97.5],
                axis=0,
            )

        # ===== Entropy -------------------------------------------------------- #
        thermo_profiles["allk"] = thermo_profiles["allp"] * thermo_profiles["d"] ** (
            -5 / 3
        )
        if self.solid == "bf":
            thermo_profiles["k"] = thermo_profiles["p"] * thermo_profiles["d"] ** (
                -5 / 3
            )
            errk = np.std(thermo_profiles["allk"], axis=0, ddof=1)
            thermo_profiles["errk"] = [
                thermo_profiles["k"] - i * errk for i in [-2, -1, 1, 2]
            ]
        else:
            thermo_profiles["k"] = np.median(thermo_profiles["allk"], axis=0)
            thermo_profiles["errk"] = np.percentile(
                thermo_profiles["allk"],
                [2.5, 16, 84, 97.5],
                axis=0,
            )

        # ===== HSE Mass ------------------------------------------------------- #
        thermo_profiles["allm"] = (
            (
                np.array([mass_profile(i) for i in range(nsteps)])
                * u.Unit("kpc keV")
                / dens_fact
            )
            .to("Msun")
            .value
        )

        if self.solid == "bf":
            thermo_profiles["m"] = (
                (mass_profile(-1) * u.Unit("kpc keV") / dens_fact).to("Msun").value
            )
            errm = np.std(thermo_profiles["allm"], axis=0, ddof=1)
            thermo_profiles["errm"] = [
                thermo_profiles["m"] - i * errm for i in [-2, -1, 1, 2]
            ]
        else:
            thermo_profiles["m"] = np.median(thermo_profiles["allm"], axis=0)
            thermo_profiles["errm"] = np.percentile(
                thermo_profiles["allm"],
                [2.5, 16, 84, 97.5],
                axis=0,
            )

        self.profiles_toplot["t"] = {
            "filename": "temperature.pdf",
            "ylabel": r"Temperature $kT_e(r) \; [{\rm keV}]$",
            "legend_loc": 1,
        }
        self.profiles_toplot["k"] = {
            "filename": "entropy.pdf",
            "ylabel": r"Entropy $K_e(r) \; [{\rm keV \cdot cm^{2}}]$",
            "legend_loc": 2,
        }
        self.profiles_toplot["m"] = {
            "filename": "mass.pdf",
            "ylabel": r"HSE Mass $M(<r) \; [{\rm M_\odot}]$",
            "legend_loc": 2,
        }
        return thermo_profiles

    # ---------------------------------------------------------------------- #

    def compute_integrated_values(self, fix_R500=False):
        """
        Computes R_500, M_500 and Y_500 for all samples of the Markov chains
        to get a probability distribution for each measurement.

        Args:
            fix_R500 (bool): whether or not you want all your computations
                of M_500 and Y_500 to be performed at the same value of
                R_500.
                If False, A value of R_500 will be computed for each sample
                in the MCMC, and M_500 and Y_500 will be computed inside it,
                which will result in much larger error bars
                (see "Error bars on integrated quantities" in the doc).
        """

        cluster = self.model.cluster
        r = self.thermo_profiles["r"]
        n = self.thermo_profiles["allm"].shape[0]

        # ===== R_500 ===== #
        dens_prof = ((4.0 / 3.0) * np.pi * (r * u.kpc) ** 3 * cluster.dens_crit).to(
            "Msun"
        )
        # 1) All distribution
        contrast_prof = self.thermo_profiles["allm"] / dens_prof.value

        w500 = [  # Find where the contrast prof goes through 500
            np.logical_or(
                np.logical_and(cp_i >= 500.0, np.roll(cp_i, 1) <= 500.0),
                np.logical_and(cp_i <= 500.0, np.roll(cp_i, 1) >= 500.0),
            )
            for cp_i in contrast_prof
        ]

        w500 = [np.max(np.where(w_i)) for w_i in w500]  # Take the last (maximum radius)
        allR500 = [  # Power-law interp between the two nearest points
            interp_powerlaw(cp_i[w - 1 : w + 1], r[w - 1 : w + 1], 500.0)
            for cp_i, w in zip(contrast_prof, w500)
        ]
        allR500 = np.array(allR500)

        # 2) Central value = maximum of pdf
        kde = gaussian_kde(np.delete(allR500, np.where(np.isnan(allR500))))
        R500 = minimize(lambda x: -kde.logpdf(x), 1e3).x[0]

        # ===== M_500 ===== #
        # 1) All distribution
        if fix_R500:
            allM500 = [
                interp_powerlaw(r, self.thermo_profiles["allm"][i, :], R500)
                for i in range(n)
            ]
        else:
            allM500 = [
                interp_powerlaw(r, self.thermo_profiles["allm"][i, :], allR500[i])
                for i in range(n)
            ]
            allM500 = np.array(allM500)

        # 2) Central value = maximum of pdf
        kde = gaussian_kde(np.delete(allM500, np.where(np.isnan(allM500))))
        M500 = (
            minimize(lambda x: -kde.logpdf(x * 1e14), cluster.M_500.value / 1e14).x[0]
            * 1e14
        )

        # ===== Y_500 ===== #
        # 1) All distribution
        if fix_R500:
            allY500 = [
                self.model.compute_integrated_SZ(
                    self.model.params_to_dict(self.chains_flat[i, :]), R500
                )
                for i in range(n)
            ]
        else:
            allY500 = [
                self.model.compute_integrated_SZ(
                    self.model.params_to_dict(self.chains_flat[i, :]), allR500[i]
                )
                for i in range(n)
            ]
        allY500 = np.array(allY500)

        # 2) Central value = maximum of pdf
        kde = gaussian_kde(np.delete(allY500, np.where(np.isnan(allY500))))
        Y500 = minimize(lambda x: -kde.logpdf(x), cluster.Y_500_kpc2).x[0]

        # Clean up
        mask = np.where(np.logical_or(np.isnan(allR500), np.isnan(allM500)))
        allR500 = np.delete(allR500, mask)
        allM500 = np.delete(allM500, mask)

        self.all_int = {
            "R_500": R500,
            "R_500_dist": allR500,
            "M_500": M500,
            "M_500_dist": allM500,
            "Y_500": Y500,
            "Y_500_dist": allY500,
        }
        np.savez(self.path + "/integrated_values_samples.npz", **self.all_int)

    # ---------------------------------------------------------------------- #

    def plot_profiles(self, x=True, r_limits=(0.1, 1.0), vlines={}):
        """
        Plot the thermodynamical profiles.

        Args:
            x (bool): whether you also want the X-ray only profiles
                on the plots.

            r_limits (tuple): radial range on which to show the
                profiles in kpc.

            vlines (dict): the vertical lines to add on the plot
                for reference. Each element should be
                ("String to display" : value in kpc)
        """

        profs_cont = self.thermo_profiles
        profs_np = self.thermo_profiles_np

        for key in self.profiles_toplot.keys():
            legend_handles = []
            legend_labels = []
            plusxmm = "+ XMM" if key != "p" else ""
            fig, ax = plt.subplots()

            # ===== Non-param points ===== #
            if self.model_type == "nonparam":
                erb_np = ax.errorbar(
                    profs_np["r"],
                    profs_np[key],
                    xerr=None,
                    yerr=[
                        profs_np[key] - profs_np["err" + key][1],
                        profs_np["err" + key][2] - profs_np[key],
                    ],
                    color="tab:blue",
                    mec="w",
                    fmt="o",
                    zorder=90,
                    **errb_kwargs,
                )
            legend_handles.append(erb_np)
            legend_labels.append(f"NIKA2 non-parametric {plusxmm}")

            # ===== Continuous profile + envelopes ===== #
            line = ax.plot(
                profs_cont["r"],
                profs_cont[key],
                zorder=99,
                ls="--",
                color="tab:blue",
                alpha=0.5,
            )
            # ax.fill_between(
            #    profs_cont["r"],
            #    profs_cont["err" + key][0],
            #    profs_cont["err" + key][3],
            #    zorder=98,
            #    color="tab:blue",
            #    alpha=0.15,
            # )
            ax.fill_between(
                profs_cont["r"],
                profs_cont["err" + key][1],
                profs_cont["err" + key][2],
                zorder=98,
                color="tab:blue",
                alpha=0.25,
            )
            fill = ax.fill(np.nan, np.nan, color="tab:blue", alpha=0.25, lw=0)
            legend_handles.append((line[0], fill[0]))
            legend_labels.append(f"NIKA2 {self.label_solid_profile} {plusxmm}")

            # ===== X profiles ===== #
            if x and self.x_profiles is not None:
                erb_x = ax.errorbar(
                    self.x_profiles["r" + key],
                    self.x_profiles[key],
                    xerr=None,
                    yerr=self.x_profiles["err" + key],
                    zorder=89,
                    color="tab:red",
                    mec="w",
                    fmt="o",
                    **errb_kwargs,
                )
                legend_handles.append(erb_x)
                legend_labels.append(f"XMM only")

            ax.set_xlim(*r_limits)

            # ===== Truth on pressure if relevant ===== #
            if (key == "p") and (self.truth is not None):
                line_truth = ax.plot(
                    np.logspace(*np.log10(r_limits), 100),
                    _model_gnfw.gNFW(
                        np.logspace(*np.log10(r_limits), 100),
                        *[self.truth[k] for k in ["P0", "rp", "a", "b", "c"]],
                    ),
                    zorder=80,
                    color="k",
                    ls=":",
                )
                legend_handles.append(line_truth[0])
                legend_labels.append(f"Input profile")

            # ===== Scales ===== #
            ax.set_xscale("log")
            ax.set_yscale("linear" if key == "t" else "log")

            # ===== Vertical lines to help vizualization ===== #
            for vline in vlines.keys():
                ax.axvline(vlines[vline], 0, 1, ls="--", color="k", zorder=0)
                ax.text(
                    vlines[vline],
                    0.02,
                    vline,
                    rotation=90,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    bbox={"facecolor": "w", "edgecolor": "w"},
                    transform=ax.get_xaxis_transform(),
                    zorder=1,
                    fontsize=8.0,
                )

            # ===== Decoration ===== #
            ax.set_xlabel(r"Radius $r \; [{\rm kpc}]$")
            ax.set_ylabel(self.profiles_toplot[key]["ylabel"])
            ax_legend(
                ax,
                legend_handles,
                legend_labels,
                loc=self.profiles_toplot[key]["legend_loc"],
            )
            ax_bothticks(ax)
            fig.savefig(self.plot_path + self.profiles_toplot[key]["filename"])
            plt.close(fig)

    # ---------------------------------------------------------------------- #

    def plot_integrated_values(self):
        cc = ChainConsumer()
        cc.add_chain(
            {
                r"$R_{500} \; [{\rm kpc}]$": self.all_int["R_500_dist"],
                r"$\mathcal{D}_{\rm A}^2 Y_{500} \; [{\rm kpc}^2]$": self.all_int[
                    "Y_500_dist"
                ],
                r"$M_{500} \; [{\rm 10^{14} \, M_\odot}]$": self.all_int["M_500_dist"]
                / 1e14,
            },
            color="#2178AE",  # matplotlib's tab:blue
            shade=True,
            shade_alpha=0.3,
            shade_gradient=0.0,
        )
        cc.configure(summary=False)
        cfig = cc.plotter.plot()
        cfig.savefig(self.plot_path + "integrated_values.pdf", bbox_inches="tight")
        plt.close(cfig)

    # ---------------------------------------------------------------------- #

    def plot_dmr(self, data, smooth_pix, rms=None):
        """
        Plot data, model, residuals.

        Args:
            data (array): the data map.

            smooth_pix (float): the sigma of the gaussian kernel
                you want to apply to smooth your maps for the plot
                (in pixel units).
        """

        data_smooth = gaussian_filter(data, smooth_pix)
        vmin = 1e3 * np.min((data_smooth.min(), -data_smooth.max()))

        if rms is not None:
            rms_smooth = 1e3 * gaussian_filter(rms, smooth_pix) / np.sqrt(
                2 * np.pi * smooth_pix ** 2
            )

        model_map, _ = self.model(self.solid_par)

        fig, axs = plt.subplots(1, 3)
        for m, ax in zip([data, model_map, data - model_map], axs.flatten()):
            m_smooth = gaussian_filter(1e3 * m, smooth_pix)
            im = ax.imshow(
                m_smooth,
                vmin=vmin,
                vmax=-vmin,
                origin="lower",
                cmap=cmap_plots,
                interpolation="gaussian",
            )
            if rms is not None:
                ax.contour(
                    -m_smooth / rms_smooth,
                    colors="k",
                    linewidths=0.75,
                    alpha=0.25,
                    levels=np.concatenate((np.arange(-10, -2), np.arange(3, 20))),
                )
            ax.set_xticks([])
            ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.0)
        cb.set_label("Surface brightness [mJy/beam]")
        fig.savefig(self.plot_path + "data_model_residuals.pdf", bbox_inches="tight")
        plt.close(fig)


# =============================================================================== #


def ax_legend(ax, *args, **kwargs):
    """
    Add a simple, good-looking legend to your plots.
    """
    leg = ax.legend(
        *args, facecolor="w", frameon=True, edgecolor="k", framealpha=1, **kwargs
    )
    leg.get_frame().set_linewidth(0.5)


def ax_bothticks(ax):
    """
    Add ticks on the top and right axis
    """
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
