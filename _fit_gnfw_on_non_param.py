import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, multivariate_normal
from scipy.special import erfc
import os
import _model_nonparam, _model_gnfw
import _results, _utils


class GNFWFitter:
    """
    Class to fit a gNFW model from non-parametric results.

    Args:
        chains (array): the chains from your non-parametric MCMC,
            shape = (nwalkers, nsteps, nparams)

        radius_tab (array): the radial bins of your non-parametric
            profile

        path_to_results (string): where to save your results,

        cluster (_cluster.Cluster): a ``Cluster`` object,

        kind (string): "gaussian" or "kde", how you want the probability
            distribution of the pressure points to be computed.

        which_pts (slice): which of the nonparam points to be fitted.
            May be used to e.g. remove the last two points,
            with slice(0, -2)

        center_bins (None or (array)): override the center of the
            marginalized distributions for the pressure bins. May be
            used to e.g. center on best-fit.
    """

    def __init__(
        self,
        chains,
        radius_tab,
        path_to_results,
        cluster,
        kind="gaussian",
        which_pts=slice(None, None),
        center_bins=None,
    ):
        self.path_to_results = path_to_results
        if not os.path.isdir(self.path_to_results):
            os.mkdir(self.path_to_results)
        self.cluster = cluster
        self.A10_params = self.cluster.A10_params

        self.radius_tab = radius_tab[which_pts]
        self.n_pts = len(self.radius_tab)
        chains = chains[:, :, : self.n_pts].reshape(-1, self.n_pts)
        self.pressure_tab = np.median(chains, axis=0)

        print(f"    Initialized gNFW fit on {self.n_pts} points:")
        for i in range(self.n_pts):
            print(f"       P{i} = {self.radius_tab[i]:.2f} kpc")

        if kind.lower() == "kde":
            kde = gaussian_kde(chains.T)
            self.log_prob_press = kde.logpdf  # this is a function

        else:
            if center_bins is None:
                mean = np.mean(chains, axis=0)
            else:
                mean = center_bins[which_pts]
            norm = multivariate_normal(mean=mean, cov=np.cov(chains.T))
            self.log_prob_press = norm.logpdf  # this is a function

    # ---------------------------------------------------------------------- #

    def log_like(self, params):
        press_gnfw = _model_gnfw.gNFW(self.radius_tab, *params)
        ll = self.log_prob_press(press_gnfw)
        return ll if np.isfinite(ll) else -np.inf

    # ---------------------------------------------------------------------- #

    def log_prior(self, params):
        lp = [np.log(my_erfc(params[i], 2 * self.A10_params[i])) for i in range(5)]
        return np.sum(lp)

    # ---------------------------------------------------------------------- #

    def log_post(self, params):
        lp = self.log_prior(params)
        if np.isfinite(lp):
            return lp + self.log_like(params)
        else:
            return -np.inf

    # ---------------------------------------------------------------------- #

    def do_fit_minuit(self):
        """
        Perform the fit using ``Minuit``, i.e. find the maximum-likelihood
        for a gNFW profile on your non-parametric posterior distribution
        """
        from iminuit import Minuit

        param_names = ["P_0", "r_p", "a", "b", "c"]
        init_pos = self.A10_params
        init_step = [0.01 * p for p in init_pos]
        limit = [(0.0, None) for p in init_pos]

        def chi2(params):
            return -2.0 * self.log_like(params)

        m = Minuit.from_array_func(
            chi2, init_pos, limit=limit, error=init_step, errordef=1, name=param_names
        )
        migrad = m.migrad()

        self.best_fit_params = np.array([p.value for p in migrad.params])
        print("    Best-fit params:")
        for name, p in zip(param_names, self.best_fit_params):
            print(f"    {name} = {p}")

    # ---------------------------------------------------------------------- #

    def do_fit_mcmc(self, nchains, nsteps, ncheck, nthreads, burn, restore=False):
        """
        Perform the fit using MCMC, i.e. sample a posterior distribution
        for a gNFW profile on your non-parametric posterior distribution

        Args:
            nchains (int): the number of walkers.

            nsteps (int): the initial number of steps to perform
                (sampling can stop before that if convergence has
                been reached).

            ncheck (int): MCMC convergence will be checked every
                ``ncheck`` iterations.

            nthreads (int): number of threads to use.

            burn (int): the burn-in period.

            restore (bool): wether you want to read existing chains
                rather than sampling.
        """
        import emcee
        from multiprocessing import Pool
        import time
        import _results

        self.out_chains_file = self.path_to_results + "chains.npz"

        try:
            pos = [self.best_fit_params[k] for k in ["P0", "rp", "a", "b", "c"]]
        except:
            pos = self.A10_params
        init_pos = [np.random.normal(pos, 1e-2 * pos) for _ in range(nchains)]

        if not restore:
            with Pool(processes=nthreads) as pool:
                ti = time.time()
                sampler = emcee.EnsembleSampler(
                    nchains, 5, self.log_post, pool=pool, moves=emcee.moves.DEMove()
                )

                # sampler.run_mcmc(init_pos, nsteps, progress=True)
                for sample in sampler.sample(
                    init_pos, iterations=nsteps, progress=True
                ):

                    it = sampler.iteration
                    if it % ncheck != 0.0:
                        continue
                    blobs = sampler.get_blobs()
                    chains = {
                        "chains": sampler.chain,
                        "lnprob": sampler.lnprobability,
                    }
                    np.savez(self.out_chains_file, **chains)

                    # ===== Test convergence ===== #
                    if it < burn:
                        continue
                    results = _results.Results(
                        self.out_chains_file,
                        burn,
                        None,
                        self.path_to_results,
                    )
                    nchains_new = results.clean_chains(
                        clip_at_autocorr=30, do_thinning=False
                    )
                    R_hat = results.gelman_rubin_stat()
                    print("R_hat =", R_hat)

                    # ===== Accept convergence if: ===== #
                    #   - Gelman-Rubin is ok
                    #   - Less than 1/3 of the chains are cut in clean_chains
                    #   Because if you only do G-R, you don't account for how crappy
                    #   some of the chains might be
                    converged = np.all(R_hat < 1.03) and (nchains_new > 2 / 3 * nchains)

                    if converged:
                        break
        else:
            print("    Restoring chains")

    # ---------------------------------------------------------------------- #

    def manage_chains(self, burn):  # those are rare!
        """
        Performs chain cleaning and MCMC diagnostic plots

        Args:
            burn (int): the burn-in period.
        """

        model = _model_gnfw.ModelGNFW(self.cluster)
        model.init_param_indices(nocalib=True)
        results = _results.Results(
            self.out_chains_file,
            burn,
            model,
            self.path_to_results,
            solid="bf",
        )

        results.plot_mcmc_diagnostics(cleaned_chains=False)
        results.clean_chains(clip_at_autocorr=30, do_thinning=True)
        results.solid_statistic()
        results.plot_mcmc_diagnostics(cleaned_chains=True)
        results.plot_distributions()

        self.results = results

    # ---------------------------------------------------------------------- #

    def compute_thermo_profiles(self, radii, x_profiles=None, nthreads=100):
        """
        Computes the thermodynamical profiles from the
        sampled posterior distribution

        Args:
            radii (array): the radii at which to compute the 
                thermodynamical properties.

            x_profiles (dict): a parprod-generated dict (see ``_xray``)

            nthreads (int): the number of threads that can be used
                for the thermodynamical properties of the ICM.
        """
        self.results.x_profiles = x_profiles
        self.results.chains2physics(radii, nthreads=nthreads)


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #


def plot_comparison(
    prof_np, prof_gnfw, path_to_results, truth=None, center_bins=None, vlines={}
):
    """
    prof_np, prof_gnfw are thermo_profiles dicts
    """

    if truth is not None:
        P0 = truth["P0"]
        rp = truth["rp"]
        a = truth["a"]
        b = truth["b"]
        c = truth["c"]

    fig, axs = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0)

    """
    Pressure profiles
    """
    ax = axs[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(prof_gnfw["r"].min(), prof_gnfw["r"].max())
    ax.set_ylabel(r"Pressure $P_e(r) \; {\rm [keV \cdot cm^{-3}]}$")

    # ===== Non-param ===== #
    ax.errorbar(
        prof_np["r"],
        prof_np["p"],
        xerr=None,
        yerr=[prof_np["p"] - prof_np["errp"][1], prof_np["errp"][2] - prof_np["p"]],
        zorder=10,
        label="Non-param fit",
        mfc="tab:blue",
        **_results.errb_kwargs,
    )

    # ===== gNFW ===== #
    ax.plot(
        prof_gnfw["r"], prof_gnfw["p"], color="tab:blue", zorder=9, label="gNFW fit"
    )
    ax.fill_between(
        prof_gnfw["r"],
        prof_gnfw["errp"][1],
        prof_gnfw["errp"][2],
        color="tab:blue",
        alpha=0.3,
        zorder=8,
    )

    # ===== Truth ===== #
    if truth is not None:
        ax.plot(
            prof_gnfw["r"],
            _model_gnfw.gNFW(prof_gnfw["r"], P0, rp, a, b, c),
            color="tab:red",
            zorder=7,
            label="Truth",
        )

    # ===== vlines ===== #
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

    """
    Difference with truth
    """
    ax = axs[1]
    ax.set_xscale("log")
    ax.set_xlim(prof_gnfw["r"].min(), prof_gnfw["r"].max())
    ax.set_xlabel(r"Radius $r \; {\rm [kpc]}$")
    ax.set_ylim(-1, 1)

    if truth is not None:
        ref_prof = lambda r: _model_gnfw.gNFW(r, P0, rp, a, b, c)
        ax.set_ylabel(r"$\left(P - P_\mathrm{truth}\right) / P_\mathrm{truth}$")
    else:
        ref_prof = lambda r: _utils.interp_powerlaw(prof_gnfw["r"], prof_gnfw["p"], r)
        ax.set_ylabel(r"$\left(P - P_\mathrm{gNFW}\right) / P_\mathrm{gNFW}$")

    # ===== Non-param ===== #
    ax.errorbar(
        prof_np["r"],
        (prof_np["p"] - ref_prof(prof_np["r"])) / ref_prof(prof_np["r"]),
        xerr=None,
        yerr=np.array(
            [prof_np["p"] - prof_np["errp"][1], prof_np["errp"][2] - prof_np["p"]]
        )
        / ref_prof(prof_np["r"]),
        zorder=10,
        label="Non-param fit",
        mfc="tab:blue",
        **_results.errb_kwargs,
    )

    # ===== gNFW ===== #
    ax.plot(
        prof_gnfw["r"],
        (prof_gnfw["p"] - ref_prof(prof_gnfw["r"])) / ref_prof(prof_gnfw["r"]),
        color="tab:blue",
        zorder=9,
        label="gNFW fit",
    )
    ax.fill_between(
        prof_gnfw["r"],
        (prof_gnfw["errp"][1] - ref_prof(prof_gnfw["r"])) / ref_prof(prof_gnfw["r"]),
        (prof_gnfw["errp"][2] - ref_prof(prof_gnfw["r"])) / ref_prof(prof_gnfw["r"]),
        color="tab:blue",
        alpha=0.3,
        zorder=8,
    )

    # ===== Truth ===== #
    if truth is not None:
        ax.plot(
            prof_gnfw["r"],
            0.0 * prof_gnfw["r"],
            color="tab:red",
            zorder=7,
            label="Truth",
        )

    # ===== vlines ===== #
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

    for ax in axs:
        _results.ax_bothticks(ax)
    _results.ax_legend(axs[0])

    fig.savefig(path_to_results + "Plots/pressure_gNFW_vs_NonParam.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------- #


def my_erfc(x, transition):
    y = erfc(x - transition)
    return y if x > 0.0 else 0.0
