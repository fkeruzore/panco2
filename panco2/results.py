import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.ndimage import gaussian_filter
from chainconsumer import ChainConsumer
from copy import copy


def load_chains(out_chains_file, burn, discard, clip_percent=0, verbose=False):
    """
    Loads raw Markov chains, cleans them, and arranges them
    in a convenient data frame.

    Parameters
    ----------
    out_chains_file : str
        Path to the `npz` file containing the raw chains produced by
        `panco2.PressureProfileFitter.run_mcmc`
    burn : int
        Length to discard as burn-in.
    discard : int
        Thinning length. See Notes.
    clip_percent : float
        Percentile considered as extreme to weed-out bad chains.
        See Notes.
    verbose : bool
        Wether or not you want results of the cleaning to be printed.

    Returns
    -------
    chains_clean: pd.DataFrame
        Cleaned chains in a DataFrame.
        Keys are the parameter names, as well as `lnprior`
        (log-prior value), `lnlike` (log-likelihood), `chain`
        (which walker the sample belongs to) and `step` (which
        step of the walk the sample was obtained at).

    Notes
    =====
    Cleaning is done as follows:

    1) Discard `burn` burn-in steps and keep one sample every
       `discard` steps

    2) Find chains that are always out of the [clip_percent,
       100 - clip_percent] interval for every parameter and
       discard them.
    """

    f = np.load(out_chains_file)
    chains = {}
    p = f.files[0]
    n_chains, n_steps = f[p].shape

    # 1) discard `burn` burn-in and keep 1 sample every `discard` steps
    keep = np.arange(burn, n_steps, discard)
    n_steps2 = keep.size
    for p in f.files:
        chains[p] = f[p][:, keep]
    if verbose:
        print(
            f"Raw length: {n_steps},",
            f"clip {burn} as burn-in, discard {discard-1}/{discard} samples",
            f" -> Final chains length: {n_steps2}",
        )

    # 2) discard chains always in the `clip_percent`% most extreme
    lims = {
        p: np.percentile(c, (clip_percent, 100 - clip_percent))
        for p, c in chains.items()
    }
    is_ok = np.zeros(n_chains, dtype=bool)
    for i in range(n_chains):
        is_ok_perp = [
            (np.mean(ch[i]) >= lims[p][0]) & (np.mean(ch[i]) <= lims[p][1])
            for p, ch in chains.items()
        ]
        is_ok[i] = np.all(np.array(is_ok_perp, dtype=bool))
    n_chains2 = is_ok.sum()
    chains = {p: ch[is_ok, :] for p, ch in chains.items()}
    if verbose:
        print(
            f"{n_chains} walkers,",
            f"remove chains with the {clip_percent}% most extreme values",
            f" -> {n_chains2} chains remaining",
        )
        print(
            f"-> Final sampling size:",
            f"{n_chains2} chains * {n_steps2} samples per chain",
            f"= {n_chains2 * n_steps2} total samples",
        )

    # 3) store all in a dataframe
    i_chain, i_step = np.unravel_index(
        np.arange(chains[p].size), chains[p].shape
    )
    chains_clean = {"chain": i_chain, "step": i_step * discard}
    for p, chain in chains.items():
        chains_clean[p] = chain.flatten()
    del chains
    chains_clean = pd.DataFrame(chains_clean)
    return pd.DataFrame(chains_clean)


def mcmc_trace_plot(chains_clean, show_probs=True, filename=None):
    """
    Create MCMC trace plots, i.e. chains evolution with steps.

    Parameters
    ----------
    chains_clean: pd.DataFrame
        Markov chains, output of `load_chains`
    show_probs : bool
        Wether or not the log likelihood and log prior chains
        should be shown along with the parameters
    filename : str or None
        Filename to save the plot

    Returns
    -------
    fig
    """
    params = copy(chains_clean.columns)
    if show_probs:
        params = [p for p in params if (p not in ["chain", "step"])]
    else:
        params = [
            p
            for p in params
            if (p not in ["chain", "step", "lnprior", "lnlike"])
        ]

    cc = ChainConsumer()
    for i in range(chains_clean["chain"].max() + 1):
        cc.add_chain(chains_clean[chains_clean["chain"] == i][params])
    cc.configure(
        serif=False,
        usetex=False,
        cmap="Spectral_r",
    )
    fig = cc.plotter.plot_walks()

    fig.align_labels()
    if filename is not None:
        fig.savefig(filename)
    return fig


def mcmc_corner_plot(
    chains_clean,
    per_chain=False,
    show_probs=True,
    filename=None,
    ppf=None,
):
    """
    Plots 1D and 2D posterior sampled by the MCMC.

    Parameters
    ----------
    chains_clean: pd.DataFrame
        Markov chains, output of `load_chains`
    per_chain: bool
        If True, plots one distribution per chain.
        This can make very cluttered plots.
    show_probs : bool
        Wether or not the log likelihood and log prior chains
        should be shown along with the parameters
    filename : str or None
        Filename to save the plot
    ppf: `PressureProfileFitter` instance or None
        The main `panco2.PressureProfileFitter` instance.
        If provided, the marginal prior distribution will be
        shown for comparison with the posterior.

    Returns
    -------
    fig
    """

    params = copy(chains_clean.columns)
    if not show_probs:
        params = [p for p in params if (p not in ["lnprior", "lnlike"])]
    if per_chain:
        params = [p for p in params if (p not in ["step"])]
    else:
        params = [p for p in params if (p not in ["chain", "step"])]

    n = len(params)
    cc = ChainConsumer()
    if ppf is not None:
        prior_sample = {
            p: rv.rvs(int(1e5)) for p, rv in ppf.model.priors.items()
        }
        cc.add_chain(
            prior_sample,
            show_as_1d_prior=True,
            name="Priors",
            color="#7D7E7F",
            bar_shade=False,
        )
        lims = {
            p: list(chains_clean[p].quantile([0.01, 0.99])) for p in params
        }
    else:
        lims = None
    cc.add_chain(chains_clean[params], name="PANCO2")
    cc.configure(
        serif=False,
        usetex=False,
        summary=False,
        cmap="Spectral" + ("_r" if ppf is None else ""),
        shade_alpha=0.3,
        shade_gradient=0.0,
    )
    fig = cc.plotter.plot(extents=lims)

    axs = np.array(fig.get_axes()).reshape(n, n)
    for i in range(n):
        for j in range(i + 1):
            ax = axs[i, j]
            ax.set_xticks(axs[-1, j].get_xticks())
            if i != n - 1:
                ax.set_xticklabels([])
            if j != i:
                ax.set_yticks(axs[i, 0].get_yticks())
            if j != 0:
                ax.set_yticklabels([])
            ax_bothticks(ax)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.align_labels()
    if filename is not None:
        fig.savefig(filename)
    return fig


def mcmc_matrices_plot(chains_clean, ppf):
    """
    Produces an overly complicated plot of the correlation and
    covariance matrices of Markov chains, where:

    - The lower corner (diagonal not included) shows the
      correlation coefficient between parameters;

    - The upper corner (diagonal included) shows the
      absolute covariance between parameters.

    Parameters
    ----------
    chains_clean: pd.DataFrame
        Markov chains, output of `load_chains`
    ppf: `PressureProfileFitter` instance
        The main `panco2.PressureProfileFitter` instance.

    Returns
    -------
    fig, ax
    """
    corrs = chains_clean[ppf.model.params].corr()
    covs = chains_clean[ppf.model.params].cov()
    corrs_arr = np.array(corrs)
    covs_arr = np.abs(np.array(covs))
    i, j = np.meshgrid(
        np.arange(corrs_arr.shape[0]), np.arange(corrs_arr.shape[0])
    )
    corrs_arr[i >= j] = np.nan
    covs_arr[i < j] = np.nan

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(
        2,
        2,
        width_ratios=[20, 1],
        height_ratios=[20, 1],
        wspace=0.0,
        hspace=0.0,
        left=0.15,
        right=0.85,
        bottom=0.15,
        top=0.85,
    )
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    ax = axs[0]

    im1 = ax.matshow(
        corrs_arr, cmap="RdBu", vmin=-1.0, vmax=1.0, interpolation=None
    )
    cb1 = fig.colorbar(im1, cax=axs[2], orientation="horizontal")
    axs[2].set_xlabel(r"Correlation $\rho_{i, j}$")

    norm = mpl.colors.LogNorm(
        vmin=np.nanmin(covs_arr), vmax=np.nanmax(covs_arr)
    )
    im2 = ax.matshow(covs_arr, cmap="YlGn", norm=norm, interpolation=None)
    cb2 = fig.colorbar(im2, cax=axs[1])
    axs[1].set_ylabel(r"Covariance $\left| \Sigma^2_{i, j} \right|$")

    ax.set_xticks(np.arange(0, len(ppf.model.params)))
    ax.set_yticks(np.arange(0, len(ppf.model.params)))
    ax.set_xticklabels(ppf.model.params, rotation=45.0)
    ax.set_yticklabels(ppf.model.params, rotation=45.0)
    ax_bothticks(ax)
    return fig, axs[0]


def plot_profile(chains_clean, ppf, r_range, ax=None, label=None, **kwargs):
    """
    Plots the pressure profile recovered by PANCO2 from the
    Markov chains.

    Parameters
    ----------
    chains_clean: pd.DataFrame
        Markov chains, output of `load_chains`
    ppf: `PressureProfileFitter` instance
        The main `panco2.PressureProfileFitter` instance.
    r_range : np.array [kpc]
        The radial range on which to show the profile.
    ax : plt.Axis or None
        If provided, an existing axis can be used
    label : str or None
        Label of the curve for legend purposes
    **kwargs : dict
        Other options to pass to `plt.plot`

    Returns
    -------
    ax
    """

    model = ppf.model
    if ax is None:
        fig, ax = plt.subplots()

    chains_arr = np.array(
        [model.par_dic2vec(dict(p)) for p in chains_clean[model.params].iloc()]
    )

    all_profs = np.array(
        [model.pressure_profile(r_range, ch) for ch in chains_arr]
    )

    perc = np.percentile(all_profs, [16.0, 50.0, 84.0], axis=0)

    ax.fill_between(r_range, perc[0], perc[2], alpha=0.3, ls="--", zorder=3)
    ax.plot(r_range, perc[1], "-", label=label, zorder=4, **kwargs)
    ax.plot(
        model.r_bins,
        model.pressure_profile(
            model.r_bins, model.par_dic2vec(chains_clean.median())
        ),
        "o",
        color="tab:blue",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r \; [{\rm kpc}]$")
    ax.set_ylabel(r"$P_{\rm e}(r) \; [{\rm keV \cdot cm^{-3}}]$")

    lines_toplot = {
        "Pixel size": ppf.cluster.arcsec2kpc(ppf.pix_size),
        "Beam HWHM": ppf.cluster.arcsec2kpc(ppf.beam_fwhm / 2.0),
        "Half map size": ppf.cluster.arcsec2kpc(ppf.map_size * 60.0 / 2.0),
    }
    for label, line in lines_toplot.items():
        ax.axvline(line, 0, 1, color="k", alpha=0.5, ls=":", zorder=1)
    ax_bothticks(ax)
    return ax


def plot_data_model_residuals(
    ppf,
    par_vec=None,
    par_dic=None,
    smooth=0.0,
    fig=None,
    axs=None,
    lims=None,
    cbar_label=None,
    cbar_fact=1.0,
    cmap="RdBu_r",
    separate_ps_model=False,
):
    """
    Plot data, model, and residuals maps.

    Parameters
    ----------
    ppf: `PressureProfileFitter` instance
        The main `panco2.PressureProfileFitter` instance.
    par_vec : list or None
        Vector in the parameter space to use to compute
        the model map to be shown. Either one of `par_vec`
        or `par_dic` must be provided.
    par_dic : dict or None
        Same as `par_vec`, but in a dictionary. Either one
        of `par_vec` or `par_dic` must be provided.
    smooth : float [pixels]
        Width (sigma) of a gaussian kernel to be used to
        smooth the maps, for visual purposes.
    fig : plt.Figure or None
        If provided, an existing figure can be used
    axs : list of plt.Axis or None
        If provided, existing axes can be used.
        The length of the list must be consistent with
        the number of maps to show
    lims : tuple or None
        Limits of the color maps, in data units
    cbar_label : str or None
        Label for the colorbar
    cbar_fact : float
        A factor by which all maps are to be multiplied
        before plotting. Useful for very small units
        like Compton-y or Jy/beam
    cmap : str or mpl.colors.Colorbar
        The color map to use. Always make pretty plots!
    separate_ps_model : bool
        If your model fits both SZ and point sources, makes the
        model and residuals for SZ/PS/SZ+PS (i.e. 5 total plots)

    Returns
    -------
    fig, axs


    Raises
    ======
    Exception
        If neither `par_vec` or `par_dic` are provided.
    """

    if (par_dic is None) and (par_vec is None):
        raise Exception("Either `par_dic` or `par_vec` must be provided.")

    if (par_dic is not None) and (par_vec is None):
        par_vec = ppf.model.par_dic2vec(par_dic)
    if (par_vec is not None) and (par_dic is None):
        par_dic = ppf.model.par_vec2dic(par_vec)

    do_ps = ppf.model.n_ps > 0
    if do_ps and separate_ps_model:
        if fig is None:
            fig = plt.figure(figsize=(10, 8))
        if axs is None:
            axs = [
                fig.add_subplot(i, projection=ppf.wcs)
                for i in (334, 332, 333, 335, 336, 338, 339)
            ]
        mod_sz = ppf.model.sz_map(par_vec)
        mod_ps = ppf.model.ps_map(par_vec)
        maps_toplot = [
            gaussian_filter(m, smooth) * cbar_fact
            for m in [
                ppf.sz_map,
                mod_sz,
                ppf.sz_map - mod_sz,
                mod_sz + mod_ps,
                ppf.sz_map - mod_sz - mod_ps,
                mod_ps,
                ppf.sz_map - mod_ps,
            ]
        ]

    else:
        if fig is None:
            fig = plt.figure(figsize=(10, 5))
        if axs is None:
            axs = [
                fig.add_subplot(131 + i, projection=ppf.wcs) for i in range(3)
            ]

        mod = ppf.model.sz_map(par_vec)
        if do_ps:
            mod += ppf.model.ps_map(par_vec)
        maps_toplot = [
            gaussian_filter(m, smooth) * cbar_fact
            for m in [ppf.sz_map, mod, ppf.sz_map - mod]
        ]

    noise = ppf.sz_rms * cbar_fact
    if smooth != 0.0:
        noise = gaussian_filter(noise, smooth) / np.sqrt(
            2 * np.pi * smooth**2
        )
    # if par_dic["conv"] < 0:
    #    noise *= -1.0  # for negative SNR

    if isinstance(lims, (tuple, list, np.ndarray)):
        vmin, vmax = lims
    elif isinstance(lims, str):
        if lims == "sym":
            vmin, vmax = np.min(maps_toplot), np.max(maps_toplot)
            vmax = np.max(np.abs([vmin, vmax]))
            vmin = -vmax
    else:
        vmin, vmax = np.min(maps_toplot), np.max(maps_toplot)

    for i, (ax, m) in enumerate(zip(axs, maps_toplot)):
        im = ax.imshow(
            m,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            interpolation="gaussian",
            cmap=cmap,
        )
        ct = ax.contour(
            m / noise,
            origin="lower",
            linestyles="-",
            colors="#00000077",
            linewidths=0.5,
            levels=np.concatenate(
                (np.arange(-50, -2, 2), np.arange(3, 50, 2))
            ),
        )
        ax.set_xlabel("Right ascension (J2000)")
        if i == 0:
            ax.set_ylabel("Declination (J200)")
        else:
            ax.set_ylabel(" ")

    cb = fig.colorbar(im, ax=axs, orientation="horizontal", aspect=40)
    cb.set_label(cbar_label)
    return fig, axs


def plot_acf(ppf, max_delta_tau=None, min_autocorr_times=None):
    n, tau = ppf.autocorr
    dtau = np.abs(np.ediff1d(tau)) / tau[1:]

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(n, tau, "o-")
    axs[1].plot(n[1:], dtau, "o-")
    xlims = np.array(axs[0].get_xlim())
    for ax in axs.flatten():
        ax.set_xlim(*xlims)
        ax_bothticks(ax)

    if min_autocorr_times is not None:
        ylims = np.array(axs[0].get_ylim())
        axs[0].set_ylim(*ylims)
        axs[0].fill_between(
            xlims,
            ylims[0],
            xlims / min_autocorr_times,
            color="tab:green",
            alpha=0.2,
        )

    if max_delta_tau is not None:
        ylims = np.array(axs[1].get_ylim())
        axs[1].set_ylim(*ylims)
        axs[1].fill_between(
            xlims,
            ylims[0],
            max_delta_tau,
            color="tab:green",
            alpha=0.2,
        )

    axs[0].set_xticklabels([])
    axs[0].set_ylabel(r"$\tau_i$")
    axs[1].set_ylabel(r"$|\tau_{i - 1} - \tau_{i}| \; / \; \tau_{i}$")
    axs[1].set_xlabel(r"MCMC step $i$")

    fig.subplots_adjust(left=0.1, right=0.9, hspace=0)
    fig.align_labels(axs)
    fig.suptitle(r"Integrated autocorrelation time $\tau$")

    return fig, axs


def ax_bothticks(ax):
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")


def set_plot_style(style):
    plt.style.use("default")
    if style == "paper":
        style = "./paper.mplstyle"
    plt.style.use(style)
