import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.coordinates import SkyCoord
import scipy.stats as ss
import sys

sys.path.append("../..")

import panco2 as p2


def ax_bothticks(ax):
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")


ppf = p2.PressureProfileFitter.load_from_file("./C2_nk2.panco2")
ppf.add_point_sources(
    [SkyCoord("12h00m03s +00d00m45s"), SkyCoord("12h00m00s -00d00m30s")], 18.0
)
ppf.define_priors(
    P_bins=[ppf.model.priors[f"P_{i}"] for i in range(ppf.model.n_bins)],
    conv=ppf.model.priors["conv"],
    zero=ppf.model.priors["zero"],
    ps_fluxes=[ss.norm(1e-3, 1e-4), ss.norm(1e-3, 1e-4)],
)

par_vec = np.concatenate(
    (
        p2.utils.gNFW(ppf.model.r_bins, *ppf.cluster.A10_params),
        [-12.0, 0.0, 1e-3, 1e-3],
    )
)
par_dic = ppf.model.par_vec2dic(par_vec)
y_map_unfilt = ppf.model.compton_map(par_vec) * 1e6
y_map_filt = ppf.model.filter(y_map_unfilt)
sz_map = ppf.model.sz_map(par_vec) * 1e3
tot_map = sz_map + ppf.model.ps_map(par_vec) * 1e3

ymax = np.max(y_map_unfilt)
szmax = np.max(np.abs(sz_map))

# ==================================================== Figure

fig = plt.figure(figsize=(4.1, 15))

# Pressure profile
ax = fig.add_subplot(511)
r_range = np.logspace(
    np.log10(0.5 * ppf.model.r_bins[0]),
    np.log10(2 * ppf.model.r_bins[-1]),
    100,
)
ax.plot(
    r_range,
    ppf.model.pressure_profile(r_range, par_vec),
    "--",
    color="tab:blue",
)
ax.loglog(
    ppf.model.r_bins,
    par_vec[ppf.model.indices_press],
    "o-",
    color="tab:blue",
)
ax.set_xlabel("$r \; [{\\rm kpc}]$")
ax.set_ylabel("$P_{\\rm e}(r) \; [{\\rm keV \cdot cm^{-3}}]$")
ax_bothticks(ax)
fig.subplots_adjust(top=0.99, right=0.93)

axs = []
axs_cbars = []
axs.append(ax)
axs_cbars.append(ax)

# maps
gs = GridSpec(
    ncols=20, nrows=4, hspace=0.05, wspace=0, top=0.8, bottom=0.01, right=0.95
)


def plotmap(m, ax, cax, label, **kwargs):
    im = ax.imshow(m, interpolation="gaussian", **kwargs)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label)
    cax.yaxis.tick_left()
    cax.yaxis.set_label_position("left")
    ax_bothticks(ax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


axs.append(fig.add_subplot(gs[0, 1:]))
axs_cbars.append(fig.add_subplot(gs[0, 0]))
ax = axs[-1]
ax_cb = axs_cbars[-1]
plotmap(
    y_map_unfilt,
    ax,
    ax_cb,
    "Compton$-y \\times 10^6$",
    vmin=0.0,
    vmax=ymax,
    cmap="magma_r",
)

axs.append(fig.add_subplot(gs[1, 1:]))
axs_cbars.append(fig.add_subplot(gs[1, 0]))
ax = axs[-1]
ax_cb = axs_cbars[-1]
plotmap(
    y_map_filt,
    ax,
    ax_cb,
    "Compton$-y \\times 10^6$",
    vmin=0.0,
    vmax=ymax,
    cmap="magma_r",
)

axs.append(fig.add_subplot(gs[2, 1:]))
axs_cbars.append(fig.add_subplot(gs[2, 0]))
ax = axs[-1]
ax_cb = axs_cbars[-1]
plotmap(
    sz_map,
    ax,
    ax_cb,
    "tSZ surface brightness @150 GHz",
    vmin=-szmax,
    vmax=szmax,
    cmap="RdBu_r",
)

axs.append(fig.add_subplot(gs[3, 1:]))
axs_cbars.append(fig.add_subplot(gs[3, 0]))
ax = axs[-1]
ax_cb = axs_cbars[-1]
plotmap(
    tot_map,
    ax,
    ax_cb,
    "Surface brightness @150 GHz",
    vmin=-szmax,
    vmax=szmax,
    cmap="RdBu_r",
)

for ax, letter in zip(axs, "ABCDE"):
    ax.text(
        0.95,
        0.95,
        f"({letter})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=16,
    )

fig.subplots_adjust(left=0.18)
fig.align_ylabels(axs)
fig.align_ylabels(axs_cbars)
fig.savefig("./fwmod.svg")
