#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import emcee
from iminuit import Minuit
import sys
import os
import shutil
import json
import time
import argparse
from multiprocessing import Pool
import pdb
from _cluster import Cluster
import _data
import _xray
import _probability
import _utils
import _model_gnfw, _model_nonparam
import _fit_gnfw_on_non_param
import _results

np.set_printoptions(precision=3)
_utils.ignore_astropy_warnings()

"""
This is PANCO2's main routine.

Sections:
    1. Arguments and options
    2. Initializations
    3. Posterior distribution definition
    4. MCMC starting point construction
    5. MCMC sampling
    6. Chains management
    7. Results exploitation
"""

# =============================================================================== #
# --  1. ARGUMENTS AND OPTIONS  ------------------------------------------------- #
# =============================================================================== #

parser = argparse.ArgumentParser()
parser.add_argument(
    "--restore",
    default=None,
    help="Resore previously sampled chains",
)
args = parser.parse_args()

# ===== Load option file ===== #
from _default_params import *

if args.restore is not None:
    config_file = os.path.join(args.restore, "panco_params.py")
    with open(config_file) as f:
        code = compile(f.read(), config_file, "exec")
        exec(code, globals(), locals())
    args.restore = True
else:
    from panco_params import *

    if not os.path.isdir(path_to_results):
        os.makedirs(path_to_results)

    _ = shutil.copy2("panco_params.py", path_to_results)


out_chains_file = path_to_results + "chains.npz"

# ===== Run on simulation ===== #
if sim:
    file_nk2 = file_nk2_sim
    truth = json.load(open(file_truth, "r"))
    if do_ps:
        truth["ps_fluxes"] = np.array(truth["ps_fluxes"])
else:
    truth = None

if not os.path.isfile(file_nk2):
    raise Exception("NIKA2 map nor found. Aborting.")

if os.path.isfile(out_chains_file) and (not args.restore):
    print("Results already exist: " + path_to_results)
    prompt = input("Are you sure you want to erase them? [y/N] ")
    if prompt not in ["Y", "y", "yes"]:
        sys.exit("Exitting.")

np.seterr(all="ignore")

# =============================================================================== #
# --  2. INITIALIZE ALL THINGS  ------------------------------------------------- #
# =============================================================================== #

print("==> Initialization")

cluster = Cluster(**cluster_kwargs)
model_type = model_type.lower()
if model_type == "gnfw":
    model = _model_gnfw.ModelGNFW(cluster, fit_zl=fit_zl, do_ps=do_ps)
elif model_type == "nonparam":
    model = _model_nonparam.ModelNonParam(cluster, fit_zl=fit_zl, do_ps=do_ps)
else:
    raise Exception("Unrecognized `model_type`: " + model_type)

# ===== NIKA2 radii maps ===== #
if (args.restore) and (file_covmatinv is None):
    file_covmatinv = path_to_results + "inv_covmat.npy"
data_nk2, rms_nk2, covmat, inv_covmat, wcs_nk2, reso_nk2 = _data.read_data(
    file_nk2,
    hdu_data,
    hdu_rms,
    crop=crop,
    coords_center=coords_center,
    inv_covmat=np.load(file_covmatinv) if (file_covmatinv is not None) else None,
    file_noise_simus=file_noise_simus,
)
reso_nk2_arcsec = reso_nk2.to("arcsec").value
npix = data_nk2.shape[0]
if (not args.restore) and (inv_covmat is not None):
    np.save(path_to_results + "inv_covmat.npy", inv_covmat)

# ===== Non-param binning ===== #
if args.restore:
    radius_tab = np.load(path_to_results + "radius_tab.npy")
else:
    radius_tab = None

# ===== SZ model computation initialization ===== #
model.init_profiles_radii(
    reso=reso_nk2_arcsec,
    npix=npix,
    center=(0.0, 0.0),
    r_min_z=1e-3,
    r_max_z=5 * cluster.R_500_kpc,
    nbins_z=100,
    mode=integ_mode,
    radius_tab=radius_tab,
    n_bins=n_bins,
)

if model_type == "nonparam":
    np.save(path_to_results + "radius_tab.npy", model.radius_tab)

print(
    f"    Large scale signal constrained by Y_{integ_mode}",
    f"= ({model.init_prof['integ_Y']:.2f}",
    f" +/- {model.init_prof['err_integ_Y']:.2f}) kpc2",
)

# ===== Point sources ===== #
if do_ps:
    print("    Initializing point sources")
    data_nk2, ps_fluxes_sed = model.init_point_sources(
        path=path_ps,
        data=data_nk2,
        wcs=wcs_nk2,
        reso=reso_nk2,
        beam=17.6 * u.arcsec,
        ps_prior_type="pdf",
        # fixed_error=5e-5,
        do_subtract=not sim,
        which_ps=which_ps,
    )
    nps = model.init_ps["nps"]

# ===== Initialize transfer function ===== #
side_tf = 9 * u.arcmin  # TODO: this should NOT have to be hardcoded
pad_tf = int(0.5 * _utils.adim(side_tf / reso_nk2))
model.init_transfer_function(
    file_tf,
    data_nk2.shape[0],
    pad_tf,
    reso_nk2.to("arcsec").value,
)

# ===== X-ray data ===== #
if os.path.isfile(x_file):
    x_profiles = _xray.recover_x_profiles(x_file)
else:
    x_profiles = None

# ===== Finish up model initialization ===== #
model.init_param_indices()

# =============================================================================== #
# --  3. DEFINE POSTERIOR PROBABILITY DISTRIBUTION  ----------------------------- #
# =============================================================================== #

priors_computer = _probability.Prior(model, **prior_kwargs)
if inv_covmat is not None:

    def log_lhood(par):
        return _probability.log_lhood_covmat(par, model, data_nk2, inv_covmat)


else:

    def log_lhood(par):
        return _probability.log_lhood_nocovmat(par, model, data_nk2, rms_nk2)


def log_post(params):
    """
    The log-posterior to be sampled.
    Returns a tuple (log-posterior, log-prior, log-likelihood).
    The MCMC looks at the first return (the log posterior),
        the other two (likelihood and prior) can be retrieved
        with emcee blobs and used for other analyses
    """
    par = model.params_to_dict(params)
    _log_prior = priors_computer(par)
    check_mass = model.check_mass(par, x_profiles=x_profiles)

    _log_lhood = -np.inf
    if np.isfinite(_log_prior) and check_mass:
        _log_lhood = log_lhood(par)
    if np.isnan(_log_lhood):
        _log_lhood = -np.inf

    return _log_prior + _log_lhood, _log_prior, _log_lhood


# =============================================================================== #
# --  4. STARTING POINT CONSTRUCTION  ------------------------------------------- #
# =============================================================================== #

if model_type == "gnfw":
    pos = [cluster.A10_params, [-11.9]]
else:
    pos = [_model_gnfw.gNFW(model.radius_tab, *cluster.A10_params), [-11.9]]

if fit_zl:
    pos.append([0.0])
if do_ps:
    pos.append(ps_fluxes_sed)

pos = np.concatenate(pos)

if (model_type == "gnfw") and (not args.restore):  # Migrad the likelihood

    print("\n==> Finding optimal starting point with Migrad...")

    def chi2(params):
        par = model.params_to_dict(params)
        return -2.0 * _probability.log_lhood_nocovmat(par, model, data_nk2, rms_nk2)

    index_calib = model.indices["calib"]
    fix = np.zeros(pos.size, dtype=bool)
    fix[index_calib] = True
    limit = [(0.0, 5.0 * p) for p in pos]
    limit[index_calib] = (pos[index_calib], pos[index_calib])

    m = Minuit.from_array_func(chi2, pos, fix=fix, limit=limit, errordef=1)
    mig = m.migrad()
    pos = np.array([p.value for p in mig.params])

par = model.params_to_dict(pos)  # for debugging purposes

if not args.restore:
    print("    MCMC starting point:")
    for k in par.keys():
        print(f"        {k} = {par[k]}")

# =============================================================================== #
# --  5. MCMC SAMPLING  --------------------------------------------------------- #
# =============================================================================== #

# Foolproof-ing
nsteps, burn, ncheck, ndim = (int(nsteps), int(burn), int(ncheck), len(pos))

init_pos = []
for i in range(nchains):
    pos_ = []
    for j in pos:
        if j != 0.0:
            pos_.append(np.random.normal(j, np.abs(1e-2 * j)))
        else:
            pos_.append(np.random.normal(0.0, 1e-5))
    init_pos.append(pos_)

# Crash now if you want to crash
_ = log_post(pos)

# ===== MCMC Sampling: Posterior ===== #
if not args.restore:
    print("\n==> MCMC sampling...")
    print(f"    Convergence will be checked every {ncheck} steps")

    with Pool(processes=nthreads) as pool:
        ti = time.time()
        sampler = emcee.EnsembleSampler(
            nchains, ndim, log_post, pool=pool, moves=emcee.moves.DEMove()
        )

        for sample in sampler.sample(init_pos, iterations=nsteps, progress=True):

            it = sampler.iteration
            if it % ncheck != 0.0:
                continue  # The following is only executed if it = n * ncheck

            print(f"    {it} iterations")
            # ===== Save chains ===== #
            blobs = sampler.get_blobs()
            chains = {
                "chains": sampler.chain,
                "lnprob": sampler.lnprobability,
                "lnprior": blobs[:, :, 0].T,
                "lnlike": blobs[:, :, 1].T,
            }
            np.savez(out_chains_file, **chains)

            """
            Test convergence
            Accept convergence if:
            - Less than 1/3 of the chains are cut in clean_chains
            - Gelman-Rubin is ok
            """

            if it <= burn:  # No check at first step (no samples)
                continue
            results = _results.Results(
                out_chains_file,
                burn,
                model,
                path_to_results,
            )
            nchains_new = results.clean_chains(
                clip_at_sigma=2.0, clip_at_autocorr=50, thin_by=None
            )
            R_hat = results.gelman_rubin_stat()
            print(f"R_hat = {R_hat}")

            converged = (
                np.all(R_hat < 1.02)
                and (nchains_new >= 2 / 3 * nchains)
                and (it >= 5 * burn)
            )

            if converged:
                break

    tf = time.time()
    print(
        f"    MCMC running time: {time.strftime('%Hh %Mm %Ss', time.gmtime(tf - ti))}"
    )

    np.savez(out_chains_file, **chains)
    del chains

# =============================================================================== #
# --  6. MANAGE CHAINS  --------------------------------------------------------- #
# =============================================================================== #
print("\n==> Managing Markov chains")
results = _results.Results(
    out_chains_file,
    burn,
    model,
    path_to_results,
    x_profiles=x_profiles,
    truth=truth,
    dof=data_nk2.size - ndim,  # Could be slightly off if mass and/or Yinteg constraint
)
results.plot_mcmc_diagnostics(cleaned_chains=False)
results.clean_chains(clip_at_sigma=2.0, clip_at_autocorr=50, thin_by=thin_by)
results.compute_solid_statistic("max-post")
results.plot_mcmc_diagnostics(cleaned_chains=True)

# =============================================================================== #
# --  7. RESULTS EXPLOITATION  -------------------------------------------------- #
# =============================================================================== #

print("\n==> Thermodynamical properties computation")

# ===== Compute profiles ===== #
r_display = np.logspace(
    np.log10(model.reso_kpc), np.log10(6.0 * cluster.R_500_kpc), 100
)
# ===== Physics ===== #
if model_type == "nonparam":
    thermo_profiles_np = results.chains2physics(model.radius_tab, nthreads=nthreads)
    results.thermo_profiles_np = thermo_profiles_np
else:
    results.thermo_profiles_np = None

thermo_profiles = results.chains2physics(
    r_display,
    nthreads=nthreads,
    method=interp_method_np,
)
results.thermo_profiles = thermo_profiles

print("\n==> Plotting things")

# ===== Plot thermo profiles ===== #
vlines = {
    "NIKA2 beam": model.angle_distances_conversion(9 * u.arcsec).to("kpc").value,
    "NIKA2 FoV": model.angle_distances_conversion(3.25 * u.arcmin).to("kpc").value,
    "$R_{500}$": cluster.R_500_kpc,
}
print("    Thermodynamical profiles...")
results.plot_profiles(
    x=False if (sim or x_profiles is None) else True,
    r_limits=[r_display.min(), r_display.max()],
    vlines=vlines,
)

# ===== Parameter distributions ===== #
print("    Parameter distributions...")
results.plot_distributions(
    alsoplot=["chi2"],
    color=None,
)

# ===== Data, model, residuals ===== #
print("    Data / model / residuals...")
results.plot_dmr(data_nk2, 0.4247 * 10 / reso_nk2_arcsec, rms=rms_nk2)

# ===== Integrated values ===== #
if x_profiles is not None:
    print("\n==> Integrated values from your fit")
    results.compute_integrated_values()
    results.plot_integrated_values()
    print(
        f"    R_500 = {results.all_int['R_500']:.2f} +/-",
        f"{np.nanstd(results.all_int['R_500_dist'], ddof=1):.2f} kpc",
    )
    print(
        f"    M_500 = {results.all_int['M_500'] / 1e14:.2f} +/-",
        f"{np.nanstd(results.all_int['M_500_dist'], ddof=1) / 1e14:.2f} e14 Msun",
    )
    print(
        f"    Y_500 = {results.all_int['Y_500']:.2f} +/-",
        f"{np.nanstd(results.all_int['Y_500_dist'], ddof=1):.2f} kpc2",
    )

print("\n==> End of program.")
print(f"    Your results are stored in: {path_to_results}\n")
