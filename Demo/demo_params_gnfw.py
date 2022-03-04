import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

# fmt: off

### Options
do_ps      = False  # Treat point source fluxes as free parameters of the fit
fit_zl     = True   # Fit the zero level of the map
sim        = True   # Run on a simulation

model_type = "gNFW"         # "gNFW" or "Nonparam" (case insensitive)

crop = 6.5 * u.arcmin # Only use the NIKA2 data in a given FoV, can also be None
coords_center = SkyCoord(33.868157 * u.deg, 0.5088958 * u.deg) # Can be None

### Results paths
ana_name = "demo_gNFW"   # Name of your analysis (i.e. of the save directory)
path_to_results = f"./Results/{ana_name}/"  # Path to the results

### MCMC parameters (will be overriden if --debug option is passed)
nsteps   = 1e6  # Initial number of steps to make, can stop before if converged
ncheck   = 1e4  # Check convergence every `ncheck` steps
burn     = 1e4  # Burn-in
nchains  = 30   # Number of chains
nthreads = 30   # Number of threads to use
thin_by  = "autocorr" # Only keep one point every n steps, can be "autocorr"

### Data paths
file_nk2 = "./Demo/Data/map_input_sim.fits"
hdu_data = 4
hdu_rms  = 5
path_ps  = "./PointSources/"
file_tf  = "./Demo/Data/transfer_function.fits"
x_file   = "./Demo/Data/parprod4.json"

if sim: # Testing on simulations
    file_nk2_sim = "./Demo/Data/map_input_sim.fits"
    file_truth = "./Demo/Data/sim_truth.json"

### Cluster parameters
cluster_kwargs = {
    "z": 0.865,
    "Y_500": 1.8e-4 * u.arcmin ** 2,
    "err_Y_500": 0.5e-4 * u.arcmin ** 2,
    "theta_500": None,
}
integ_mode = "500"  # "tot" or "500", which Y is used in the likelihood

### Prior parameters, see panco_likelihood.Prior
prior_kwargs = {
    "ref": "A10",           # Can also be "Planck"
    "sigma": 1.0,           # Dispersion in units of central value
    "calib": (-11.9, 1.19), # (Value, dispersion) for y2mJy/beam
    "flat": False,          # Flat priors (if False, Gaussian)
    # "a":(2.0, 2.0),       # You can even override the slopes!
    # "b":(3.5, 1.0),
}

# fmt: on
