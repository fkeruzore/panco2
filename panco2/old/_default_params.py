"""
These are the parameters loaded by default. Each variable is overriden
by its counterpart in ``panco_params.py`` if it exists.
Not all parameters are written here: for some of them, the user
input is too important to be left to a default.
"""

# ===== Options ===== #
do_ps = False
fit_zl = True
sim = False

interp_method_np = "interp_powerlaw"  # "interp_spline", "interp_powerlaw", or "gNFW"

crop = None
coords_center = None

# ===== Results paths ===== #
ana_name = ""  # Name of your analysis (i.e. of the save directory)
path_to_results = f"./Results/{ana_name}/"  # Path to the results

# ===== MCMC parameters (will be overriden if --debug option is passed) ===== #
nsteps = 1e5  # Initial number of steps to make, can stop before if converged
ncheck = 1e3  # Check convergence every `ncheck` steps
burn = 1e3  # Burn-in
nchains = 30  # Number of chains
nthreads = 30  # Number of threads to use
thin_by = "autocorr"  # Only keep one point every n steps, can be "autocorr"

# ===== Data paths ===== #
file_nk2 = ""
file_covmatinv = None
file_noise_simus = None
hdu_data = 0
hdu_rms = 1
path_ps = ""
file_tf = ""
x_file = ""

if sim:  # Testing on simulations
    file_nk2_sim = ""
    file_truth = ""

integ_mode = "500"  # "tot" or "500", which Y is used in the likelihood
