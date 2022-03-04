# PANCO

Pipeline for the Analysis of NIKA2 Cluster Observations

## Launching
1. Create your `panco_params.py` file (an example is given in `demo_params.py`)
2. Run `panco.py` !

### Running options
- `./panco.py --restore=path/to/results`: Restore previously sampled chains (*i.e.* do not run the MCMC, useful for plots modifications etc).

### History
- Oct. 2020: release 1.0_oct20
- Nov. 2019: creation

### Notes and TODOs
- There is no noise covariance matrix management yet
- There is no SZ relativistic corrections computation yet
- There is no management of Planck maps yet
