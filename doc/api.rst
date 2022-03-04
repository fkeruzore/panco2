API
***

``_cluster.py``
===============

.. automodule:: _cluster
.. autoclass:: Cluster

``_data.py``
===============

.. automodule:: _data
.. autofunction:: read_data

``_model.py``
=============

The ``Model`` class
-------------------
.. automodule:: _model
.. autoclass:: Model
.. automethod:: Model.init_point_sources
.. automethod:: Model.init_transfer_function
.. automethod:: Model.params_to_dict
.. automethod:: Model.convolve_tf
.. automethod:: Model.__call__

``_model_gnfw.py``
==================
.. automodule:: _model_gnfw

``ModelGNFW``
-------------
.. autoclass:: ModelGNFW
.. automethod:: ModelGNFW.init_profiles_radii
.. automethod:: ModelGNFW.init_param_indices
.. automethod:: ModelGNFW.dict_to_params

``_model_nonparam.py``
======================
.. automodule:: _model_nonparam

``ModelNonParam``
-----------------
.. autoclass:: ModelNonParam
.. automethod:: ModelNonParam.init_profiles_radii
.. automethod:: ModelNonParam.init_param_indices
.. automethod:: ModelNonParam.dict_to_params

``_probability.py``
===================
.. automodule:: _probability
.. autofunction:: log_lhood_nocovmat

The ``Prior`` class
-------------------
.. autoclass:: Prior
.. automethod:: Prior.__call__

``_results.py``
===============
.. automodule:: _results
.. autoclass:: Results

Data analysis methods
---------------------
.. automethod:: Results.clean_chains
.. automethod:: Results.compute_solid_statistic
.. automethod:: Results.chains2physics
.. automethod:: Results.compute_integrated_values

Plotting methods
----------------
.. automethod:: Results.plot_mcmc_diagnostics
.. automethod:: Results.plot_distributions
.. automethod:: Results.plot_profiles
.. automethod:: Results.plot_dmr

``_fit_gnfw_on_non_param.py``
=============================
.. automodule:: _fit_gnfw_on_non_param
.. autoclass:: GNFWFitter
.. automethod:: GNFWFitter.do_fit_minuit
.. automethod:: GNFWFitter.do_fit_mcmc
.. automethod:: GNFWFitter.manage_chains
.. automethod:: GNFWFitter.compute_thermo_profiles

``_xray.py``
============
.. automodule:: _xray
.. autofunction:: recover_x_profiles

``_utils.py``
=============
.. automodule:: _utils
.. autoclass:: LogLogSpline
.. automethod:: LogLogSpline.__call__
.. automethod:: LogLogSpline.differentiate

.. autofunction:: interp_powerlaw
.. autofunction:: adim
.. autofunction:: sph_integ_within
.. autofunction:: cyl_integ_within
.. autofunction:: prof2map

