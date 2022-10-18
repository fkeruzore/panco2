.. panco2 documentation master file, created by
   sphinx-quickstart on Thu Jul 21 14:33:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to panco2's documentation!
==================================

panco2 is a Python library dedicated to extracting measurements of the pressure profile of the hot gas inside galaxy clusters from millimeter-wave observations.
The extraction is performed using forward modeling the millimeter-wave signal of clusters and MCMC sampling of a posterior distribution for the parameters given the input data.
Many characteristic features of millimeter-wave observations can be taken into account, such as filtering (both through PSF smearing and transfer functions), point source contamination, and correlated noise.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   fwmod
   mcmc
   example_C2_NIKA2.ipynb
   api
   refs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
