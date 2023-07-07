.. panco2 documentation master file, created by
   sphinx-quickstart on Thu Jul 21 14:33:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to panco2's documentation!
==================================

panco2 is a Python library dedicated to extracting measurements of the pressure profile of the hot gas inside galaxy clusters from millimeter-wave observations.
The extraction is performed using forward modeling the millimeter-wave signal of clusters and MCMC sampling of a posterior distribution for the parameters given the input data.
Many characteristic features of millimeter-wave observations can be taken into account, such as filtering (both through PSF smearing and transfer functions), point source contamination, and correlated noise.
``panco2`` is further described in :cite:`keruzore_panco2_2023`.

Installation
============

To install, clone the repository and use pip:

.. code-block:: sh

   git clone git@github.com:fkeruzore/panco2.git
   cd panco2
   pip install .


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   method
   examples
   api
   refs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


License & Attribution
=====================

Copyright 2022 Florian Kéruzoré and contributors.
``panco2`` is free software made available under the MIT License. For details see the ``LICENSE.md`` file in the ``git`` repository.
If you use panco2 in your research, please cite :cite:`keruzore_panco2_2023`:

.. code-block:: bib

    @article{keruzore_panco2_2023,
      title = {Panco2: A {{Python}} Library to Measure Intracluster Medium Pressure Profiles from {{Sunyaev-Zeldovich}} Observations},
      shorttitle = {panco2},
      author = {K{\'e}ruzor{\'e}, Florian and Mayet, Fr{\'e}d{\'e}ric and Artis, Emmanuel and {Mac{\'i}as-P{\'e}rez}, Juan-Francisco and {Mu{\~n}oz-Echeverr{\'i}a}, Miren and Perotto, Laurence and Ruppin, Florian},
      year = {2023},
      month = mar,
      journal = {The Open Journal of Astrophysics},
      volume = {6},
      publisher = {{Maynooth Academic Publishing}},
      doi = {10.21105/astro.2212.01439},
      urldate = {2023-03-14},
      langid = {english},
    }
