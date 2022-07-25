Using PANCO2
************

This section explains how to use ``PANCO2``, *i.e.* how to run the code to produce results.
The ``git`` repository includes two demonstrations that can be launched for a new user
to get their hands in the code.
They consist in performing a complete ``PANCO2`` fit on a simulated map of 
ACT-CL J0215.4+0030, in gNFW and non-parametric mode.

To launch them, simply run::

   $ cp Demo/demo_params_np.py panco_params.py
   $ ./panco.py

for the non-parametric mode, or, for gNFW::

   $ cp Demo/demo_params_gNFW.py panco_params.py
   $ ./panco.py

The main code: ``panco.py``
===========================

The main code is a compilation of the successive steps described in :numref:```PANCO2``'s algorithm`.
``panco.py`` is an executable python3 file to be called as follows::

   $ ./panco.py [--restore=path]

The ``--restore`` argument is optional and can be passed to re-process the data of a
previous ``PANCO2`` execution, by restoring the chains rather than re-sampling.
For example, if you have results in ``./Results/demo`` and you have modified a part of
the code that does not affect the sampling, you can reprocess the results with::

   $ ./panco.py --restore=./Results/demo/

``panco.py`` is organized in seven sections:

#. Arguments and options

   This is where everything the user gives ``PANCO2`` is managed, especially the 
   ``panco_params.py`` configuration file -- see next

#. Initializations

   This part initializes the functions that will be needed for the model evaluation,
   and computes everything that only needs to be computed once

#. Posterior probability distribution definition

#. MCMC starting point construction

   See :numref:`Starting point`

#. MCMC sampling

   Run the MCMC and monitor its progression and convergence, save chains regularly

#. Markov chains management

   Clean up the chains and format them to prepare for the inference of physical 
   properties, perform the MCMC diagnostics

#. Results exploitation

   Convert the posterior distribution sampled during the MCMC into physical properties
   (thermodynamical profiles, integrated quantities) and do the plots

The ``panco_params.py`` configuration file
==========================================

The user input part of ``PANCO2`` is managed via a python script called ``panco_params.py``.
In this script, the user defines all the options to be used in the fit, as well as the
path to the input data, to the results, and many other parameters.

To run ``PANCO2``, the ``panco_params.py`` file must be created by the user and placed in the 
same directory as the ``panco.py`` executable.
When running ``panco.py``, the ``panco_params.py`` is loaded and copied at the location
where the results will be saved, so that you always have a trace of how the results were
obtained when analyzing them.
If the ``--restore=[path]`` option is given, ``PANCO2`` loads the options from the 
``panco_params.py`` located in ``[path]``.

The configuration files included in the demonstration include detailed comments on what
options the user can personalize for both a gNFW and a non-parametric fit.
Both these examples are repeated below.

.. raw:: latex

    \newpage

``/Demo/demo_params_np.py``

.. literalinclude:: ../Demo/demo_params_np.py
   :linenos:


.. raw:: latex

    \newpage

``/Demo/demo_params_gnfw.py``

.. literalinclude:: ../Demo/demo_params_gnfw.py
   :linenos:

Output files
============

When finished, ``PANCO2`` will prompt the path where all results were saved, which is
the path that has been specified as ``path_to_results`` in ``panco_params.py``.
Looking in this directory, several files can be found:

* ``Plots``: a directory where your plots are saved;

* ``chains.npz``: a dictionary file where the Markov chains are stored;

* ``integrated_values_samples.npz``: a dictionary file the integrated values resulting 
  from the fit, if X-ray data were available;

* ``panco_params.py``: your input parameter file, for future reference;

* ``radius_tab.npy``: in non parametric mode, the radial bins at which the pressure was
  evaluated during the fit;

* ``thermo_profiles.npz``: the thermodynamical profiles computed for all points in the
  final PDF.

Many of these files are ``.npz`` files, which are files used by ``numpy`` to store
dictionaries, similarly to ``.save`` files in ``IDL``.
Here is an example on how to read them in ``python``::

    import numpy as np
    f = np.load("yourfile.npz")
    print(f.files())  # Will show the available entries
    a = f["a"]  # To access the data in the "a" entry




