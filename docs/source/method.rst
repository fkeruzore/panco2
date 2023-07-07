panco2's methodology
********************

The gas component of galaxy clusters, called the intracluster medium (ICM), can bee observed at millimeter wavelengths through the Sunyaev-Zeldovich (SZ) effect, which is the spectral distortion of the cosmic microwave background due to the interaction of its photons with the ICM.
In particular, the dominating component of SZ signal, called the thermal SZ (tSZ) effect, has an amplitude that is proportional to the integral of the ICM electron pressure along the corresponding line of sight:

    .. math:: y(\theta) = \frac{\sigma_{\rm T}}{m_{\rm e} c^2} \int_{\rm LoS(\theta)} P_{\rm e} \; {\rm d}l,
       :label: compton

where :math:`y(\theta)` is the Compton parameter at the position :math:`\theta` on the sky, :math:`\sigma_{\rm T}` is the Thompson cross-section, and :math:`m_{\rm e} c^2` the electron rest-frame energy.

Through this dependence, we can infer the distribution of the electron pressure of the ICM from a mapping of the tSZ effect towards a cluster.
This is what ``panco2`` was designed to do, using forward modeling and MCMC sampling.
Here we give a brief description of the methodology; more details can be found in :cite:`keruzore_panco2_2023`.

.. _sec_fwmod:

Forward modeling of thermal SZ signal
=====================================

Physical modeling of the ICM
----------------------------

``panco2`` assumes spherical symmetry of the ICM pressure distribution, which is described by a pressure profile, :math:`P(r)`.
It uses a radially-binned model, where the pressure profile of the ICM is written as

    .. math:: P(R_i < r < R_{i+1}) = P_i \left(\frac{r}{R_i}\right)^{-\alpha_i},
       :label: nonparam

where a radial binning covering the cluster's spatial extension is defined, and the value of the pressure profile at each radial bin is a parameter of the model.
The value of the pressure between two bins is computed by a power law interpolation between the two closest bins.  The number of parameters of the model therefore depends on the binning chosen by the user.

Compton parameter map computation
---------------------------------

The first step is to compute the Compton parameter :math:`y(\theta)` for each pixel in the map using :eq:`compton`.
In practice, since spherical symmetry of the ICM pressure distribution is assumed, the :math:`y`-map is azimuthally symmetric.
Therefore, we only compute Compton :math:`y` values on a vector of distances from the map center ranging from 0 to (map size / :math:`\sqrt{2}`) with the map pixel size as spacing.
This Compton parameter profile is then interpolated at the radii corresponding to each pixel in the input map, making the model computation much faster than it would be to compute it for every map pixel.
The integration is performed analytically on an infinite line of sight, following the work of :cite:`romero_multi-instrument_2018`.


.. _transfer_fct:

Map filtering
-------------

For the model map to be comparable to data, its signal must be affected by the same filtering.

``panco2`` can account for two types of filtering.
First, a Gaussian kernel can be applied to account for filtering due to the instrumental point-spread function.
Whether this filtering should be used, as well as the width of the kernel, has to be specified by the user.
In addition, filtering due to data processing can be accounted for by applying an arbitrary kernel to the data, referred to as "transfer function".
This transfer function can be azimuthally symmetric (for an isotropic filtering of the map) or not, and needs to be provided by the user.

After these operations, the :math:`y`-map has been affected by a similar filtering to that undergone by the signal of interest in the data, making the maps comparable.

Map conversion
--------------

Users may wish to extract a pressure profile measurement from a millimeter-wave map that is not converted to Compton parameter (e.g. surface brightness units or CMB temperature).
Converting between these unit systems can be done through a conversion coefficient, that depends on many observational characteristics, and therefore carries uncertainty.
To propagate this uncertainty to final science products, ``panco2`` treats the conversion coefficient as a prior-dominated model parameter that is marginalized on (see :any:`mcmc`)
In the forward modeling, the filtered Compton parameter map is multiplied by the conversion coefficient corresponding to the position in the parameter space, creating a map that has been filtered similarly to the input data and is in the same units.


Point source contamination
--------------------------

SZ observations are very prone to contamination by point sources, especially at frequencies lower than 217 GHz, at which the tSZ signal is a decrement in CMB surface brightness that can be compensated -- partially or totally -- by the (positive) fluxes of astrophysical sources.
To take it into account, ``panco2`` can use the methodology described in :cite:`keruzore_exploiting_2020`, which consists in treating the fluxes of known point sources as parameters of the fit, constrained by a prior knowledge of each source's flux.
This allows to propagate the uncertainty in the point sources fluxes to ``panco2``'s results.
If asked, ``panco2`` will therefore add for each source a 2D gaussian function representing the NIKA2 instrumental beam, with an amplitude given by the flux of the source. 
This addition is performed before the convolution by the transfer function, as the point sources re just as affected by the filtering as the SZ signal.

Integrated signal
-----------------

Large-scale signal is often missing from SZ observations because of the filtering due to data processing (see :any:`transfer_fct`).
SZ cluster surveys often release measurement of the integrated Compton parameter :math:`Y_R` within a radius :math:`R` for their cluster detections.
Depending on how large :math:`R` is, this may contain information on large-scale signal filtered in the data, and can therefore provide effective constraining power on the pressure profile in the outskirts of clusters.
Similarly to the model map, a value of integrated Compton parameter corresponding to the model associated to a position in the parameter space can be computed as:

.. math:: Y_R = 4\pi\frac{\sigma_\mathrm{T}}{m_e c^2} \int_0^R r^2 P(r) \,\mathrm{d}r
   :label: yinteg

where :math:`R` must be provided by the user, and :math:`P(r)` can be computed from eq. :eq:`nonparam`.

Summary
-------

The parameters of the model used by panco can be summarized in a vector :math:`\vartheta` composed of:

- The parameters of the pressure profile: :math:`P_i,\; i = 0 \cdots n_{\rm bins}`;
- The "calibration coefficient" to convert Compton parameter measurements to the units of the input map;
- If asked, a zero-level can also be used as a free parameter to account for possible
  residual noise,
- If asked, a flux value for each known point source in the map.

From these parameters, a model map :math:`\mathcal{M}(\vartheta)` can be generated that can be directly compared to the input data map.
This procedure is illustrated in |fig_fwmod|.

.. |fig_fwmod| image:: fwmod.pdf
  :width: 400
  :alt: this figure





.. _mcmc:

Pressure profile fitting
========================

``panco2`` aims at finding the probability distribution for the parameters of the chosen model given the input data.
It does so by using Bayesian Monte Carlo Markov Chains (MCMC) sampling: let :math:`D` be the input data and :math:`\vartheta` the set of parameters of the model.
The probability for :math:`\theta` to accurately describe the data is given by the Bayes theorem:

.. math:: P(\vartheta \,|\, D) = \frac{P(D \,|\, \vartheta) \, P(\vartheta)}{P(D)},
   :label: post

where :math:`P(\vartheta \,|\, D)` is called the *posterior distribution*, :math:`P(D \,|\, \vartheta)` is the *likelihood function* comparing the model to the data, :math:`P(\vartheta)` is the *prior distribution* encapsulating the user's prior knowledge about the parameters, and :math:`P(D)` is the data evidence, here treated as a normalization constant.

The likelihood function
-----------------------

``panco2`` uses a multivariate Gaussian likelihood function to compare the model to data.

.. math:: 
   \mathrm{log} \, \mathcal{L}(\vartheta) &= \mathrm{log} \, P(D \, | \, \vartheta) \\
      &= - \frac{1}{2} \left(D - \mathcal{M}(\vartheta)\right)^{\rm T} \Sigma^{-1} \left(D - \mathcal{M}(\vartheta)\right)
	  - \frac{1}{2} \left(\frac{Y_R^{\rm meas.} - Y_R(\vartheta)}{\Delta Y_R^{\rm meas.}}\right)^2
   :label: loglike

where :math:`D` is the measured millimeter-wave map, :math:`\Sigma` is the noise covariance matrix, :math:`\mathcal{M}(\vartheta)` is the model map described in :any:`sec_fwmod`, :math:`Y_R(\vartheta)` is the integrated SZ signal computed from Eq. :eq:`yinteg`, and :math:`Y_R^{\rm meas.}` and :math:`\Delta Y_R^{\rm meas.}` are the measured integrated SZ signal and its uncertainty, respectively.
``panco2`` provides different routines to compute the noise covariance matrix (and its inverse) from different types of inputs -- see :any:`sec_examples`
Otherwise, the noise is considered to be white and the pixels uncorrelated, simplifying Eq. :eq:`loglike` and greatly improving the computation time.


The prior distribution
----------------------

``panco2`` considers priors on each parameter to be uncorrelated, meaning the prior distribution is the product of the priors on each individual parameter.
These individual priors are to be specified by the user using the distributions implemented in the ``scipy.stats`` module, offering a very high flexibility in the analysis.


Posterior distribution sampling
-------------------------------

The fit is performed by Monte Carlo Markov Chain (MCMC) sampling of the posterior probability distribution of Eq. :eq:`post`.
We specifically use the multithreaded affine-invariant ensemble sampling implemented in the ``emcee`` Python library :cite:`foreman-mackey_emcee_2019`.
The number of walkers, as well as the number of threads to use, is to be specified by the user.
In the following, we summarize some specificities of the MCMC sampling implemented in ``panco2``.

Starting point
^^^^^^^^^^^^^^

The starting point of the random walk of the MCMC is determined by randomly drawing positions from the prior distribution.
One position is drawn for each of the walkers of the affine-invariant ensemble sampler.

Chains convergence
^^^^^^^^^^^^^^^^^^

One crucial step of MCMC analyses is to know at which point the chains have reached convergence, indicating that more sampling is not needed and that the MCMC can be stopped.
We follow the recommendations of ``emcee`` developers and implement a convergence check based on the autocorrelation function of the chains.
At regular intervals (with a frequency to be specified by the user), the mean autocorrelation of the chains :math:`\tau` is computed, and convergence is accepted if:

#. The current chain length is larger than a multiple of the mean autocorrelation (with the specific number to be specified);

#. The mean autocorrelation has changed by less than given fraction (e.g. 1%, but any number may be used) over the last two convergence checks.

These two criteria ensure that the sampling contains enough uncorrelated random positions in the parameter space for inference, and that the chains have converged to the final posterior distribution.

Results exploitation
--------------------

Once the chains have reached convergence, they constitute a random sample for which the probability distribution is the posterior distribution of Eq. :eq:`post`.
They are then cleaned by removing an initial burn-in length and by discarding the majority of the sample, only considering samples separated by a length to be specified by the user.
These are used to infer measurements of the physical properties of the ICM.

The pressure profile of the ICM is the property directly probed by our fit.
Its value is given by the computation of our model for the set of parameters that maximize the posterior distribution sampled in our fit.
Each set of parameters in the final chains is then used to compute a pressure profile on a wide radius range.
The dispersion of these profiles gives a measurement of the statistical error on the pressure for the whole radial range considered.