panco2's methodology
********************

The gas component of galaxy clusters, called the intracluster medium (ICM), can bee observed at millimeter wavelengths through the Sunyaev-Zeldovich (SZ) effect, which is the spectral distortion of the cosmic microwave background due to the interaction of its photons with the ICM.
In particular, the dominating component of SZ signal, called the thermal SZ (tSZ) effect, has an amplitude that is proportional to the integral of the ICM electron pressure along the corresponding line of sight:
    .. math:: y(\theta) = \frac{\sigma_{\rm T}}{m_{\rm e} c^2} \int_{\rm LoS(\theta)} P_{\rm e} \; {\rm d}l,
       :label: compton
where :math:`y(\theta)` is the Compton parameter at the position :math:`\theta` on the sky, :math:`\sigma_{\rm T}` is the Thompson cross-section, and :math:`m_{\rm e} c^2` the electron rest-frame energy.

Through this dependence, we can infer the distribution of the electron pressure of the ICM from a mapping of the tSZ effect towards a cluster.
This is what ``panco2`` was designed to do, using forward modeling and MCMC sampling.
Here we give a brief description of the methodology; more details can be found in :cite:`keruzore_panco2_2022`.

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

The Compton parameter map obtained must be converted to surface brightness units in order to be comparable with observed data.
The unit used for NIKA2 maps is the Jy/beam.
The conversion from :math:`y` to Jy/beam depends on NIKA2 bandpasses, as well as its instrumental beam, and must be computed for each run separately.
In practice, ``panco2`` uses a *calibration factor*, for which the user must provide an input value and uncertainty, and which is treated as a parameter of the fit.
This allows one to propagate the uncertainty on the calibration of NIKA2 maps to our measurement of the pressure profile of galaxy clusters.


Point source contamination
--------------------------

SZ observations are very prone to contamination by point sources.
To take it into account, ``panco2`` can use the methodology described in :cite:`keruzore_exploiting_2020`, which consists in treating the fluxes of known point sources as parameters of the fit, constrained by a prior knowledge of each source's flux.
This allows to propagate the uncertainty in the point sources fluxes to ``panco2``'s results.

If asked, ``panco2`` will therefore add for each source a 2D gaussian function representing the NIKA2 instrumental beam, with an amplitude given by the flux of the source. 
This addition is performed before the convolution by the transfer function, as the point sources re just as affected by the filtering as the SZ signal.

Integrated signal
-----------------

As ``panco2`` was designed for the NIKA2 SZ Large Program (LPSZ, :cite:`mayet_cluster_2020`), it uses the knowledge of the integrated SZ flux of the cluster, which is always available for LPSZ clusters from the ACT and *Planck* surveys.
Therefore, along with the model map computation, a model integrated SZ signal is computed as

.. math:: Y_R = 4\pi\frac{\sigma_\mathrm{T}}{m_e c^2} \int_0^R r^2 P(r) \,\mathrm{d}r,
   :label: yinteg

where :math:`R` can either be :math:`R_{500}` or :math:`5R_{500}` depending on what is available. 
This integrated signal can be compared to the actual survey measurement as a way to constrain large-scale emission of the cluster.

Summary
-------

The parameters of the model used by panco can be summarized in a vector :math:`\vartheta` composed of:

- The parameters of the pressure profile: :math:`P_0,\,r_p,\,a,\,b,\,c` for a gNFW fit,
  :math:`P_i,\; i = 0 \cdots n_{\rm bins}` for a non-parametric fit;
- The "calibration coefficient" to convert Compton parameter measurements to Jy/beam,
- If asked, a zero-level can also be used as a free parameter to account for possible
  residual noise,
- If asked, a flux value for each known point source in the map.

From these parameters, a model map :math:`\mathcal{M}(\vartheta)` can be generated that can be directly compared to NIKA2 observations, as well as a value of spherically-integrated SZ signal :math:`Y`.

Pressure profile fitting
========================

``panco2`` aims at finding the probability distribution for the parameters of the chosen model
given the input data.  It does so by using Bayesian Monte Carlo Markov Chains (MCMC)
sampling: let :math:`D` be the input data and :math:`\vartheta` the set of parameters of
the model.  The probability for :math:`\theta` to accurately describe the data is given
by the Bayes theorem:

.. math:: P(\vartheta \,|\, D) = \frac{P(D \,|\, \vartheta) \, P(\vartheta)}{P(D)},
   :label: post

where :math:`P(\vartheta \,|\, D)` is called the *posterior distribution*, :math:`P(D
\,|\, \vartheta)` is the *likelihood function* comparing the model to the data,
:math:`P(\vartheta)` is the *prior distribution* encapsulating the user's prior knowledge
about the parameters, and :math:`P(D)` is the data evidence, here treated as a
normalization constant.

The likelihood function
-----------------------

``panco2`` uses a multivariate gaussian likelihood function to compare the model to data.
Starting from ``v1.1_dec20``, each data point (*i.e.* each pixel of the map) can either be considered independent or correlated to the others: for a
parameter set :math:`\vartheta`,


.. math:: \mathrm{log} \, \mathcal{L}(\vartheta)
	= \mathrm{log} \, P(D \, | \, \vartheta) = - \frac{1}{2}
	\left(D - \mathcal{M}(\vartheta)\right)^{\rm T} \Sigma^{-1} \left(D -
	\mathcal{M}(\vartheta)\right)
	  - \frac{1}{2} \left(\frac{Y_R^{\rm meas.} - Y_R(\vartheta)}{\Delta Y_R^{\rm
	  meas.}}\right)^2 - \Delta_{\rm mass}
    :label: loglike

where :math:`D` is the measured NIKA2 map, :math:`\Sigma` is the noise covariance matrix,
:math:`\mathcal{M}(\vartheta)` is the model map described in :numref:`Model Computation`,
:math:`Y_R(\vartheta)` is the integrated SZ signal computed from Eq.  :eq:`yinteg`, and
:math:`Y_R^{\rm meas.}` and :math:`\Delta Y_R^{\rm meas.}` are the measured integrated
SZ signal and its uncertainty, respectively.
The noise covariance matrix can be computed in ``panco2`` if the user provides a set of correlated noise realizations (as produced by ``SZ_IMCM``).
Otherwise, the noise is considered to be white and the pixels uncorrelated, simplifying Eq. :eq:`loglike` and greatly improving the computation time.


The prior distribution
----------------------

``panco2`` uses a prior distribution where all parameters are assumed uncorrelated.  Some
parameters accept a wide variety of priors, that we detail here.


Posterior distribution sampling
-------------------------------

The fit is performed by Monte Carlo Markov Chain (MCMC) sampling of the posterior
probability distribution of Eq.  :eq:`post`.  This section quickly reviews this
statistical technique and presents the specific implementation done in ``panco2``.


Starting point
^^^^^^^^^^^^^^

The starting point of the Markov chains in the parameter space is an input of any MCMC
analysis.  The sampling can either be started from a random position -- in which case
the sampler needs to find the optimal region -- or from an initial guess of the user.
For ``panco2``, we chose the latter, in order to speed up the process.

**In gNFW mode,** the parameters maximizing the posterior distribution of Eq.
:eq:`post` are found using the ``migrad`` algorithm of ``iMinuit``, the Python
implementation of the ``MINUIT`` suite :cite:`hans_dembinski_scikit-hepiminuit_2020`.
The MCMC is started at this position in the parameter space.

**In non-parametric mode,** the parameters are the pressure in each bin, and their
starting point are computed as the value of the universal pressure profile of
:cite:`arnaud_universal_2010` at each radius bin.

For other parameters (calibration coefficient, zero level, point source fluxes), the
starting point of each parameter is the maximum of its prior distribution.

Chains convergence
^^^^^^^^^^^^^^^^^^

One crucial step of MCMC analyses is to know at which point the sampling can be stopped.
To do so, the following test was implemented and is performed regularly:

#. Apply a burn-in cut, *i.e.* discard a portion of each chain at its beginning (the
   time needed to reach the "correct" part of the parameter space);

#. Compute the autocorrelation length :math:`l`, which is the number of steps :math:`n`
   a walker has to perform from a position :math:`\theta_i` so that :math:`\theta_{i+n}` is
   independent from :math:`\theta_i` (or, more simply put, the time it takes a walker to
   forget where it comes from);

#. The convergence of the chains is accepted if:

    - More than 2/3 of the chains have walked more than 40 times their autocorrelation
      length,

    - These chains pass the R-hat test of :cite:`gelman_inference_1992`:

	  .. math:: \hat{R} = \sqrt{\frac{V}{W}} < 1.02

   where :math:`V` measures the variance between all chains, and :math:`W` measures
   the average variance within one chain.

The whole process ensures that more than two thirds of the chains are long enough that
they can be used for inference, and that they are correctly mixed, *i.e.* that the
individual properties of each chains are similar to those of the whole sample.

Results exploitation
--------------------

Once the chains have reached convergence, they constitute a random sample for which the
probability distribution is the posterior distribution of Eq.  :eq:`post`.  These are
used to infer measurements of the physical properties of the ICM.

The pressure profile of the ICM is the property directly probed by our fit.  Its value
is given by the computation of our model for the set of parameters that maximize the
posterior distribution sampled in our fit.  Each set of parameters sampled in the MCMC
is then used to compute a pressure profile on a wide radius range.  The dispersion of
these profiles gives a measurement of the statistical error on the pressure for the
whole radial range considered.

In non-parametric mode, at the end of the MCMC, another regression can be performed to
fit a gNFW profile on the non-parametric pressure bins. This fit is difficult to perform,
as it requires to fit a model with highly correlated parameters on a small number of
data points. Sampling a posterior distribution is therefore tedious without tight priors.
Still, a module is implemented to allow the user to perform such a fit, using MCMC as
well as a maximum-likelihood approach is implemented, where the user may use ``MIGRAD``
to find the gNFW profile that best describe the pressure profile. In each case, the full
posterior distribution of ``panco2``'s first MCMC is used in the fit (*i.e.* the
correlations between the different bins are taken into account).