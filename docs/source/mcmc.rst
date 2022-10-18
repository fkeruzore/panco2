Pressure profile fitting
************************

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
=======================

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
======================

``panco2`` uses a prior distribution where all parameters are assumed uncorrelated.  Some
parameters accept a wide variety of priors, that we detail here.


Posterior distribution sampling
===============================

The fit is performed by Monte Carlo Markov Chain (MCMC) sampling of the posterior
probability distribution of Eq.  :eq:`post`.  This section quickly reviews this
statistical technique and presents the specific implementation done in ``panco2``.


Starting point
--------------

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
------------------

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
====================

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