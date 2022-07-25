PANCO2's algorithm
******************

This section describes the step-by-step algorithm implemented in ``PANCO2`` to infer a measurement of the pressure profile and thermodynamical properties of the ICM from NIKA2 observations.

Physical modeling of the ICM
============================

The spherical symmetry of the ICM is assumed by ``PANCO2``.  Therefore, the pressure distribution is described by a pressure profile, :math:`P(r)`.
Two models can be used:

- A parametric, generalized Navarro-Frenk-White (gNFW) model :cite:`zhao_analytical_1996,nagai_effects_2007`: in this case, the pressure profile of the ICM is written as

    .. math:: P(r) = P_0 \left(\frac{r}{r_p}\right)^{-c}
                 \left[1 + \left(\frac{r}{r_p}\right)^a \right]^\frac{c-b}{a},
       :label: gnfw

  where the pressure profile is described by *five* parameters: a normalization :math:`P_0`, external and internal slopes :math:`b` and :math:`b`, a characteristic radius of transition between the two regimes, :math:`r_p`, and the smoothness of this transition :math:`a`.

- A non-parametric model, where the pressure profile of the ICM is written as

    .. math:: P(R_i < r < R_{i+1}) = P_i \left(\frac{r}{R_i}\right)^{-\alpha_i} .
       :label: nonparam

  where a radial binning covering the cluster's spatial extension is defined, and the value of the pressure profile at each radial bin is a parameter of the model.
  The value of the pressure between two bins is computed by a power law interpolation between the two closest bins.  The number of parameters of the model therefore depends on the binning chosen by the user.

Each model has its advantages and drawbacks.
The gNFW model is smooth (and therefore easily extrapolated), and widely used in the literature.
Nonetheless, performing non-parametric fits is strongly recommended for several reasons.
Non-parametric profiles offer the ability to detect features that may be smoothed in a gNFW profile (such as local overpressures).
But, as far as running time is concerned, the most important advantage of non-parametric fits over gNFW is the relatively weak correlation between the model parameters.
The degeneracy of the parameters in a gNFW profile is well known, and is the source of many technical complications when trying to fit a model on data.
As a consequence, the running time of ``PANCO2`` in non-parametric mode is much faster than in gNFW mode on the same map (:math:`\sim 15` minutes vs :math:`\sim 4` hours with the same resources).
It is therefore highly recommended to use non-parametric models.

*Conventions and notations*

The sky plane will be designed as the :math:`(x, y)` plane.  The :math:`z` direction will refer to the line of sight.
For each cluster, a characteristic radius :math:`R_{500}` can be defined, corresponding to the radius inside which the average density is 500 times higher than :math:`\rho_c(z)`, the critical density of the Universe at the cluster's redshift :math:`z`.
Quantities with a 500 subscript refer to properties of the cluster inside this radius.

Model computation
=================

Regardless of the user's choice of pressure profile model, ``PANCO2``'s model computation follows the same procedure to compute a model to be compared with the input data.

Compton parameter map computation
---------------------------------

The first step is to compute the Compton parameter profile :math:`y(r)` on a radial range that entirely covers the one covered by the input NIKA2 map.

**In gNFW mode**, the pressure is computed in the :math:`(x, z)` plane. 
The :math:`x` direction is binned as the center of the pixels in the input map, while the :math:`z` direction is binned with 500 log-spaced points from 1 pc to :math:`5R_{500}`.
The pressure is then integrated along the :math:`z` direction.

**In non-parametric mode**, the pressure profile is analytically integrated from 0 to infinity on the :math:`z` axis, following the work of :cite:`romero_multi-instrument_2018`.

In both cases, this procedure yields a Compton parameter profile as described by eq. :eq:`compton`, computed at the radii of each pixel from the center of the map to its edge.

This Compton parameter map is then interpolated at the radii corresponding to each pixel in the input map.
This approach is much shorter than computing the line-of-sight integration for each pixel of the map.

Filtered surface brightness map
-------------------------------

The Compton parameter map obtained must be converted to surface brightness units in order to be comparable with observed data.
The unit used for NIKA2 maps is the Jy/beam.
The conversion from :math:`y` to Jy/beam depends on NIKA2 bandpasses, as well as its instrumental beam, and must be computed for each run separately.
In practice, ``PANCO2`` uses a *calibration factor*, for which the user must provide an input value and uncertainty, and which is treated as a parameter of the fit.
This allows one to propagate the uncertainty on the calibration of NIKA2 maps to our measurement of the pressure profile of galaxy clusters.

The map obtained by multiplying the Compton parameter map with the calibration factor is then convolved with a gaussian kernel of :math:`{\rm FWHM} = 17.6 "` to account for the NIKA2 instrumental filtering, yielding a map in Jy/beam.

Finally, the pipeline filtering is taken into account by convolving the surface brightness map with a transfer function.
This function is an output of both versions of SZ-oriented NIKA2 pipeline wrappers, ``SZ_RDA`` and ``SZ_IMCM``.
It is computed by running a simulation through the pipeline and comparing the azimuthally-averaged power spectra of the input and output maps.
The resulting map has therefore undergone the same filtering than the input map during data processing, and the two can be compared.

Point source contamination
--------------------------

SZ observations are very prone to contamination by point sources.
To take it into account, ``PANCO2`` can use the methodology described in :cite:`keruzore_exploiting_2020`, which consists in treating the fluxes of known point sources as parameters of the fit, constrained by a prior knowledge of each source's flux.
This allows to propagate the uncertainty in the point sources fluxes to ``PANCO2``'s results.

If asked, ``PANCO2`` will therefore add for each source a 2D gaussian function representing
the NIKA2 instrumental beam, with an amplitude given by the flux of the source.  This
addition is performed before the convolution by the transfer function, as the point
sources re just as affected by the filtering as the SZ signal.

Integrated signal
-----------------

As ``PANCO2`` was designed for the NIKA2 SZ Large Program (LPSZ,
:cite:`mayet_cluster_2020`), it uses the knowledge of the integrated SZ flux of the
cluster, which is always available for LPSZ clusters from the ACT and *Planck* surveys.
Therefore, along with the model map computation, a model integrated SZ signal is
computed as

.. math:: Y_R = 4\pi\frac{\sigma_\mathrm{T}}{m_e c^2} \int_0^R r^2 P(r) \,\mathrm{d}r,
   :label: yinteg

where :math:`R` can either be :math:`R_{500}` or :math:`5R_{500}` depending on what is
available.  This integrated signal can be compared to the actual survey measurement as a
way to constrain large-scale emission of the cluster.

Summary
-------

The parameters of the model used by PANCO can be summarized in a vector
:math:`\vartheta` composed of:

- The parameters of the pressure profile: :math:`P_0,\,r_p,\,a,\,b,\,c` for a gNFW fit,
  :math:`P_i,\; i = 0 \cdots n_{\rm bins}` for a non-parametric fit;
- The "calibration coefficient" to convert Compton parameter measurements to Jy/beam,
- If asked, a zero-level can also be used as a free parameter to account for possible
  residual noise,
- If asked, a flux value for each known point source in the map.

From these parameters, a model map :math:`\mathcal{M}(\vartheta)` can be generated that
can be directly compared to NIKA2 observations, as well as a value of
spherically-integrated SZ signal :math:`Y`.


Model adjustment
================

``PANCO2`` aims at finding the probability distribution for the parameters of the chosen model
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

``PANCO2`` uses a multivariate gaussian likelihood function to compare the model to data.
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
The noise covariance matrix can be computed in ``PANCO2`` if the user provides a set of correlated noise realizations (as produced by ``SZ_IMCM``).
Otherwise, the noise is considered to be white and the pixels uncorrelated, simplifying Eq. :eq:`loglike` and greatly improving the computation time.

In gNFW mode, if X-ray data are available, a term :math:`\Delta_{\rm mass}` is added,
ensuring that the hydrostatic mass profile from Eq. :eq:`mhse` always increases with
radius:

.. math:: \Delta_{\rm mass} =
    \begin{cases}
	\; 0 \;\text{if}\; \mathrm{d}M / \mathrm{d}r \geq 0 \,\forall r, \\ \; -\infty
	\;\text{otherwise.}
    \end{cases}

The prior distribution
----------------------

``PANCO2`` uses a prior distribution where all parameters are assumed uncorrelated.  Some
parameters accept a wide variety of priors, that we detail here.

Pressure profile parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**In gNFW mode**, priors are usually needed because of the strong correlations between
parameters. They can be:

- Gaussian functions, where the user is free to either specify the mean and standard
  deviation of each distribution manually or to adopt one of the reference papers
  :cite:`arnaud_universal_2010,planck_collaboration_planck_2013`,
- Flat, where the user can specify a lower and higher limit for each
  parameter.

Note that in any case, negative parameters are not allowed.

**In non-parametric mode**, the only constraint imposed is that every pressure bin must
be positive.

Point source fluxes
^^^^^^^^^^^^^^^^^^^

If the map is contaminated by point sources, the user may adjust them along with the SZ
signal.  In that case, the prior on each point source flux can be:

- The probability distribution function of the flux evaluated via kernel density
  estimation from a previous sampling (*e.g.* from ``PSTools`` :cite:`f_keruzore_pstools_2019`),
- A Gaussian function, for which the user must specify a mean and standard deviation,
- A flat distribution, for which the user must specify a lower and higher bound.

Other parameters
^^^^^^^^^^^^^^^^

The prior on the "calibration factor" is a Gaussian function, the mean and standard
deviation of which must be specified by the user.

The prior on the zero level of the map is a Gaussian function with mean :math:`\mu = 0`
and :math:`\sigma = 5 \times 10^{-4} \; {\rm Jy/beam}`.

Posterior distribution sampling
-------------------------------

The fit is performed by Monte Carlo Markov Chain (MCMC) sampling of the posterior
probability distribution of Eq.  :eq:`post`.  This section quickly reviews this
statistical technique and presents the specific implementation done in ``PANCO2``.

Monte-Carlo Markov Chain sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MCMC is a statistical tool aiming at sampling a distribution, *i.e.* generating random
samples that follow the probability distribution function (PDF). The broad idea of MCMC
can be described by the Metropolis-Hastings algorithm:

#. Start from a given position in the parameter space, :math:`\theta`, and compute the
   value of the probability distribution of interest, :math:`{\rm PDF}(\theta)`;

#. Select a new position near :math:`\theta`, :math:`\theta'`, and compute
   :math:`{\rm PDF}(\theta')`;

#. If :math:`{\rm PDF}(\theta') \geq {\rm PDF}(\theta)`, accept :math:`\theta'` as your
   new position ; otherwise, accept it with a probability

    .. math:: P \propto \frac{\rm PDF (\theta')}{\rm PDF (\theta)};

#. Start again from step 2.

The set of accepted positions in the parameter space constitutes a random walk called a
Markov chain.  The algorithm can be run simultaneously by several independent chains,
providing an easy way to parallelize the sampling.  In ``PANCO2``, the sampler used is the
affine-invariant sampler implemented in the ``emcee`` Python library
:cite:`foreman-mackey_emcee_2019`, where the parameter space is explored by "walkers",
the key difference being that walkers are not independent and communicate with each
other during the sampling.

Starting point
^^^^^^^^^^^^^^

The starting point of the Markov chains in the parameter space is an input of any MCMC
analysis.  The sampling can either be started from a random position -- in which case
the sampler needs to find the optimal region -- or from an initial guess of the user.
For ``PANCO2``, we chose the latter, in order to speed up the process.

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
====================

Once the chains have reached convergence, they constitute a random sample for which the
probability distribution is the posterior distribution of Eq.  :eq:`post`.  These are
used to infer measurements of the physical properties of the ICM.

The ICM pressure profile
------------------------

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
posterior distribution of ``PANCO2``'s first MCMC is used in the fit (*i.e.* the
correlations between the different bins are taken into account).

Other thermodynamical profiles
------------------------------

If X-ray data are available for the cluster, ``PANCO2`` will also combine them with the
resulting pressure profile to infer more measurements of thermodynamical properties.
Namely, the pressure profile from NIKA2 :math:`P(r)` will be combined with the density
profile from X-ray data :math:`n(r)` to compute the radial profiles of the ICM
temperature :math:`k_{\rm B}T(r) = P(r) / n(r)`, entropy :math:`K(r) = P(r) \,
n^{-5/3}(r)`, hydrostatic mass through the equation of hydrostatic equilibrium
:eq:`mhse`.  The statistical error on each profile are computed the same way as for the
pressure profile (see :numref:`The ICM pressure profile`).

Integrated quantities
---------------------

One of the goals of the NIKA2 LPSZ is to compute the scaling relation between
:math:`Y_{500}` and :math:`M_{500}`, namely the integrated SZ signal and mass contained
in a radius :math:`R_{500}`.  These quantities are computed by ``PANCO2``.

If X-ray data are available, a measurement of :math:`R_{500}` can be computed by finding
the radius inside which the average density contrast is 500, *i.e.* by solving

.. math:: \delta_c (R_{500})
	= \frac{M_{\rm HSE}(R_{500})}{\rho_c(z) \times \frac{4}{3} \pi R_{500}^3} = 500

where :math:`\rho_c(z)` is the critical density of the Universe at the redshift of the
cluster :math:`z`.  Computing it for all sets of parameter sampled by our MCMC gives the
probability distribution of :math:`R_{500}`.  Similarly, the integrated SZ signal
:math:`Y_{500}` and hydrostatic mass :math:`M_{500}` can be computed for all the sampled
pressure profiles.

Integrated quantities for non-parametric fits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As seen in Eq.  :eq:`mhse`, the hydrostatic mass profile is proportional to the
derivative of the pressure profile with respect to radius.  A consequence is that for
non-parametric profiles, which are not smooth functions of the radius, mass profiles can
present undesirable features such as discontinuities.  This can potentially have strong
effects on the measurement of :math:`R_{500}` and :math:`M_{500}`.

We bypass this problem by using spline interpolations.  The samples of the posterior
distribution (*i.e.* the value of the pressure bins sampled during the MCMC) are each
interpolated in the log-log plane using a spline.  The derivative of the pressure needed
to evaluate the hydrostatic mass is then computed as the derivative of the spline.  This
smoothes the pressure profile, making its derivation more stable.

Error bars on integrated quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since :math:`Y_{500}` and :math:`M_{500}` are both computed inside a radius
:math:`R_{500}`, the correlation between the three variables is obviously important.  A
consequence is that the values can be evaluated in two distinct ways:

* Each sampled pressure profile can be used to compute a value of :math:`R_{500}`,
  inside which we compute the integrated quantities :math:`Y_{500}` and :math:`M_{500}` ;

* The best-fitting pressure profile cam be used to compute a fixed value of
  :math:`R_{500}`, inside which we compute the integrated quantities :math:`Y_{500}` and
  :math:`M_{500}` for each sampled profile.

The latter leads to significantly smaller error bars on the integrated quantities, as it
does not allow to propagate the uncertainty on :math:`R_{500}` to the integrated
quantities.  Both ways are implemented in ``PANCO2``, although by default the former
approach will be used.
