Forward modeling of thermal SZ maps
***********************************

This section describes the step-by-step algorithm implemented in ``panco2`` to infer a measurement of the pressure profile and thermodynamic properties of the ICM from a tSZ map.

Physical modeling of the ICM
============================

``panco2`` assumes spherical symmetry of the ICM pressure distribution, which is described by a pressure profile, :math:`P(r)`.
It uses a radially-binned model, where the pressure profile of the ICM is written as

    .. math:: P(R_i < r < R_{i+1}) = P_i \left(\frac{r}{R_i}\right)^{-\alpha_i},
       :label: nonparam

where a radial binning covering the cluster's spatial extension is defined, and the value of the pressure profile at each radial bin is a parameter of the model.
The value of the pressure between two bins is computed by a power law interpolation between the two closest bins.  The number of parameters of the model therefore depends on the binning chosen by the user.

*Conventions and notations*

The sky plane will be designed as the :math:`(x, y)` plane.  The :math:`z` direction will refer to the line of sight.


Compton parameter map computation
=================================

The first step is to compute the Compton parameter profile :math:`y(r)` on a radial range that entirely covers the one covered by the input NIKA2 map.

The pressure profile is analytically integrated from 0 to infinity on the :math:`z` axis, following the work of :cite:`romero_multi-instrument_2018`.

In both cases, this procedure yields a Compton parameter profile as described by eq. :eq:`compton`, computed at the radii of each pixel from the center of the map to its edge.

This Compton parameter profile is then interpolated at the radii corresponding to each pixel in the input map.
This approach is much shorter than computing the line-of-sight integration for each pixel of the map.

Filtered surface brightness map
===============================

The Compton parameter map obtained must be converted to surface brightness units in order to be comparable with observed data.
The unit used for NIKA2 maps is the Jy/beam.
The conversion from :math:`y` to Jy/beam depends on NIKA2 bandpasses, as well as its instrumental beam, and must be computed for each run separately.
In practice, ``panco2`` uses a *calibration factor*, for which the user must provide an input value and uncertainty, and which is treated as a parameter of the fit.
This allows one to propagate the uncertainty on the calibration of NIKA2 maps to our measurement of the pressure profile of galaxy clusters.

The map obtained by multiplying the Compton parameter map with the calibration factor is then convolved with a gaussian kernel of :math:`{\rm FWHM} = 17.6 "` to account for the NIKA2 instrumental filtering, yielding a map in Jy/beam.

Finally, the pipeline filtering is taken into account by convolving the surface brightness map with a transfer function.
This function is an output of both versions of SZ-oriented NIKA2 pipeline wrappers, ``SZ_RDA`` and ``SZ_IMCM``.
It is computed by running a simulation through the pipeline and comparing the azimuthally-averaged power spectra of the input and output maps.
The resulting map has therefore undergone the same filtering than the input map during data processing, and the two can be compared.

Possible additional components
==============================

Point source contamination
--------------------------

SZ observations are very prone to contamination by point sources.
To take it into account, ``panco2`` can use the methodology described in :cite:`keruzore_exploiting_2020`, which consists in treating the fluxes of known point sources as parameters of the fit, constrained by a prior knowledge of each source's flux.
This allows to propagate the uncertainty in the point sources fluxes to ``panco2``'s results.

If asked, ``panco2`` will therefore add for each source a 2D gaussian function representing
the NIKA2 instrumental beam, with an amplitude given by the flux of the source.  This
addition is performed before the convolution by the transfer function, as the point
sources re just as affected by the filtering as the SZ signal.

Integrated signal
-----------------

As ``panco2`` was designed for the NIKA2 SZ Large Program (LPSZ,
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

The parameters of the model used by panco can be summarized in a vector
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