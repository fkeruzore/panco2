The physics probed by NIKA2 observations
========================================

The Sunyaev-Zeldovich (SZ) effect
---------------------------------

The NIKA2 camera is able to map the ICM of galaxy clusters through the SZ effect, which is the Compton scattering of low-energy cosmic microwave background (CMB) photons on hot ICM electrons.
This effect creates a spectral distorsion of the CMB: interacting photons gain energy, causing more photons to be on the high-energy part of the spectrum.
This distorsion is observationally detected as a deficit in the CMB surface brightness at frequencies lower than 217 GHz, and an excess at higher frequencies.

The amplitude of this distorsion is given by the *Compton parameter y*, which is proportional to the electron pressure of the ICM :math:`P_e` integrated along the line of sight (LoS):

.. math:: y = \frac{\sigma_\mathrm{T}}{m_e c^2} \int_\mathrm{LoS} P_e \,\mathrm{d}l,
   :label: compton

where :math:`\sigma_\mathrm{T}` is the Thompson scattering cross-section, and :math:`m_e` is the electron mass.

The conversion from Compton parameter and surface brightness units (*i.e.* the direct observable) depends on the shape of the SZ spectrum, as well as on properties of the instrument used to observe clusters, and will be discussed in :numref:`Filtered surface brightness map`.

The SZ observable is therefore directly linked to the pressure distribution in the ICM.
Assuming spherical symmetry of the cluster, this distribution can be described as a pressure profile :math:`P_e(r)`, and measured from SZ observations.

Hydrostatic mass and X-ray measurements
---------------------------------------

One of the quantities needed for cluster cosmology is the mass of each galaxy cluster in the survey.
There are several ways to measure the mass of a cluster, among which is the *hydrostatic mass*.
Assuming the hydrostatic equilibrium, the mass enclosed in a sphere of radius :math:`r` can be written as

.. math:: M_\mathrm{HSE}(r) = -\frac{1}{\mu m_p G} \frac{r^2}{n(r)} \frac{\mathrm{d}P}{\mathrm{d} r},
    :label: mhse

where :math:`\mu` is the gas mean molecular weight, :math:`m_p` is the proton mass, :math:`G` is the gravitational constant, and :math:`n_e` and :math:`P_e` are the electron density and pressure inside the sphere.
The mass of a cluster can therefore be probed by the thermodynamical properties of the ICM.

X-ray observations of galaxy clusters also provide valuable information on the thermodynamic properties of their ICM.
The X-ray surface brightness, :math:`S_\mathrm{X}`, is expressed as

.. math:: S_\mathrm{X} = \frac{1}{4 \pi (1+z)^4} \int_\mathrm{LoS} n_e^2 \, \Lambda(Z,\,T_e) \,\mathrm{d}l,
   :label: xray

where :math:`n_e` is the electron density of the ICM, and :math:`\Lambda` is the cooling function, depending on the ICM metallicity :math:`Z` and electron temperature :math:`T_e`.
Similarly to SZ observations, X-rays can be used to reconstruct the electron density of the ICM, as well as its temperature.
The combination of SZ and X-ray observations can therefore be used to measure the masses of clusters.
This can be done in ``PANCO2``, provided X-ray data are available.
