import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


class Cluster:
    """
    Defines a cluster and its basic properties.

    Attributes
    ----------
    P_500 : float
        Normalized pressure, corresponds to the first two terms of
        eq. 13 in Arnaud et al. 2010 (with a slightly different
        notation) [keV cm-3]
    R_500 : float
        Radius around the cluster center within which the mean density
        is 500 times the critical density of the Universe at the
        redshift of the cluster [kpc]
    theta_500 : float
        Angle on the sky subtended by R_500 [arcmin]
    M_500 : float
        Cluster mass contained within R_500 [Msun]
    cosmo : astropy.cosmology.Cosmology
        Cosmology assumed
    z : float
        Cluster redshift
    E : float
        Efunc H(z)/H_0 at the redshift of the cluster
    A10_params : list
        [P_0*P_500], r_p, a, b, c] for this cluster if it has an A10
        universal pressure profile (eqs 12-23 of Arnaud et al. 2010)
    sz_fact : float
        (sigma_T / (m_e c^2)) in [keV-1 cm3 kpc-1].
        Multiply with a pressure distribution in [keV cm-3] integrated
        along a line of sight in [kpc] to get a dimensionless
        Compton y
    Y_500_arcmin2 : float
        Spherically integrated pressure profile within R_500 [arcmin2]
    d_a : float
        Angular diameter distance to the cluster [kpc]
    Y_500_kpc2 : float
        Spherically integrated pressure profile within R_500 [kpc2]
    dens_crit : float
        Critical density of the Universe at the redshift of the cluster
        [Msun kpc-3]

    """

    def __init__(self, z, M_500, cosmo=FlatLambdaCDM(70.0, 0.3)):
        """
        Parameters
        ----------
        z : float
            Cluster's redshift
        M_500 : float [Msun]
            Cluster's mass within R_500.
        cosmo : astropy.cosmology.Cosmology, optional
            The cosmology to assume for distance computations.
            Defaults to flat LCDM with h=0.7, Om0=0.3.
        """

        # Cosmology related parameters
        self.cosmo = cosmo
        self.z = z
        self.E = self.cosmo.efunc(z)
        self.dens_crit = self.cosmo.critical_density(z).to("Msun kpc-3").value
        self.d_a = self.cosmo.angular_diameter_distance(z).to("kpc").value
        h_70 = self.cosmo.h / 0.7
        self.sz_fact = (
            (const.sigma_T / (const.m_e * const.c**2))
            .to(u.cm**3 / u.keV / u.kpc)
            .value
        )

        # Prior information on characteristic quantities
        self.M_500 = M_500
        self.R_500 = (3 * M_500 / (4 * np.pi * 500 * self.dens_crit)) ** (
            1.0 / 3.0
        )
        self.theta_500 = np.arctan(self.R_500 / self.d_a)
        self.theta_500 *= u.rad.to("arcmin")

        # Arnaud et al. (2010) pressure profile parameters
        self.P_500 = (
            1.65e-3
            * self.E ** (8.0 / 3)
            * ((self.M_500 * h_70 / 3e14) ** (2.0 / 3.0 + 0.12))
            * (h_70) ** 2
        )  # equation (13) of A10, slightly modified notation

        self.A10_params = np.array(  # equation (12) of A10
            [
                8.403 * self.P_500,
                self.R_500 / 1.177,
                1.0510,
                5.4905,
                0.3081,
            ]
        )

    def arcsec2kpc(self, angle):
        angle *= u.arcsec.to("rad")
        dist = np.tan(angle) * self.d_a
        return dist

    def kpc2arcsec(self, dist):
        angle = np.arctan(dist / self.d_a)
        return angle * u.rad.to("arcsec")
