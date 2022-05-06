import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from . import utils


class Cluster:
    """
    Defines a cluster and its basic properties.

    Attributes
    ----------
        P_500 :
        R_500 :
        theta_500 :
        M_500 :
        cosmo :
        z :
        E :
        A10_params :
        sz_fact :
        Y_500_arcmin2 :
        d_a :
        Y_500_kpc2 :
        dens_crit :

    """

    def __init__(self, z, M_500):
        """

        Parameters
        ----------
        z : float
            Cluster's redshift
        M_500 : float [Msun]
            Cluster's mass within R_500.
        """

        # Cosmology related parameters
        self.cosmo = FlatLambdaCDM(70.0 * u.Unit("km s-1 Mpc-1"), 0.3)
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
