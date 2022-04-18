import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import _utils


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

    def __init__(self, z, Y_500=None, M_500=None):
        """

        Parameters
        ----------
        z : float
            Cluster's redshift
        Y_500 : float [arcmin2]
            Cluster's integrated Compton parameter within R_500.
        M_500 : float [Msun]
            Cluster's mass within R_500.

        Raises
        ------
        Exception : if neither Y_500 nor M_500 are specified.

        """

        # Cosmology related parameters
        self.cosmo = FlatLambdaCDM(70.0 * u.Unit("km s-1 Mpc-1"), 0.3)
        self.z = z
        self.E = self.cosmo.efunc(z)
        self.dens_crit = self.cosmo.critical_density(z).to("g/cm3")
        self.d_a = self.cosmo.angular_diameter_distance(z).to("kpc")
        h_70 = self.cosmo.h / 0.7
        self.sz_fact = (
            (const.sigma_T / (const.m_e * const.c**2))
            .to(u.cm**3 / u.keV / u.kpc)
            .value
        )

        # Prior information on characteristic quantities
        if (M_500 is None) and (Y_500 is None):
            raise Exception(
                "Need either Y_500 or M_500 to initialize "
                + "`cluster` object"
            )
        if Y_500 is not None:
            self.Y_500_arcmin2 = Y_500
            Y_500_Mpc2 = (
                Y_500.to("rad2") * (self.d_a.to("Mpc")) ** 2
            ) / u.rad**2
            self.Y_500_kpc2 = Y_500_Mpc2.to("kpc2").value
            if M_500 is None:
                self.M_500 = (
                    3.0e14
                    / h_70
                    * (
                        10**4.739
                        * self.E ** (-2.0 / 3.0)
                        * h_70**2.5
                        * Y_500_Mpc2
                    )
                    ** (1.0 / 1.79)
                )  # Msun
        self.M_500 = M_500
        R_500 = (3 * self.M_500 / (4 * np.pi * 500 * self.dens_crit)) ** (
            1.0 / 3.0
        )
        self.R_500 = R_500.to("kpc").value
        self.theta_500 = np.arctan(R_500 / self.d_a.value) * u.rad.to("arcmin")

        # Arnaud et al. (2010) pressure profile parameters
        self.P_500 = (
            1.65e-3
            * (self.E) ** (8.0 / 3)
            * ((self.M_500 * h_70) / ((3.0e14) * const.M_sun))
            ** (2.0 / 3.0 + 0.12)
            * (h_70) ** 2
            * u.keV
            / u.cm**3
        ).to(
            "keV cm-3"
        )  # equation (13) of A10

        self.A10_params = np.array(
            [
                8.403 * self.P_500.value,
                self.R_500.value / 1.177,
                1.0510,
                5.4905,
                0.3081,
            ]
        )
