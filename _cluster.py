import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import _utils


class Cluster:
    """
    Define a galaxy cluster and its properties.

    Args:
        z (float):
            redshift.
        Y_500 (astropy.units.Quantity):
            Integrated Compton parameter.
        err_Y_500 (astropy.units.Quantity):
            Optional, error on Y_500.
        R_500 (astropy.Units.Quantity):
            Optional, default from A10 scaling relation
        theta_500 (astropy.Units.Quantity):
            Optional, default from A10 scaling relation)

    Notes:
        This class is a standalone that does not depend on any data.
        You can easily play around with it to check if the parameters behave
        as you intend.
    """

    def __init__(self, z, Y_500, err_Y_500=None, R_500=None, theta_500=None):

        """
        Cosmology related parameters
        """
        self.cosmo = FlatLambdaCDM(70.0 * u.Unit("km s-1 Mpc-1"), 0.3)
        self.z = z
        self.E = self.cosmo.efunc(z)
        self.dens_crit = self.cosmo.critical_density(z).to("g/cm3")
        self.d_a = self.cosmo.angular_diameter_distance(z).to("kpc")
        h_70 = self.cosmo.h / 0.7
        self.sz_fact = (
            (const.sigma_T / (const.m_e * const.c ** 2))
            .to(u.cm ** 3 / u.keV / u.kpc)
            .value
        )

        """
        Prior information on the ICM and universal quantities
        """
        self.Y_500_arcmin2 = Y_500.value
        if err_Y_500 is None:
            self.err_Y_500_arcmin2 = 0.3 * self.Y_500_arcmin2
        else:
            self.err_Y_500_arcmin2 = err_Y_500.value

        # Y500 in Mpc2 to compute M500 with equation (20) of A10
        self.Y_500 = (Y_500.to("rad2") * (self.d_a.to("Mpc")) ** 2) / u.rad ** 2
        self.M_500 = (
            3.0e14
            * const.M_sun
            / h_70
            * (
                10 ** 4.739
                * self.E ** (-2.0 / 3.0)
                * h_70 ** 2.5
                * self.Y_500.to("Mpc2").value
            )
            ** (1.0 / 1.79)
        ).to("Msun")

        # Y500 in kpc2
        self.Y_500_kpc2 = (self.Y_500.to("kpc2")).value
        if err_Y_500 is not None:
            self.err_Y_500_kpc2 = (
                err_Y_500.to("rad2") * (self.d_a.to("kpc") ** 2)
            ).value
        else:
            self.err_Y_500_kpc2 = 0.3 * self.Y_500_kpc2

        self.P_500 = (
            1.65e-3
            * (self.E) ** (8.0 / 3)
            * ((self.M_500 * h_70) / ((3.0e14) * const.M_sun)) ** (2.0 / 3.0 + 0.12)
            * (h_70) ** 2
            * u.keV
            / u.cm ** 3
        ).to(
            "keV cm-3"
        )  # equation (13) of A10

        if (R_500 is None) and (theta_500 is None):
            self.R_500 = (3 * self.M_500 / (4 * np.pi * 500 * self.dens_crit)) ** (
                1.0 / 3.0
            )
            self.theta_500 = _utils.adim(self.R_500 / self.d_a) * u.rad

        elif (R_500 is None) and (theta_500 is not None):
            self.theta_500 = theta_500
            self.R_500 = self.d_a.to("kpc") * np.tan(self.theta_500.to("rad"))

        elif (R_500 is not None) and (theta_500 is None):
            self.R_500 = R_500
            self.theta_500 = _utils.adim(self.R_500 / self.d_a) * u.rad

        else:
            self.R_500 = R_500
            self.theta_500 = theta_500

        self.R_500 = self.R_500.to("kpc")
        self.R_500_kpc = self.R_500.value
        self.theta_500 = self.theta_500.to("arcmin")
        self.theta_500_arcmin = self.theta_500.value

        self.A10_params = np.array(
            [8.403 * self.P_500.value, self.R_500.value / 1.177, 1.0510, 5.4905, 0.3081]
        )

    def angle_distances_conversion(self, to_convert):
        """
        Converts an angular size to physical distance, or reciprocally.

        Args:
            to_convert (astropy.units.Quantity): an angle or distance
                value.

        Returns:
            (astropy.units.Quantity) the input converted to distance
                or angle.
        """
        input_unit = to_convert.unit
        if input_unit.is_equivalent(u.rad):  # input is angle
            return np.tan(to_convert.to("rad").value) * self.d_a
        elif input_unit.is_equivalent(u.kpc):  # input is angle
            return np.arctan((to_convert.to("kpc") / self.d_a).value) * u.rad
