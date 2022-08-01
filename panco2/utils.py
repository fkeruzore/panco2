import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import astropy.units as u
from astropy.constants import sigma_T, c, m_e

sz_fact = (sigma_T / (m_e * c**2)).to(u.cm**3 / u.keV / u.kpc).value
del sigma_T, c, m_e


def gNFW(r, P0, rp, a, b, c):
    """
    gNFW pressure profile.

    Inputs :
    --------
    - r : radii (can be a scalar or a ndarray),
    - P0, rp, a, b, c : parameters

    Outputs :
    ---------
    - pressure evaluated at each radius (same shape as the input r)
    """

    x = r / rp
    return P0 / ((x**c) * (1.0 + x**a) ** ((b - c) / a))


# ---------------------------------------------------------------------- #


def d_gNFW_d_r(r, P0, rp, a, b, c):
    """
    Analytical derivative of the gNFW pressure profile

    Inputs :
    --------
    - r : radii (can be a scalar or a ndarray),
    - P0, rp, a, b, c : parameters

    Outputs :
    ---------
    - dp/dr evaluated at each radius (same shape as the input r)
    """

    x = r / rp
    return -P0 * c * x ** (-c) * (x**a + 1.0) ** (
        (-b + c) / a
    ) / r + P0 * x**a * x ** (-c) * (-b + c) * (x**a + 1.0) ** (
        (-b + c) / a
    ) / (
        r * (x**a + 1.0)
    )


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #


def map2prof(m_2d, r_2d, width=0):
    """
    Returns a radial profile and its confidence intervals from a
    2d map.

    Parameters
    ----------
    m_2d : np.ndarray, shape=(N, N)
        the map to be made into a profile.
    r_2d : np.ndarray, shape=(N, N)
        radii map, i.e. distances from center for each pixel
    width: float
        smoothing width, samw units as r_2d

    Returns
    -------
    r_1d: ndarray, shape=(N)
        radii
    m_1d: ndarray, shape=(N, 3)
        [16th, 50th, 84th] percentile for the profile at each radius
    """
    r_1d = np.unique(r_2d)
    m_1d = np.zeros((r_1d.size, 3))
    for i, r in enumerate(r_1d):
        is_ok = (r_2d >= r - width) & (r_2d <= r + width)
        which_m = m_2d[is_ok]
        m_1d[i, :] = np.percentile(which_m, [16.0, 50.0, 84.0])
    return r_1d, m_1d


# ---------------------------------------------------------------------- #


def adim(qty):
    """
    Removes the dimension of an adimensioned astropy quantity, e.g.
    >>> adim(2.0 * (u.m / u.cm))
    200.0

    Args:
        qty (astropy.units.Quantity): a dimensioned quantity.

    Returns:
        (float): the adimensioned quantity.
    """
    out = qty.decompose()
    if out.unit != u.dimensionless_unscaled:
        raise Exception(
            "Tried to simplify a dimensionned unit : " + str(qty.unit)
        )
    else:
        return out.value


# ---------------------------------------------------------------------- #


def interp_powerlaw(x, y, x_new, axis=0):
    """
    Interpolate/extrapolate a profile with a power-law by performing
    linear inter/extrapolation in the log-log space.

    Args:
        x (array): input x-axis.
        y (array): input `f(x)`.
        x_new (float or array): `x` value(s) at which to perform interpolation.

    Returns:
        (float or array): `f(x_new)`

    """

    w_nonzero = np.where(x > 0.0)

    log_x = np.log10(x[w_nonzero])
    log_y = np.log10(y[w_nonzero])

    interp = interp1d(
        log_x,
        log_y,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        axis=axis,
    )
    y_new = 10 ** interp(np.log10(x_new))

    return y_new


# ---------------------------------------------------------------------- #


class LogLogSpline(UnivariateSpline):
    """
    Performs a spline interpolation in the log-log plane.
    The interface is the same as `scipy.interpolate.UnivariateSpline`,
    but the interpolation is performed on the log10 of your data.

    Args:
        x (array): input `x`-axis.

        y (array): input `y`-axis

        **kwargs: kwargs to pass scipy.interpolate.UnivariateSpline

    """

    def __init__(self, x, y, **kwargs):
        super().__init__(np.log10(x), np.log10(y), **kwargs)

    def __call__(self, x):
        """
        Computes the interpolation at a given `x`.

        Args:
            x (array of float): the `x` value(s) at which to
                perform the interpolation

        Returns:
            (array or float) `f(x)`
        """
        return 10 ** super().__call__(np.log10(x))

    def differentiate(self, x, n=1):
        """
        Computes the `n`-th derivative of the spline in `x`.

        Args:
            x (array of float): the `x` value(s) at which to
                perform the interpolation
            n (int): degree of differentiation.

        Returns:
            (array or float) d`f`/d`x` `(x)`
        """
        deriv_spline = super().derivative(n=n)
        # return self.__call__(x) * deriv_spline(np.log10(x))
        return 1 / x * self.__call__(x) * deriv_spline(np.log10(x))


# ---------------------------------------------------------------------- #


def sph_integ_within(prof, r, axis=0):
    """
    Integration of a profile in spherical coordinates between
    min(r) and max(r)

    Args:
        prof (array): the profile to integrate along the x axis.
        r (array): corresponding radii vector.

    Returns:
        (float): :math:`4\pi \int_{min(r)}^{max(r)} r^2 prof(r) dr`

    """

    return 4.0 * np.pi * np.trapz(r**2 * prof, r, axis=axis)


# ---------------------------------------------------------------------- #


def cyl_integ_within(prof, r, axis=0):
    """
    Integration of a profile in cylindrical coordinates between
    min(r) and max(r)

    Args:
        prof (array): the profile to integrate along the x axis.
        r (array): corresponding radii vector.

    Returns:
        (float): :math:`2\pi \int_{0}^{max(r)} r prof(r) dr`

    """
    # log_r = np.log10(r)
    # log_prof = np.log10(prof)

    return 2.0 * np.pi * np.trapz(r * prof, r, axis=axis)


# ---------------------------------------------------------------------- #


def prof2map(prof_x, r_x, r_xy):
    """
    Compute a 2D map from a 1D profile through
    power law interpolation.

    Args:
        prof_x(array): the profile along the `x` axis.
        r_x (array):corresponding radii vector.
        r_xy array): radii map in the `(x, y)` plane.


    Returns:
        (array): Profile interpolated in the (x, y) plane
    """
    interp = interp1d(
        r_x, np.log10(prof_x), bounds_error=False, fill_value="extrapolate"
    )
    return 10 ** interp(r_xy)


# ---------------------------------------------------------------------- #


def ignore_astropy_warnings():
    from astropy.utils.exceptions import AstropyWarning
    import warnings

    warnings.simplefilter("ignore", category=AstropyWarning)
