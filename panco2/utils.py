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


# ---------------------------------------------------------------------- #


def get_planck_cmap():
    import matplotlib as mpl

    cmap = np.array(
        [
            [0, 0, 255],
            [0, 2, 255],
            [0, 5, 255],
            [0, 8, 255],
            [0, 10, 255],
            [0, 13, 255],
            [0, 16, 255],
            [0, 18, 255],
            [0, 21, 255],
            [0, 24, 255],
            [0, 26, 255],
            [0, 29, 255],
            [0, 32, 255],
            [0, 34, 255],
            [0, 37, 255],
            [0, 40, 255],
            [0, 42, 255],
            [0, 45, 255],
            [0, 48, 255],
            [0, 50, 255],
            [0, 53, 255],
            [0, 56, 255],
            [0, 58, 255],
            [0, 61, 255],
            [0, 64, 255],
            [0, 66, 255],
            [0, 69, 255],
            [0, 72, 255],
            [0, 74, 255],
            [0, 77, 255],
            [0, 80, 255],
            [0, 82, 255],
            [0, 85, 255],
            [0, 88, 255],
            [0, 90, 255],
            [0, 93, 255],
            [0, 96, 255],
            [0, 98, 255],
            [0, 101, 255],
            [0, 104, 255],
            [0, 106, 255],
            [0, 109, 255],
            [0, 112, 255],
            [0, 114, 255],
            [0, 117, 255],
            [0, 119, 255],
            [0, 122, 255],
            [0, 124, 255],
            [0, 127, 255],
            [0, 129, 255],
            [0, 132, 255],
            [0, 134, 255],
            [0, 137, 255],
            [0, 139, 255],
            [0, 142, 255],
            [0, 144, 255],
            [0, 147, 255],
            [0, 150, 255],
            [0, 152, 255],
            [0, 155, 255],
            [0, 157, 255],
            [0, 160, 255],
            [0, 162, 255],
            [0, 165, 255],
            [0, 167, 255],
            [0, 170, 255],
            [0, 172, 255],
            [0, 175, 255],
            [0, 177, 255],
            [0, 180, 255],
            [0, 182, 255],
            [0, 185, 255],
            [0, 188, 255],
            [0, 190, 255],
            [0, 193, 255],
            [0, 195, 255],
            [0, 198, 255],
            [0, 200, 255],
            [0, 203, 255],
            [0, 205, 255],
            [0, 208, 255],
            [0, 210, 255],
            [0, 213, 255],
            [0, 215, 255],
            [0, 218, 255],
            [0, 221, 255],
            [6, 221, 254],
            [12, 221, 253],
            [18, 222, 252],
            [24, 222, 251],
            [30, 222, 250],
            [36, 223, 249],
            [42, 223, 248],
            [48, 224, 247],
            [54, 224, 246],
            [60, 224, 245],
            [66, 225, 245],
            [72, 225, 244],
            [78, 225, 243],
            [85, 226, 242],
            [91, 226, 241],
            [97, 227, 240],
            [103, 227, 239],
            [109, 227, 238],
            [115, 228, 237],
            [121, 228, 236],
            [127, 229, 236],
            [133, 229, 235],
            [139, 229, 234],
            [145, 230, 233],
            [151, 230, 232],
            [157, 230, 231],
            [163, 231, 230],
            [170, 231, 229],
            [176, 232, 228],
            [182, 232, 227],
            [188, 232, 226],
            [194, 233, 226],
            [200, 233, 225],
            [206, 233, 224],
            [212, 234, 223],
            [218, 234, 222],
            [224, 235, 221],
            [230, 235, 220],
            [236, 235, 219],
            [242, 236, 218],
            [248, 236, 217],
            [255, 237, 217],
            [255, 235, 211],
            [255, 234, 206],
            [255, 233, 201],
            [255, 231, 196],
            [255, 230, 191],
            [255, 229, 186],
            [255, 227, 181],
            [255, 226, 176],
            [255, 225, 171],
            [255, 223, 166],
            [255, 222, 161],
            [255, 221, 156],
            [255, 219, 151],
            [255, 218, 146],
            [255, 217, 141],
            [255, 215, 136],
            [255, 214, 131],
            [255, 213, 126],
            [255, 211, 121],
            [255, 210, 116],
            [255, 209, 111],
            [255, 207, 105],
            [255, 206, 100],
            [255, 205, 95],
            [255, 203, 90],
            [255, 202, 85],
            [255, 201, 80],
            [255, 199, 75],
            [255, 198, 70],
            [255, 197, 65],
            [255, 195, 60],
            [255, 194, 55],
            [255, 193, 50],
            [255, 191, 45],
            [255, 190, 40],
            [255, 189, 35],
            [255, 187, 30],
            [255, 186, 25],
            [255, 185, 20],
            [255, 183, 15],
            [255, 182, 10],
            [255, 181, 5],
            [255, 180, 0],
            [255, 177, 0],
            [255, 175, 0],
            [255, 172, 0],
            [255, 170, 0],
            [255, 167, 0],
            [255, 165, 0],
            [255, 162, 0],
            [255, 160, 0],
            [255, 157, 0],
            [255, 155, 0],
            [255, 152, 0],
            [255, 150, 0],
            [255, 147, 0],
            [255, 145, 0],
            [255, 142, 0],
            [255, 140, 0],
            [255, 137, 0],
            [255, 135, 0],
            [255, 132, 0],
            [255, 130, 0],
            [255, 127, 0],
            [255, 125, 0],
            [255, 122, 0],
            [255, 120, 0],
            [255, 117, 0],
            [255, 115, 0],
            [255, 112, 0],
            [255, 110, 0],
            [255, 107, 0],
            [255, 105, 0],
            [255, 102, 0],
            [255, 100, 0],
            [255, 97, 0],
            [255, 95, 0],
            [255, 92, 0],
            [255, 90, 0],
            [255, 87, 0],
            [255, 85, 0],
            [255, 82, 0],
            [255, 80, 0],
            [255, 77, 0],
            [255, 75, 0],
            [251, 73, 0],
            [247, 71, 0],
            [244, 69, 0],
            [240, 68, 0],
            [236, 66, 0],
            [233, 64, 0],
            [229, 62, 0],
            [226, 61, 0],
            [222, 59, 0],
            [218, 57, 0],
            [215, 55, 0],
            [211, 54, 0],
            [208, 52, 0],
            [204, 50, 0],
            [200, 48, 0],
            [197, 47, 0],
            [193, 45, 0],
            [190, 43, 0],
            [186, 41, 0],
            [182, 40, 0],
            [179, 38, 0],
            [175, 36, 0],
            [172, 34, 0],
            [168, 33, 0],
            [164, 31, 0],
            [161, 29, 0],
            [157, 27, 0],
            [154, 26, 0],
            [150, 24, 0],
            [146, 22, 0],
            [143, 20, 0],
            [139, 19, 0],
            [136, 17, 0],
            [132, 15, 0],
            [128, 13, 0],
            [125, 12, 0],
            [121, 10, 0],
            [118, 8, 0],
            [114, 6, 0],
            [110, 5, 0],
            [107, 3, 0],
            [103, 1, 0],
            [100, 0, 0],
        ]
    )
    return mpl.colors.ListedColormap(cmap / 255.0)
