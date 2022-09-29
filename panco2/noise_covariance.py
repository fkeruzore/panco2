import numpy as np
from scipy.interpolate import interp1d
from scipy import linalg
from sklearn import covariance


def powspec(in_map, pix_size, n_bins=None):
    """
    Power spectrum of a map in the flat-sky approximation.

    Parameters
    ----------
    in_map : ndarray
        The input map.
    pix_size : float
        The size of one pixel in the map [arcsec]
    n_bins : int, optional
        Number of bins to use in k.
        If None (default), takes 0.25 times the number of
        pixels on the side of the map.

    Returns
    -------
    tuple
        [0] the power spectrum [1] the angular scales

    Notes
    =====
    The convention used for `k` is the same as the `numpy` one,
    i.e. the largest 1D mode is 1/(pixel size).
    """
    nx, ny = in_map.shape
    if n_bins is None:
        n_bins = int(np.min([nx, ny]) / 4)
    n_bins = int(n_bins)

    kx, ky = np.fft.fftfreq(nx, d=pix_size), np.fft.fftfreq(ny, d=pix_size)
    k = np.hypot(*np.meshgrid(kx, ky, indexing="ij"))

    ft_in_map = np.fft.fft2(in_map, norm="ortho")
    pk_in_map = np.real(ft_in_map * np.conj(ft_in_map))

    pk, k_edges = np.histogram(k, bins=n_bins, range=None, weights=pk_in_map)
    norm, _ = np.histogram(k, bins=n_bins, range=None)
    with np.errstate(invalid="ignore"):
        pk /= norm

    k_bins = k_edges[:-1] + np.ediff1d(k_edges)
    msk = np.ones_like(k_bins, dtype=bool)  # k_bins < (0.5 / pix_size)
    return pk[msk], k_bins[msk]


def powspec_to_maps(k, pk, nx, ny, pix_size, n_maps):
    """
    Creates a random map realization from a power spectrum.

    Parameters
    ----------
    k : ndarray
        Angular scales
    pk : ndarray
        Power spectrum values
    nx : int
        Number of pixels on the `x` dimension of the map
    ny : int
        Number of pixels on the `y` dimension of the map
    pix_size : float
        The size of one pixel in the map [arcsec]
    n_maps : int
        Number of maps to be created

    Returns
    -------
    ndarray
        The generated random map realizations,
        shape=(n_maps, nx, ny)

    Notes
    =====
    The convention used for `k` is the same as the `numpy` one,
    i.e. the largest 1D mode is 1/(pixel size).
    """
    sqpk = np.sqrt(pk)

    white_maps = np.random.normal(0.0, 1.0, (int(n_maps), nx, ny))
    k_filt = np.hypot(
        *np.meshgrid(
            np.fft.fftfreq(nx, d=pix_size), np.fft.fftfreq(ny, d=pix_size)
        )
    )
    filt_fct = interp1d(
        k, np.log10(sqpk), bounds_error=False, fill_value="extrapolate"
    )
    filt = 10 ** filt_fct(k_filt)
    ft_colored_maps = (
        np.fft.fft2(white_maps, axes=(-2, -1), norm="ortho")
        * filt[np.newaxis, :, :]
    )
    colored_maps = np.fft.ifft2(ft_colored_maps, axes=(-2, -1), norm="ortho")
    return np.real(colored_maps)


def make_maps_with_same_pk(in_map, pix_size, n_maps, n_bins=None):
    """
    Generates a number of random maps with the same power spectrum
    as an input map.

    Parameters
    ----------
    in_map : ndarray
        The input map to get the power spectrum from.
    pix_size : float
        The size of one pixel in the map [arcsec]
    n_maps : int
        Number of maps to be created
    n_bins : int, optional
        Number of bins to use in k.
        If None (default), takes 0.25 times the number of
        pixels on the side of the map.

    Returns
    -------
    ndarray
        The generated random map realizations,
        shape=(n_maps, nx, ny)
    """
    nx, ny = in_map.shape
    if n_bins is None:
        n_bins = int(np.min([nx, ny]) / 4)
    pk, k = powspec(in_map, pix_size, n_bins)
    maps = powspec_to_maps(k, pk, nx, ny, pix_size, int(n_maps))
    return maps


def check_inversion(m, invm):
    """
    Asserts that two matrices are the inverse of one another by
    checking that C @ C^-1 = I.

    Parameters
    ----------
    m : ndarray
        2D square matrix
    invm : ndarray
        2D square matrix that you suppose is the inverse of m.
    """
    assert np.allclose(m @ invm, np.eye(m.shape[0])), (
        "Covariance matrix inversion failed: C @ C^-1 != I(n_pix). "
        + "Please try another way to compute the covariance."
    )


def covmat_from_noise_maps(noise_maps, method="lw"):
    """
    Computes the pixel noise covariance matrix and its inverse
    from random noise realizations.

    Parameters
    ----------
    noise_maps : ndarray
        The noise realizations, shape=(n_maps, nx, nx)
    method : str, optional
        How to compute the covariance.
        Must be either "np", in which case the covariance
        is the sample covariance computed via `numpy.cov`,
        or "lw" for the Ledoit-Wolf covariance estimated
        using `sklearn.covariance.ledoit_wolf()` (default).

    Returns
    -------
    ndarray
        The covariance matrix, shape=(nx^2, nx^2)
    ndarray
        The inverted covariance matrix, shape=(nx^2, nx^2)

    Raises
    ------
    Exception
        If `method` is not set to "np" or "lw"
    """
    noise_vecs = noise_maps.reshape(noise_maps.shape[0], -1)
    if method == "lw":
        cov, shrink = covariance.ledoit_wolf(noise_vecs)
    elif method in ["np", "numpy"]:
        cov = np.cov(noise_vecs, rowvar=False)
    else:
        raise Exception(
            f"Could not understand covariance method {method}. "
            + "Please refer to documentation."
        )
    inv_cov = linalg.pinv(cov)
    check_inversion(cov, inv_cov)
    return cov, inv_cov


def covmat_from_powspec(
    ell, C_ell, n_pix, pix_size, n_maps=1000, method="lw", return_maps=False
):
    """
    Computes the pixel noise covariance matrix and its inverse
    from a noise power spectrum by generating many random
    maps with the input power spectrum.


    Parameters
    ----------
    ell : ndarray
        Multipole numbers
    C_ell : ndarray
        Power spectrum values
    n_pix : int
        Number of pixels on each dimension of the map
        (i.e. the map is n_pix * n_pix)
    pix_size : float
        The size of one pixel in the map [arcsec]
    n_maps : int, optional
        Number of maps to be created. Defaults to 1000.
    method : str, optional
        How to compute the covariance.
        Must be either "np", in which case the covariance
        is the sample covariance computed via `numpy.cov`,
        or "lw" for the Ledoit-Wolf covariance estimated
        using `sklearn.covariance.ledoit_wolf()` (default).
    return_maps: bool, optional
        If True, this function also returns the noise realizations
        used to compute the covariance, default is False.

    Returns
    -------
    ndarray
        The covariance matrix, shape=(nx^2, nx^2)
    ndarray
        The inverted covariance matrix, shape=(nx^2, nx^2)
    ndarray
        If `return_maps` is True, the noise maps used to compute the
        covariance, shape=(n_maps, nx, nx)
    """
    k = ell / (180.0 * 3600.0)  # arcsec-1
    noise_maps = powspec_to_maps(k, C_ell, n_pix, n_pix, pix_size, int(n_maps))
    cov, inv_cov = covmat_from_noise_maps(noise_maps, method=method)
    if return_maps:
        return cov, inv_cov, noise_maps
    else:
        return cov, inv_cov


def covmats_from_noise_map(
    noise_map, pix_size, n_maps=1000, method="lw", return_maps=False
):
    """
    Computes the pixel noise covariance matrix and its inverse
    from a noise map by generating many random realizations
    with the same power spectrum.

    Parameters
    ----------
    noise_map : ndarray
        The noise realizations, shape=(nx, nx)
    pix_size : float
        The size of one pixel in the map [arcsec]
    n_maps : int, optional
        Number of maps to be created. Defaults to 1000.
    method : str, optional
        How to compute the covariance.
        Must be either "np", in which case the covariance
        is the sample covariance computed via `numpy.cov`,
        or "lw" for the Ledoit-Wolf covariance estimated
        using `sklearn.covariance.ledoit_wolf()` (default).
    return_maps: bool, optional
        If True, this function also returns the noise realizations
        used to compute the covariance, default is False.

    Returns
    -------
    ndarray
        The covariance matrix, shape=(nx^2, nx^2)
    ndarray
        The inverted covariance matrix, shape=(nx^2, nx^2)
    ndarray
        If `return_maps` is True, the noise maps used to compute the
        covariance, shape=(n_maps, nx, nx)
    """
    noise_maps = make_maps_with_same_pk(noise_map, pix_size, int(n_maps))
    cov, inv_cov = covmat_from_noise_maps(noise_maps, method=method)
    if return_maps:
        return cov, inv_cov, noise_maps
    else:
        return cov, inv_cov
