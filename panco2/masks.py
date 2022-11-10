import numpy as np
from astropy.io import fits
from astropy.wcs.utils import skycoord_to_pixel, proj_plane_pixel_scales
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


def mask_holes(ppf, coords, radius):
    """
    Creates a boolean mask by punching circular holes at given
    sky positions.

    Parameters
    ----------
    ppf : `panco2.PressureProfileFitter` instance
        The panco2 object to create a mask for.
        This is used to read in the WCS and map size.
    coords : List of `astropy.coordinates.SkyCoord` instances
        Coordinates of the centers of the holes.
    radius : float
        Radius of each hole, in pixels.

    Returns
    -------
    np.ndarray
        Boolean mask with circular holes at given positions.
        The convention is the right one to use as input for
        `panco2.PressureProfileFitter.add_mask()`, i.e. the
        mask is True in the pixels *to be masked* and False
        elsewhere.
    """
    mask = np.ones_like(ppf.sz_map, dtype=bool)
    n_pix = mask.shape[0]

    for coord in coords:
        pix_pos = skycoord_to_pixel(coord, ppf.wcs)
        xmap, ymap = np.meshgrid(
            np.arange(n_pix) - pix_pos[0], np.arange(n_pix) - pix_pos[1]
        )
        rmap = np.hypot(xmap, ymap)
        mask &= rmap > radius

    return ~mask


def cut_mask_from_fits(ppf, filename, hdu_mask):
    """
    Crops a mask from a FITS file.

    Parameters
    ----------
    ppf : `panco2.PressureProfileFitter` instance
        The panco2 object to create a mask for.
        This is used to read in the WCS and map size.
    filename : str
        Path to the FITS file containing the mask to be cropped.
    hdu_mask : int
        Index of the extension of the FITS file containing the mask.

    Returns
    -------
    np.ndarray
        Boolean mask with the same shape as the data.
        Warning: this does not check the mask -- if the FITS mask
        is 0 for masked pixels and 1 elsewhere, you'll need to invert
        it to use it in `panco2.PressureProfileFitter.add_mask()`.
    """
    hdulist = fits.open(filename)
    hdu = hdulist[hdu_mask]
    wcs_mask = WCS(hdu.header)

    # Sanity checks
    pix_size = np.abs(proj_plane_pixel_scales(wcs_mask) * 3600)
    assert pix_size.size == 2, "Can't process the header, is it not 2d?"
    assert (
        pix_size[0] == pix_size[1]
    ), "Can't process map with different pixel sizes in RA and dec"
    assert pix_size[0] == ppf.pix_size, (
        "Incompatible pixel size: "
        + f"PressureProfileFitter data ({ppf.pix_size} arcsec), "
        + f"FITS file ({pix_size[0]} arcsec)"
    )

    n_pix = ppf.sz_map.shape[0]
    mask = Cutout2D(hdu.data, ppf.coords_center, n_pix, wcs=wcs_mask)

    return mask.data.astype(bool)
