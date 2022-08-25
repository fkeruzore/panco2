import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Ellipse

plt.ion()


def mapview(
    data,
    smooth=None,
    imrange="minmax",
    fig=None,
    ax=None,
    cmap="RdBu_r",
    wcs=None,
    cbar_label=None,
):
    """
    Args
    ----
    - data

    Kwargs
    ------
    - smooth  : None or float, sigma of smoothing to apply in pixels,
    - imrange : image range to display, one of
        * "minmax" : between minimum and maximum of the map,
        * "sym"    : symmetrical around zero,
        * f        : between the (f) and (100-f) percentiles of pixels,
        * [f1, f2] : between (f1) and (f2)
    - fig  : plt.figure instance,
    - ax   : plt.ax instance,
    - cmap : either a string for a valid plt colormap, or a colormap
    - cbar_label : string, label for the colorbar

    Returns
    -------
    - fig : plt.figure instance,
    - ax   : plt.ax instance
    """

    if smooth is not None:
        data = gaussian_filter(data, smooth)
        interpolation = "gaussian"
    else:
        interpolation = "none"

    if imrange == "minmax":
        vmin = np.min(data)
        vmax = np.max(data)
    elif imrange == "sym":
        vmax = np.max([data.max(), -data.min()])
        vmin = -vmax
    elif type(imrange) in [float, int]:
        percentiles = np.sort([imrange, 100.0 - imrange])
        vmin, vmax = np.percentile(data, percentiles)
    elif type(imrange) in [list, np.ndarray]:
        vmin = np.min(imrange)
        vmax = np.max(imrange)

    subplot_kw = {"projection": wcs} if wcs is not None else {}

    if fig is None and ax is None:
        fig, ax = plt.subplots(subplot_kw=subplot_kw)
    elif fig is not None and ax is None:
        ax = fig.add_subplot(111, **subplot_kw)

    im = ax.imshow(
        data,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        cmap=cmap,
        interpolation=interpolation,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)

    return fig, ax


def fitsview(
    fits_file,
    ext,
    crop=None,
    scale=1.0,
    offset=0.0,
    smooth=None,
    imrange="minmax",
    fig=None,
    ax=None,
    cmap="RdBu_r",
    fwhm=None,
    beam_color="k",
    cbar_label=None,
):
    """
    Args
    ----
    - fits_file
    - ext : int, index of hdu tou want to see

    Kwargs
    ------
    - crop    : None or astropy quantity, the size of the map you want to display
    - scale   : float, factor to multiply your map before displaying it
    - offset  : float, factor to add to your map before displaying it
    - smooth  : None or float, sigma of smoothing to apply in pixels,
    - imrange : image range to display, one of
        * "minmax" : between minimum and maximum of the map,
        * "sym"    : symmetrical around zero,
        * f        : between the (f) and (100-f) percentiles of pixels,
        * [f1, f2] : between (f1) and (f2)
    - fig  : plt.figure instance,
    - ax   : plt.ax instance,
    - cmap : either a string for a valid plt colormap, or a colormap
    - fwhm : None, float or Quantity, FWHM to add at the bottom left of the plot.
             If float, in pixels, if Quantity, in angular units.
    - cbar_label : string, label for the colorbar

    Returns
    -------
    fig, ax
    """

    hdulist = fits.open(fits_file)
    hdu = hdulist[ext]
    wcs = WCS(hdu.header)

    if crop is not None:
        center = SkyCoord(
            hdu.header["CRVAL1"] * u.deg, hdu.header["CRVAL2"] * u.deg
        )  # RA, dec
        crop = Cutout2D(hdu.data, center, crop, wcs=wcs)

        data = crop.data
        wcs = crop.wcs

    else:
        data = hdu.data

    data *= scale
    data += offset

    fig, ax = mapview(
        data,
        smooth=smooth,
        imrange=imrange,
        fig=fig,
        ax=ax,
        cmap=cmap,
        wcs=wcs,
        cbar_label=cbar_label,
    )

    if isinstance(fwhm, u.Quantity):
        pix_size = (np.abs(proj_plane_pixel_scales(wcs)) * u.deg)[0].to(
            "arcsec"
        )
        fwhm = (fwhm / pix_size).decompose().value
    if fwhm is not None:
        fwhm_plot = Ellipse(
            (fwhm, fwhm),
            fwhm,
            fwhm,
            edgecolor=beam_color,
            fill=False,
            hatch=r"xxxx",
        )
        ax.add_patch(fwhm_plot)

    hdulist.close()

    return fig, ax

def cmap_from_colors(col_list):
    import matplotlib as mpl
    from scipy.interpolate import interp1d

    x = np.arange(256)
    n = len(col_list)
    rgb_list = np.array([mpl.colors.to_rgb(col) for col in col_list])
    x_list = np.linspace(0.0, 255.0, n)
    rgb_interp = np.array(
        [interp1d(x_list, rgb_list[:, i], kind="linear")(x) for i in range(3)]
    )
    return mpl.colors.ListedColormap(rgb_interp.T)
