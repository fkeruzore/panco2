import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from scipy.stats import gaussian_kde
import time
import _utils

# ---------------------------------------------------------------------- #


def ps_map(ps_fluxes, ps_pos, xmap_ps, ymap_ps, beam_pix):
    ps_fluxes = np.array(ps_fluxes)
    nps = np.product(ps_fluxes.shape)
    return np.sum(
        ps_fluxes.reshape(nps, 1, 1)
        * np.exp(
            -0.5
            * (
                (xmap_ps - ps_pos[:, 0].reshape(nps, 1, 1)) ** 2
                + (ymap_ps - ps_pos[:, 1].reshape(nps, 1, 1)) ** 2
            )
            / beam_pix ** 2
        ),
        axis=0,
    )


# ---------------------------------------------------------------------- #


def subtract_ps(in_map, ps_fluxes, ps_pos, beam_pix):
    npix = in_map.shape[0]
    nps = len(ps_fluxes)
    xmap, ymap = np.meshgrid(np.arange(npix), np.arange(npix))
    xmap_ps = np.stack([xmap for i in range(nps)])
    ymap_ps = np.stack([ymap for i in range(nps)])
    return in_map - ps_map(ps_fluxes, ps_pos, xmap_ps, ymap_ps, beam_pix)
