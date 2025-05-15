import numpy as np
import scipy.special as sps
from . import reduce_integrals as ri


def shell_pl(
    epsnot, sindex, rmin, rmax, radarr, c=1.0, ff=1e-3, epsatrmin=0, narm=False
):
    rmin = np.maximum(rmin, 0)
    radarr = np.array(radarr, dtype=float)
    radarr = np.where((radarr == 0) & (sindex > 0), ff, radarr)
    p = sindex / 2.0

    # Vectorized case selection
    rmax_inf = rmax < 0
    rmin_zero = rmin == 0
    rmin_lt_rmax = rmin < rmax
    rmin_gt_rmax = rmin > rmax

    # scase: 0=sphere, 1=shell, 2=sphole, 3=infty
    scase = (
        3 * (rmax_inf & rmin_zero)
        + 2 * (rmax_inf & ~rmin_zero)
        + 0 * (~rmax_inf & rmin_zero)
        + 1 * (~rmax_inf & ~rmin_zero & rmin_lt_rmax)
        + 3 * (~rmax_inf & ~rmin_zero & rmin_gt_rmax)
    )

    # epsnorm logic, avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_rmin = np.where(rmin == 0, np.nan, rmin)
        epsnorm = np.where(
            (scase == 1) & (narm is False),
            epsnot * (rmax / safe_rmin) ** (sindex),
            epsnot,
        )
    prefactors = epsnorm * c

    # Compute all cases
    res_sphere = plsphere_vec(p, rmin, rmax, radarr)
    res_shell = plshell_vec(p, rmin, rmax, radarr)
    res_sphole = plsphole_vec(p, rmin, rmax, radarr)
    res_infty = plinfty_vec(p, rmin, rmax, radarr)

    # Select result for each element
    result = (
        np.where(scase == 0, res_sphere, 0)
        + np.where(scase == 1, res_shell, 0)
        + np.where(scase == 2, res_sphole, 0)
        + np.where(scase == 3, res_infty, 0)
    )
    return result * prefactors


def plsphere_vec(p, rmin, rmax, radarr):
    radarr = np.asarray(radarr)
    c1 = radarr <= rmax
    with np.errstate(divide="ignore", invalid="ignore"):
        sir = np.where(c1, radarr / rmax, 0)
    isni = (2.0 * p == np.floor(2.0 * p)) & (p <= 1)
    # Vectorized special case
    tmax = np.where(c1, np.arctan(np.sqrt(1.0 - sir**2) / sir), 0)
    plint_isni = np.where(
        c1 & isni,
        ri.myredcosine(tmax, 2.0 * p - 2.0) * (sir ** (1.0 - 2.0 * p)) * 2.0,
        0,
    )
    cbf = (sps.gamma(p - 0.5) * np.sqrt(np.pi)) / sps.gamma(p)
    ibir = ri.myrincbeta(sir**2, p - 0.5, 0.5)
    plint_else = np.where(
        c1 & ~isni,
        (sir ** (1.0 - 2.0 * p)) * (1.0 - ibir) * cbf,
        0,
    )
    myres = np.where(isni, plint_isni, plint_else)
    return myres * rmax


def plshell_vec(p, rmin, rmax, radarr):
    radarr = np.asarray(radarr)
    c1 = radarr <= rmax
    c2 = np.where(c1, radarr < rmin, False)
    c3 = radarr < rmin
    with np.errstate(divide="ignore", invalid="ignore"):
        sir = np.where(c1, radarr / rmin, 0)
        sor = np.where(c1, radarr / rmax, 0)
    isni = (2.0 * p == np.floor(2.0 * p)) & (p <= 1)
    tmxo = np.zeros_like(radarr)
    tmxo[c1] = np.arctan(np.sqrt(1.0 - sor[c1] ** 2) / sor[c1])
    tmxi = np.zeros_like(radarr)
    if np.any(c2):
        tmxi[c2] = np.arctan(np.sqrt(1.0 - sir[c2] ** 2) / sir[c2])
    # Special case
    plint_isni = np.zeros_like(radarr)
    plint_isni[c1] = ri.myredcosine(tmxo[c1], 2.0 * p - 2.0)
    if np.any(c2):
        plint_isni[c2] -= ri.myredcosine(tmxi[c2], 2.0 * p - 2.0)
    plint_isni[c1] *= (sir[c1] ** (1.0 - 2.0 * p)) * 2.0
    # General case
    cbf = (sps.gamma(p - 0.5) * np.sqrt(np.pi)) / sps.gamma(p)
    ibir = np.zeros_like(radarr)
    ibor = np.zeros_like(radarr)
    if np.any(c2):
        ibir[c2] = ri.myrincbeta(sir[c2] ** 2, p - 0.5, 0.5)
    if np.any(c1):
        ibor[c1] = ri.myrincbeta(sor[c1] ** 2, p - 0.5, 0.5)
    plinn = np.zeros_like(radarr)
    if np.any(c1):
        plinn[c1] = sir[c1] ** (1.0 - 2.0 * p)
    plint_else = np.zeros_like(radarr)
    if np.any(c1):
        plint_else[c1] = plinn[c1] * (1.0 - ibor[c1]) * cbf
    if np.any(c2):
        plint_else[c3] = plinn[c2] * (ibir[c2] - ibor[c2]) * cbf
    # Combine
    myres = np.where(isni, plint_isni, plint_else)
    return myres * rmin


def plsphole_vec(p, rmin, rmax, radarr):
    radarr = np.asarray(radarr)
    p_half = p <= 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = radarr / rmin
    cbf = (sps.gamma(p - 0.5) * np.sqrt(np.pi)) / sps.gamma(p)
    ibor = ri.myrincbeta(sr**2, p - 0.5, 0.5)
    plt = sr ** (1.0 - 2.0 * p)
    myres = np.where(p_half, radarr * 0 - 1.0e10, 0)
    myres = np.where(~p_half & (radarr < rmin), plt * ibor * cbf, myres)
    myres = np.where(~p_half & (radarr >= rmin), plt * cbf, myres)
    return myres * rmin


def plinfty_vec(p, rmin, rmax, radarr):
    # Always returns 0 (as in original)
    return np.zeros_like(radarr)
