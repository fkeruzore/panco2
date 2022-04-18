import numpy as np
import scipy as sp
import scipy.special as sps
import reduce_integrals as ri

# https://github.com/CharlesERomero/MCMC_ICM_PP/blob/master/integrations/shell_pl.py

def shell_pl(
    epsnot, sindex, rmin, rmax, radarr, c=1.0, ff=1e-3, epsatrmin=0, narm=False
):

    # ========================================================================== #
    # ===== Written by Charles Romero. IRAM.
    # =====
    # ===== PURPOSE: Integrate a power law function (similiar to emissivity)
    # =====          along the z-axis (i.e. line of sight). This performs the
    # =====          integration analytically.
    # =====
    # ===== HISTORY:
    # ===== 25.06.2016 - CR: Created.
    # =====
    # ========================================================================== #
    # ===== INPUTS:
    # =====
    #
    # EPSNOT    - The normalization factor. The default behavior is for
    #             this to be defined at RMAX, the outer edge of a sphere
    #             or shell. If you integrate to infinity, then this should
    #             be defined at RMIN. And of course, RMIN=0, and RMAX as
    #             infinity provides no scale on which to define EPSNOT.
    #             See the optional variable EPSATRMIN.
    # SINDEX    - "Spectral Index". That is, the power law
    #             (without the minus sign) that the "emissivity"
    #             follows within your bin. If you want to integrate to
    #             infinity, you must have SINDEX > 1. All other cases can
    #             handle any SINDEX value.
    # RMIN      - Minimum radius for your bin. Can be 0.
    # RMAX      - Maximum radius for your bin. If you wish to set this
    #             to infinity, then set it to a negative value.
    #
    # ===== -- NOTE -- If RMIN = 0 and RMAX < 0, then this program will return 0.
    #
    # RADARR    - A radial array of projected radii (same units as RMIN
    #             and RMAX) for which projected values will be calculated.
    #             If the innermost value is zero, its value, in the scaled
    #             radius array will be set to FF.
    # [C=1]     - The scaling axis for an ellipse along the line of sight.
    #             The default
    # [FF=1e-3] - Fudge Factor. If the inner
    # [EPSATRMIN] - Set this to a value greater than 0 if you want EPSNOT to be
    #               defined at RMIN. This automatically happens if RMAX<0
    # [NARM]    - Normalized At R_Min. This option specifies that you have *already*
    #             normalized the bins at R_Min (for a shell case). The other two cases are
    #             strictly imposed where the normalization is set. The default is False,
    #             because that is just how I started using this.
    #
    # ========================================================================== #
    # ===== OUTPUTS:
    # =====
    #
    # PLINT     - PLINT is the integration along the z-axis (line of sight) for
    #             an ellipsoid (a sphere) where the "emissivity" is governed by
    #             a power law. The units are thus given as the units on EPSNOT
    #             times the units on RADARR (and therefore RMIN and RMAX).
    #
    #             It is then dependent on you to make the appropriate
    #             conversions to the units you would like.
    #
    # ========================================================================== #

    # ===== Perform some double-checks. ===== #

    if rmin < 0:
        print("found rmin < 0; setting rmin equal to 0")
        rmin = 0

    rrmm = radarr == np.amin(radarr)
    if (radarr[rrmm] == 0) and (sindex > 0):
        radarr[rrmm] = ff

    # ===== Determine the appropriate case (and an extra double check) ===== #

    if rmax < 0:
        if rmin == 0:
            scase = 3
        else:
            scase = 2
            epsatrmin = 1
    else:
        if rmin == 0:
            scase = 0
        else:
            if rmin < rmax:
                scase = 1
                epsatrmin = 1
            else:
                print("You made a mistake: rmin > rmax; sending to infty integration.")
                # If a mistake is possible, it will happen, eventually.
                scase = 3

    # ===== Direct program to appropriate case: ===== #
    shellcase = {
        0: plsphere,  # You are integrating from r=0 to R (finite)
        1: plshell,  # You are integrating from r=R_1 to R_2 (finite)
        2: plsphole,  # You are integrating from r=R (finite, >0) to infinity
        3: plinfty,  # You are integrating from r=0 to infinity
    }

    # Redo some numbers to agree with hand-written calculations
    p = sindex / 2.0  # e(r) = e_0 * (r^2)^(-p) for this notation / program

    # In a way, I actually like having EPSNORM default to being defined at RMIN
    # (Easier to compare to hand-written calculations.

    if scase == 1 and narm == False:
        epsnorm = epsnot * (rmax / rmin) ** (sindex)
    else:
        epsnorm = epsnot

    # Prefactors change a bit depending on integration method.
    # These are the only "pre"factors common to all (both) methods.
    prefactors = epsnorm * c
    # Now integrate for the appropriate case
    myintegration = shellcase[scase](p, rmin, rmax, radarr)
    answer = myintegration * prefactors  ## And get your answer!
    return answer


# ===== Integration cases, as directed above. ===== #


def plsphere(p, rmin, rmax, radarr):
    c1 = radarr <= rmax  # condition 1
    c2 = radarr > rmax  # condition 2
    #    c1 = np.where(radarr<=rmax)     # condition 1
    #    c2 = np.where(radarr>rmax)      # condition 2
    sir = radarr[c1] / rmax  # scaled radii
    isni = (2.0 * p == np.floor(2.0 * p)) and (p <= 1)  # Special cases -> "method 2"
    if isni:
        tmax = np.arctan(np.sqrt(1.0 - sir ** 2) / sir)  # Theta max
        plint = (
            ri.myredcosine(tmax, 2.0 * p - 2.0) * (sir ** (1.0 - 2.0 * p)) * 2.0
        )  # Integration + prefactors
    else:
        cbf = (sps.gamma(p - 0.5) * np.sqrt(np.pi)) / sps.gamma(
            p
        )  # complete beta function
        ibir = ri.myrincbeta(sir ** 2, p - 0.5, 0.5)  # incomplete beta function
        plint = (
            (sir ** (1.0 - 2.0 * p)) * (1.0 - ibir) * cbf
        )  # Apply appropriate "pre"-factors

    myres = radarr * 0  # Just make my array (unecessary?)
    myres[c1] = plint  # Define values for R < RMIN
    return myres * rmax  # The results we want


def plshell(p, rmin, rmax, radarr):
    c1 = radarr <= rmax  # condition 1
    c2 = radarr[c1] < rmin  # condition 2
    c3 = radarr < rmin  # c1[c2] as I would expect in IDL
    #    c1 = np.where(radarr<=rmax)     # condition 1
    #    c2 = np.where(radarr[c1]<rmin)  # condition 2
    sir = radarr[c1] / rmin  # scaled inner radii
    sor = radarr[c1] / rmax  # scaled outer radii
    isni = (2.0 * p == np.floor(2.0 * p)) and (p <= 1)  # Special cases -> "method 2"
    myres = radarr * 0  # Just make my array (unecessary?)
    if isni:
        tmxo = np.arctan(np.sqrt(1.0 - sor ** 2) / sor)  # Theta max...outer circle
        tmxi = np.arctan(
            np.sqrt(1.0 - sir[c2] ** 2) / sir[c2]
        )  # Theta max...inner circle
        plint = ri.myredcosine(tmxo, 2.0 * p - 2.0)  # Integrate for outer circle.
        plint[c2] -= ri.myredcosine(
            tmxi, 2.0 * p - 2.0
        )  # Integrate and subtract inner circle
        #      myres[c1]=plint*(sor**(1.0-2.0*p))*2.0    # Pre-(24 July 2017) line.
        myres[c1] = (
            plint * (sir ** (1.0 - 2.0 * p)) * 2.0
        )  # Apply appropriate "pre"-factors

    else:
        cbf = (sps.gamma(p - 0.5) * np.sqrt(np.pi)) / sps.gamma(
            p
        )  # complete beta function
        ibir = ri.myrincbeta(sir[c2] ** 2, p - 0.5, 0.5)  # Inc. Beta for inn. rad.
        ibor = ri.myrincbeta(sor ** 2, p - 0.5, 0.5)  # Inc. Beta for out. rad.
        plinn = sir ** (1.0 - 2.0 * p)  # Power law term for inner radii
        myres[c1] = plinn * (1.0 - ibor) * cbf  # Define values for the enclosed circle
        #      import pdb;pdb.set_trace()
        #      myres[c1[c2]]=plinn[c2]*(ibir-ibor[c2])*cbf # Correct the values for the
        # ===== Changed this March 9, 2018: ===== #
        myres[c3] = plinn[c2] * (ibir - ibor[c2]) * cbf  # Correct the values for the
        # inner circle
    return myres * rmin  # The results we want


def plsphole(p, rmin, rmax, radarr):
    if p <= 0.5:
        #      print p,'Error iminent'
        return radarr * 0 - 1.0e10

    else:
        c1 = radarr < rmin  # condition 1
        c2 = radarr >= rmin  # condition 2
        #      c1 = np.where(radarr<rmin)     # condition 1
        #      c2 = np.where(radarr>=rmin)    # condition 2
        sr = radarr / rmin  # scaled radii
        cbf = (sps.gamma(p - 0.5) * np.sqrt(np.pi)) / sps.gamma(
            p
        )  # complete beta function
        ibor = ri.myrincbeta(sr[c1] ** 2, p - 0.5, 0.5)  # Inc. Beta for out. rad.
        plt = sr ** (1.0 - 2.0 * p)  # Power law term
        myres = radarr * 0  # Just make my array (unecessary?)
        myres[c1] = plt[c1] * ibor * cbf  # Define values for R < RMIN
        myres[c2] = plt[c2] * cbf  # Define values for R > RMIN
        return myres * rmin


def plinfty(p, rmin, rmax, radarr):
    sr = radarr  # scaled radii
    cbf = (sps.gamma(p - 0.5) * np.sqrt(np.pi)) / sps.gamma(p)  # complete beta function
    plt = sr ** (1.0 - 2.0 * p)  # Power law term

    # There is no scaling to be done: RMIN=0; RMAX=infinity...
    # This is madness, but if you can set >>SOME<< scaling radius, this can work.
    # However, the practical implementation of this is not foreseen / understood
    # how it should look. Therefore, for now, I will return 0.

    return 0  # Scale invariant. Right. Fail.
