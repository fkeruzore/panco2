import numpy as np
import scipy.special as sps


def myrincbeta(x, a, b):
    """
    Compute the regularized incomplete beta function in a vectorized way.
    Handles negative a recursively, otherwise uses scipy's betainc.
    """
    x = np.asarray(x)
    if np.any(a < 0):
        # Only works if a is scalar or all a < 0
        cbf = (sps.gamma(a) * sps.gamma(b)) / sps.gamma(a + b)
        res = (x**a * (1.0 - x) ** b) / (a * cbf)
        return myrincbeta(x, a + 1.0, b) + res
    else:
        # sps.betainc is already vectorized and regularized
        return sps.betainc(a, b, x)


def myredcosine(tmax, n):
    """
    Computes \int_0^tmax cos^n(x) dx in a vectorized way.
    Handles n < -2 recursively, otherwise uses closed forms for
    n == -1 or n == -2.
    """
    tmax = np.asarray(tmax)
    if np.any(n < -2):
        # Only works if n is scalar or all n < -2
        res = np.cos(tmax) ** (n + 1) * np.sin(tmax) / (n + 1)
        return myredcosine(tmax, n + 2) * (n + 2) / (n + 1) - res
    else:
        if n == -1.0:
            return np.log(np.abs(1.0 / np.cos(tmax) + np.tan(tmax)))
        elif n == -2.0:
            return np.tan(tmax)
        else:
            # For other n >= -2, not defined in original code
            return np.zeros_like(tmax)
