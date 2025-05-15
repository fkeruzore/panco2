import numpy as np
from numpy.testing import assert_allclose

from panco2 import reduce_integrals, reduce_integrals_vectorized


def test_myrincbeta_equivalence():
    # Test a range of x, a, b values
    xvals = [0.0, 0.1, 0.5, 0.9, 1.0, np.linspace(0, 1, 10)]
    abvals = [
        (0.5, 0.5),
        (2.0, 2.0),
        (1.0, 3.0),
        (-0.5, 2.0),
        (2.0, -0.5),
    ]
    for x in xvals:
        for a, b in abvals:
            # Skip undefined cases: x == 0 and a < 0, or x == 1 and b < 0
            arr_x = np.asarray(x)
            if (np.any(arr_x == 0) and a < 0) or (
                np.any(arr_x == 1) and b < 0
            ):
                continue
            out_old = reduce_integrals.myrincbeta(x, a, b)
            out_new = reduce_integrals_vectorized.myrincbeta(x, a, b)
            assert_allclose(out_old, out_new, rtol=1e-10, atol=1e-12)


def test_myredcosine_equivalence():
    # Test a range of tmax, n values
    tvals = [0.0, 0.1, 0.5, 1.0, np.pi / 2, np.linspace(0, np.pi / 2, 10)]
    nvals = [-3.0, -2.0, -1.0, 0.0, 1.0]
    for tmax in tvals:
        for n in nvals:
            # Only test n == -1.0 or n == -2.0 (and n < -2 for recursion)
            if not (n == -1.0 or n == -2.0 or n < -2):
                continue
            out_old = reduce_integrals.myredcosine(tmax, n)
            out_new = reduce_integrals_vectorized.myredcosine(tmax, n)
            assert_allclose(out_old, out_new, rtol=1e-10, atol=1e-12)
