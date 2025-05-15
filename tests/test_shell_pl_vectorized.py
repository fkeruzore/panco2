import numpy as np
from numpy.testing import assert_allclose

from panco2 import shell_pl, reduce_integrals
from panco2 import shell_pl_vectorized, reduce_integrals_vectorized


def test_shell_pl_equivalence():
    # Test a range of parameters, including edge cases
    test_cases = [
        # (epsnot, sindex, rmin, rmax, radarr)
        (1.0, 2.0, 0.0, 1.0, np.linspace(0, 1, 10)),
        (2.0, 1.5, 0.1, 2.0, np.linspace(0.1, 2, 20)),
        (1.0, 2.5, 0.0, -1.0, np.linspace(0, 2, 15)),  # rmax < 0 (infty)
        (1.0, 2.0, 0.5, 1.5, np.array([0.0, 0.5, 1.0, 1.5, 2.0])),
        (1.0, 2.0, 0.0, 1.0, np.array([0.0, 0.1, 0.5, 0.9, 1.0])),
    ]
    for epsnot, sindex, rmin, rmax, radarr in test_cases:
        # Skip undefined cases for myrincbeta/myredcosine inside shell_pl
        # (radarr == 0 and sindex/2 < 0) or (radarr == 1 and sindex/2 < 0)
        p = sindex / 2.0
        arr_radarr = np.asarray(radarr)
        if (np.any(arr_radarr == 0) and p < 0) or (
            np.any(arr_radarr == 1) and p < 0
        ):
            continue
        out_old = shell_pl.shell_pl(epsnot, sindex, rmin, rmax, radarr)
        out_new = shell_pl_vectorized.shell_pl(
            epsnot, sindex, rmin, rmax, radarr
        )
        assert_allclose(out_old, out_new, rtol=1e-10, atol=1e-12)


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


# ------------------- BENCHMARKS -------------------


def test_benchmark_shell_pl_old(benchmark):
    epsnot, sindex, rmin, rmax = 1.0, 2.0, 0.1, 2.0
    radarr = np.linspace(0.1, 2.0, 10000)
    benchmark(lambda: shell_pl.shell_pl(epsnot, sindex, rmin, rmax, radarr))


def test_benchmark_shell_pl_vectorized(benchmark):
    epsnot, sindex, rmin, rmax = 1.0, 2.0, 0.1, 2.0
    radarr = np.linspace(0.1, 2.0, 10000)
    benchmark(
        lambda: shell_pl_vectorized.shell_pl(
            epsnot, sindex, rmin, rmax, radarr
        )
    )


def test_benchmark_myrincbeta_old(benchmark):
    x = np.linspace(0.1, 0.9, 10000)  # avoid 0 and 1
    a, b = 2.0, 2.0
    benchmark(lambda: reduce_integrals.myrincbeta(x, a, b))


def test_benchmark_myrincbeta_vectorized(benchmark):
    x = np.linspace(0.1, 0.9, 10000)  # avoid 0 and 1
    a, b = 2.0, 2.0
    benchmark(lambda: reduce_integrals_vectorized.myrincbeta(x, a, b))


def test_benchmark_myredcosine_old(benchmark):
    tmax = np.linspace(0.1, np.pi / 2, 10000)
    n = -1.0
    benchmark(lambda: reduce_integrals.myredcosine(tmax, n))


def test_benchmark_myredcosine_vectorized(benchmark):
    tmax = np.linspace(0.1, np.pi / 2, 10000)
    n = -1.0
    benchmark(lambda: reduce_integrals_vectorized.myredcosine(tmax, n))
