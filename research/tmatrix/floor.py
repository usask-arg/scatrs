"""The f64 floor of the EBCM solve with exactly computed elements, over a grid.

For one mode: the elements of P and Q at high precision, rounded to 53 bits and solved at 53 bits
(C: what cancellation-free integrals give at best in f64), with Q's Skeel condition number.

Usage: python floor.py x eps s_re s_im m nmax ntheta
"""

import sys
import time

import mpmath as mp
from columns import err, rounded, skeel
from ebcm import qmatrices


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    t0 = time.time()
    mp.mp.dps = 45
    q, p = qmatrices(x, eps, mp.mpc(sr, si), m, nmax, nth)
    t = -(p * mp.inverse(q))
    with mp.workprec(53):
        tc = -(rounded(p) * mp.inverse(rounded(q)))
    print(
        f"x={x:g} eps={eps:g} s={sr:g}+{si:g}i m={m} N={nmax} nθ={nth}: "
        f"Skeel {mp.nstr(skeel(q), 2)} f64 floor {mp.nstr(err(tc, t), 2)} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
