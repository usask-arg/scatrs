"""Where does the EBCM lose precision? For one mode of one spheroid:

  truth  elements and solve at HI digits;
  B      everything at 53 bits (plain f64);
  C      elements at HI digits rounded to 53 bits, solve at 53 bits;
  D      the same rounded elements, solved at HI digits;

with the normwise condition number of Q and its Skeel (componentwise) condition number
‖ |Q⁻¹| |Q| ‖∞, which bounds the error of T caused by relative errors of 2⁻⁵³ in the elements of
Q, whatever their scaling. Errors are max |ΔT| / max |T| over the block.

Usage: python precision.py x eps s_re s_im m nmax ntheta
"""

import sys
import time

import mpmath as mp
from ebcm import qmatrices

HI = 60


def rounded(a, prec=53):
    with mp.workprec(prec):
        return mp.matrix([[+a[i, j] for j in range(a.cols)] for i in range(a.rows)])


def solve(p, q, prec):
    with mp.workprec(prec):
        return -(p * mp.inverse(q))


def err(t, ref):
    big = max(abs(ref[i, j]) for i in range(ref.rows) for j in range(ref.cols))
    return max(abs(t[i, j] - ref[i, j]) for i in range(ref.rows) for j in range(ref.cols)) / big


def norm_inf(a):
    return max(sum(abs(a[i, j]) for j in range(a.cols)) for i in range(a.rows))


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    s = mp.mpc(sr, si)
    t0 = time.time()
    mp.mp.dps = HI
    q, p = qmatrices(x, eps, s, m, nmax, nth)
    t_true = solve(p, q, mp.mp.prec)
    qinv = mp.inverse(q)
    cond = norm_inf(q) * norm_inf(qinv)
    n = q.rows
    absq = mp.matrix([[abs(q[i, j]) for j in range(n)] for i in range(n)])
    absqi = mp.matrix([[abs(qinv[i, j]) for j in range(n)] for i in range(n)])
    skeel = norm_inf(absqi * absq)
    qr, pr = rounded(q), rounded(p)
    t_c = solve(pr, qr, 53)
    t_d = solve(pr, qr, mp.mp.prec)
    with mp.workprec(53):
        q53, p53 = qmatrices(x, eps, s, m, nmax, nth)
        t_b = -(p53 * mp.inverse(q53))
    # Element accuracy of the 53-bit integrals (normwise per matrix).
    eq = err(q53, q)
    ep = err(p53, p)
    # Componentwise: the worst relative error over the elements of Q.
    rel = max(
        abs(q53[i, j] - q[i, j]) / abs(q[i, j])
        for i in range(n)
        for j in range(n)
        if abs(q[i, j]) > 0
    )
    print(
        f"x={x} eps={eps} s={sr}+{si}i m={m} N={nmax} nθ={nth}: "
        f"cond {mp.nstr(cond, 2)} skeel {mp.nstr(skeel, 2)} | "
        f"B {mp.nstr(err(t_b, t_true), 2)} C {mp.nstr(err(t_c, t_true), 2)} "
        f"D {mp.nstr(err(t_d, t_true), 2)} | Q53 norm {mp.nstr(eq, 2)} worst-rel {mp.nstr(rel, 2)} "
        f"P53 {mp.nstr(ep, 2)} ({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
