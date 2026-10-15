"""Mixed precision: which elements need more than f64? Every element of P and Q rounded to 53 bits
except those in the internal orders (columns) k ≥ cut, or the outgoing orders (rows) n ≥ cut,
kept exact; solved exactly. Shows how much of the f64 floor a few columns or rows carry.

Usage: python mixed.py x eps s_re s_im m nmax ntheta
"""

import sys

import mpmath as mp
from columns import err
from ebcm import qmatrices


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    mp.mp.dps = 45
    q, p = qmatrices(x, eps, mp.mpc(sr, si), m, nmax, nth)
    t = -(p * mp.inverse(q))
    n = q.rows
    kk = n // 2
    n0 = max(m, 1)

    def order(i):
        return n0 + (i % kk)

    def solve_with(keep):
        qq, pp = q.copy(), p.copy()
        with mp.workprec(53):
            for i in range(n):
                for j in range(n):
                    if not keep(i, j):
                        qq[i, j] = +q[i, j]
                        pp[i, j] = +p[i, j]
        return err(-(pp * mp.inverse(qq)), t)

    line = [f"x={x:g} eps={eps:g} N={nmax}: all f64 {mp.nstr(solve_with(lambda i, j: False), 2)}"]
    for frac in (0.75, 0.5, 0.25):
        cut = n0 + int(frac * (nmax - n0 + 1))
        c = solve_with(lambda i, j, cut=cut: order(j) >= cut)
        r = solve_with(lambda i, j, cut=cut: order(i) >= cut)
        line.append(f"exact k>={cut}: {mp.nstr(c, 2)} / n>={cut}: {mp.nstr(r, 2)}")
    print("; ".join(line), flush=True)


if __name__ == "__main__":
    main()
