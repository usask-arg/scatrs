"""Mixed-precision iterative refinement for T = −P Q⁻¹.

Elements of P and Q in the outgoing orders n ≥ cut as double-double integrals would give them
(rounded to 106 bits), the rest rounded to 53; Q factorized at 53 bits; then
    X ← X + (R) Q₅₃⁻¹,  R = −P − X Q  (residual at 106 bits, with the mixed elements),
solving X Q = −P from the right. Converges when Skeel(Q)·2⁻⁵³ < 1.

Usage: python refine.py x eps s_re s_im m nmax ntheta
"""

import sys
import time

import mpmath as mp
from columns import err
from ebcm import qmatrices


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    t0 = time.time()
    mp.mp.dps = 45
    q, p = qmatrices(x, eps, mp.mpc(sr, si), m, nmax, nth)
    t = -(p * mp.inverse(q))
    n = q.rows
    kk = n // 2
    n0 = max(m, 1)

    def run(cut):
        qq, pp = q.copy(), p.copy()
        for i in range(n):
            prec = 106 if n0 + (i % kk) >= cut else 53
            with mp.workprec(prec):
                for j in range(n):
                    qq[i, j] = +q[i, j]
                    pp[i, j] = +p[i, j]
        with mp.workprec(53):
            q53 = mp.matrix([[+qq[i, j] for j in range(n)] for i in range(n)])
            inv53 = mp.inverse(q53)  # stands for the f64 LU factors
            xs = -(mp.matrix([[+pp[i, j] for j in range(n)] for i in range(n)]) * inv53)
        errs = [err(xs, t)]
        for _ in range(4):
            with mp.workprec(106):
                res = -pp - xs * qq
            with mp.workprec(53):
                corr = mp.matrix([[+res[i, j] for j in range(n)] for i in range(n)]) * inv53
            with mp.workprec(106):
                xs = xs + corr
            errs.append(err(xs, t))
        return errs

    out = []
    for label, cut in (
        ("all rows dd", n0),
        ("rows n>=N/2 dd", n0 + kk // 2),
        ("rows n>=N/4 dd", n0 + kk // 4),
    ):
        e = run(cut)
        out.append(f"{label}: " + " -> ".join(mp.nstr(v, 2) for v in e))
    print(
        f"x={x:g} eps={eps:g} N={nmax}: " + "; ".join(out) + f" ({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
