"""How sensitive is T to relative errors of 2⁻⁵³ in the sampled functions at the nodes, each
factor on its own: the internal functions (shared by every element of their columns, in P and
Q alike), the regular external ones (P's rows) and the outgoing ones (Q's rows)? Elements and
solve otherwise exact.

Usage: python sensitivity.py x eps s_re s_im m nmax ntheta
"""

import random
import sys

import mpmath as mp
from columns import err
from ebcm import qmatrices


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    mp.mp.dps = 45
    s = mp.mpc(sr, si)
    q, p = qmatrices(x, eps, s, m, nmax, nth)
    t = -(p * mp.inverse(q))
    u = mp.mpf(2) ** -53
    out = []
    for which in ("inner", "reg", "out"):
        rng = random.Random(1)

        def perturb(kind, funcs, which=which, rng=rng):
            if kind != which:
                return funcs
            res = list(funcs)
            for i, f in enumerate(funcs):
                if f is None:
                    continue
                res[i] = tuple(
                    c * (1 + u * mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1))) for c in f
                )
            return res

        q2, p2 = qmatrices(x, eps, s, m, nmax, nth, perturb=perturb)
        out.append(f"{which} {mp.nstr(err(-(p2 * mp.inverse(q2)), t), 2)}")
    print(
        f"x={x:g} eps={eps:g} m={m} N={nmax}: T error from 2^-53 sample errors: " + ", ".join(out),
        flush=True,
    )


if __name__ == "__main__":
    main()
