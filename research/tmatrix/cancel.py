"""The cancellation inside the Q integrals, element by element: for each element, the sum of the
absolute values of every elementary product (per node, per vector component, per block) over
|Q_nk|. Times 2⁻⁵³ it predicts the f64 error of the element. Split by the vector component of
the triple product: the r̂·(θ̂×φ̂) part (n_r) and the −(r'/r) θ̂·(φ̂×r̂) part (n_θ).

Usage: python cancel.py x eps s_re s_im m nmax ntheta
"""

import sys

import mpmath as mp
from ebcm import angular, gauss_legendre, spheroid, vswf


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    mp.mp.dps = 40
    s = mp.mpc(sr, si)
    shape = spheroid(x, eps)
    nodes, weights = gauss_legendre(nth)
    n0 = max(m, 1)
    kk = nmax - n0 + 1
    # Q blocks as weighted J's: Q11 = −i s J12 − i J21, Q12 = −i s J11 − i J22,
    # Q21 = −i s J22 − i J11, Q22 = −i s J21 − i J12 (outgoing X only).
    blocks = {
        (1, 1): ((1, 2, s), (2, 1, 1)),
        (1, 2): ((1, 1, s), (2, 2, 1)),
        (2, 1): ((2, 2, s), (1, 1, 1)),
        (2, 2): ((2, 1, s), (1, 2, 1)),
    }
    val = {b: mp.matrix(kk, kk) for b in blocks}
    absr = {b: mp.matrix(kk, kk) for b in blocks}
    abst = {b: mp.matrix(kk, kk) for b in blocks}
    for ct, w in zip(nodes, weights, strict=False):
        st = mp.sqrt(1 - ct * ct)
        r, dr = shape(ct, st)
        d, pi_, tau = angular(m, nmax, ct, st)
        nth_ = -dr / r
        fac = w * r * r * 2 * mp.pi
        ym, yn = vswf(m, nmax, s * r, "j", d, pi_, tau, flip=False)
        xm, xn = vswf(m, nmax, r, "h", d, pi_, tau, flip=True)
        X = {1: xm, 2: xn}
        Y = {1: ym, 2: yn}
        for b, terms in blocks.items():
            for a1, b1, wgt in terms:
                for i in range(n0, nmax + 1):
                    A = X[a1][i]
                    for j in range(n0, nmax + 1):
                        B = Y[b1][j]
                        p1, p2 = A[1] * B[2], A[2] * B[1]
                        p3, p4 = A[2] * B[0], A[0] * B[2]
                        val[b][i - n0, j - n0] += -1j * wgt * fac * ((p1 - p2) + nth_ * (p3 - p4))
                        absr[b][i - n0, j - n0] += abs(wgt * fac) * (abs(p1) + abs(p2))
                        abst[b][i - n0, j - n0] += abs(wgt * fac * nth_) * (abs(p3) + abs(p4))
    for b in blocks:
        worst = (0, None)
        for i in range(kk):
            for j in range(kk):
                v = abs(val[b][i, j])
                if v < mp.mpf(10) ** -30 * (absr[b][i, j] + abst[b][i, j] + 1e-300):
                    continue  # a structural zero (parity)
                ratio = (absr[b][i, j] + abst[b][i, j]) / v
                if ratio > worst[0]:
                    worst = (ratio, (i + n0, j + n0, absr[b][i, j] / v, abst[b][i, j] / v, v))
        r_, (n, k, fr, ft, v) = worst
        print(
            f"Q{b[0]}{b[1]}: worst cancellation {mp.nstr(r_, 3)} at n={n}, k={k} "
            f"(r̂ part {mp.nstr(fr, 3)}, θ̂ part {mp.nstr(ft, 3)}, |Q|={mp.nstr(v, 3)})"
        )


if __name__ == "__main__":
    main()
