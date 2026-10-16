"""Integrals in an orthonormalized internal basis, at 53 bits.

The columns of P and Q are pairs of internal tangential fields (Ŷ₁, Ŷ₂): (s RgN_k, RgM_k) for
the first K columns and (s RgM_k, RgN_k) for the next K, integrated against (M_n, N_n) for the
first K rows and (N_n, M_n) for the next K (the J combinations of ebcm.qmatrices). A column
transform R applied to the internal functions, W_j = Σ_c Ŷ_c (R⁻¹)_{cj}, gives P R⁻¹ and Q R⁻¹ and
the same T.

Experiment: R from the QR of the exact stacked [P; Q]; the transformed internal functions W and
the outgoing and regular external functions are computed exactly and rounded to 53 bits at each
node, then the integrals and the solve run at 53 bits. Compares with the 53-bit plain integrals.
If this is accurate, an accurately computed orthonormal internal basis makes plain f64
integration sufficient.

Usage: python basis.py x eps s_re s_im m nmax ntheta
"""

import sys

import mpmath as mp
from columns import err, qr_r
from ebcm import angular, gauss_legendre, qmatrices, spheroid, vswf


def node_functions(x, eps, s, m, nmax, ct, st, shape):
    r, dr = shape(ct, st)
    d, pi_, tau = angular(m, nmax, ct, st)
    ym, yn = vswf(m, nmax, s * r, "j", d, pi_, tau, flip=False)
    hm, hn = vswf(m, nmax, r, "h", d, pi_, tau, flip=True)
    jm, jn = vswf(m, nmax, r, "j", d, pi_, tau, flip=True)
    return r, dr, (ym, yn), (hm, hn), (jm, jn)


def triple(nr, nt, a, b):
    return nr * (a[1] * b[2] - a[2] * b[1]) + nt * (a[2] * b[0] - a[0] * b[2])


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    mp.mp.dps = 60
    s = mp.mpc(sr, si)
    q, p = qmatrices(x, eps, s, m, nmax, nth)
    t = -(p * mp.inverse(q))
    n = q.rows
    stacked = mp.matrix(2 * n, n)
    for i in range(n):
        for j in range(n):
            stacked[i, j] = p[i, j]
            stacked[n + i, j] = q[i, j]
    rinv = mp.inverse(qr_r(stacked))
    n0 = max(m, 1)
    shape = spheroid(x, eps)
    nodes, weights = gauss_legendre(nth)
    acc = {"q": mp.matrix(n, n), "p": mp.matrix(n, n)}
    accp = {"q": mp.matrix(n, n), "p": mp.matrix(n, n)}  # plain 53-bit, for comparison

    def r53(v):
        with mp.workprec(53):
            return tuple(+c for c in v) if isinstance(v, tuple) else +v

    for ct, w in zip(nodes, weights, strict=False):
        st = mp.sqrt(1 - ct * ct)
        r, dr, (ym, yn), outh, outj = node_functions(x, eps, s, m, nmax, ct, st, shape)
        # Column functions (Ŷ1, Ŷ2) for c = 0..2K−1, exact.
        cols = []
        for k in range(n0, nmax + 1):
            cols.append((tuple(s * c for c in yn[k]), ym[k]))
        for k in range(n0, nmax + 1):
            cols.append((tuple(s * c for c in ym[k]), yn[k]))
        # Transformed columns W_j, exact, then rounded to 53 bits.
        wcols = []
        for j in range(n):
            w1 = [sum(cols[c][0][comp] * rinv[c, j] for c in range(n)) for comp in range(3)]
            w2 = [sum(cols[c][1][comp] * rinv[c, j] for c in range(n)) for comp in range(3)]
            wcols.append((r53(tuple(w1)), r53(tuple(w2))))
        cols53 = [(r53(a), r53(b)) for a, b in cols]
        with mp.workprec(53):
            nr, nt = mp.mpf(1), r53(-dr / r)
            fac = r53(w * r * r * 2 * mp.pi)
            for key, (xm, xn) in (("q", outh), ("p", outj)):
                rows = []
                for k in range(n0, nmax + 1):
                    rows.append((r53(xm[k]), r53(xn[k])))
                for k in range(n0, nmax + 1):
                    rows.append((r53(xn[k]), r53(xm[k])))
                for i in range(n):
                    a1, a2 = rows[i]
                    for j in range(n):
                        b1, b2 = wcols[j]
                        acc[key][i, j] += (
                            -1j * fac * (triple(nr, nt, a1, b1) + triple(nr, nt, a2, b2))
                        )
                        c1, c2 = cols53[j]
                        accp[key][i, j] += (
                            -1j * fac * (triple(nr, nt, a1, c1) + triple(nr, nt, a2, c2))
                        )
    with mp.workprec(53):
        t_basis = -(acc["p"] * mp.inverse(acc["q"]))
        t_plain = -(accp["p"] * mp.inverse(accp["q"]))
    print(
        f"x={x} eps={eps} m={m} N={nmax}: 53-bit plain {mp.nstr(err(t_plain, t), 3)}; "
        f"53-bit in the orthonormalized basis {mp.nstr(err(t_basis, t), 3)}"
    )


if __name__ == "__main__":
    main()
