"""Is the conditioning of Q column-wise (the internal field functions nearly dependent over the
surface), and does a change of internal basis remove it?

T = −P Q⁻¹ = −(P R⁻¹)(Q R⁻¹)⁻¹ for any invertible R. With R from the QR factorization of the
stacked [P; Q] (computed at high precision), compare:
  Skeel(Q) and Skeel(Q R⁻¹);
  the error of T when the elements of (P, Q), or of (P R⁻¹, Q R⁻¹), are rounded to 53 bits and
  the system is then solved exactly (the representation floor of each form).

Usage: python columns.py x eps s_re s_im m nmax ntheta
"""

import sys

import mpmath as mp
from ebcm import qmatrices


def norm_inf(a):
    return max(sum(abs(a[i, j]) for j in range(a.cols)) for i in range(a.rows))


def skeel(q):
    qi = mp.inverse(q)
    n = q.rows
    aq = mp.matrix([[abs(q[i, j]) for j in range(n)] for i in range(n)])
    ai = mp.matrix([[abs(qi[i, j]) for j in range(n)] for i in range(n)])
    return norm_inf(ai * aq)


def rounded(a):
    with mp.workprec(53):
        return mp.matrix([[+a[i, j] for j in range(a.cols)] for i in range(a.rows)])


def err(t, ref):
    big = max(abs(ref[i, j]) for i in range(ref.rows) for j in range(ref.cols))
    return max(abs(t[i, j] - ref[i, j]) for i in range(ref.rows) for j in range(ref.cols)) / big


def qr_r(a):
    """R of the thin (complex) QR factorization of a, by modified Gram–Schmidt (twice)."""
    rows, cols = a.rows, a.cols
    v = [mp.matrix([a[i, j] for i in range(rows)]) for j in range(cols)]
    r = mp.matrix(cols, cols)
    for j in range(cols):
        for _ in range(2):
            for i in range(j):
                c = sum(mp.conj(v[i][k]) * v[j][k] for k in range(rows))
                r[i, j] += c
                for k in range(rows):
                    v[j][k] -= c * v[i][k]
        nrm = mp.sqrt(sum(abs(v[j][k]) ** 2 for k in range(rows)))
        r[j, j] = nrm
        for k in range(rows):
            v[j][k] /= nrm
    return r


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    mp.mp.dps = 60
    q, p = qmatrices(x, eps, mp.mpc(sr, si), m, nmax, nth)
    t = -(p * mp.inverse(q))
    n = q.rows
    stacked = mp.matrix(2 * n, n)
    for i in range(n):
        for j in range(n):
            stacked[i, j] = p[i, j]
            stacked[n + i, j] = q[i, j]
    ri = mp.inverse(qr_r(stacked))
    q2, p2 = q * ri, p * ri
    floor1 = err(-(rounded(p) * mp.inverse(rounded(q))), t)
    floor2 = err(-(rounded(p2) * mp.inverse(rounded(q2))), t)
    print(
        f"x={x} eps={eps} m={m} N={nmax}: "
        f"Skeel Q {mp.nstr(skeel(q), 3)} -> {mp.nstr(skeel(q2), 3)}; "
        f"f64 representation floor {mp.nstr(floor1, 3)} -> {mp.nstr(floor2, 3)}"
    )


if __name__ == "__main__":
    main()
