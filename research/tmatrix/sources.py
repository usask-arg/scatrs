"""Null-field equations tested with distributed sources (NFM-DS, Doicu, Wriedt & Eremin 2000).

The null-field conditions hold for any complete set of outgoing test fields singular inside the
particle. With test fields X_i = low-order outgoing multipoles centred at points z_i on the axis,
the internal coefficients solve Q_D c = S_D a, with Q_D = −i B(X_i, Y_k) and
S_D = −i B(X_i, RgX_n(k)) (surface-independent: a regular-wave translation coefficient), so
T = −P Q_D⁻¹ S_D, P as in the EBCM. With multipoles at the origin S_D = I and this is the EBCM.

Measures: S_D for origin tests (should be I); T_DS against the EBCM T; the cancellation in the
Q_D integrals; and the f64 floor of this formulation (P, Q_D, S_D exact, rounded to 53 bits,
solved at 53 bits) against T_DS itself, beside the EBCM's.

Usage: python sources.py x eps s_re s_im m nmax ntheta nsources [span]
  nsources points evenly spread over span × the focal segment (prolate, eps < 1); each carries
  K/nsources orders, K = nmax − max(m,1) + 1.
"""

import sys
import time

import mpmath as mp
from columns import err, rounded
from ebcm import angular, gauss_legendre, qmatrices, spheroid, vswf


def triple(nr, nt, a, b):
    return nr * (a[1] * b[2] - a[2] * b[1]) + nt * (a[2] * b[0] - a[0] * b[2])


def rotate(v, delta):
    """Components in the local spherical frame (centred on the axis) to the global frame at the
    same point; delta = θ − θ' between the two radial directions."""
    cd, sd = mp.cos(delta), mp.sin(delta)
    return (v[0] * cd + v[1] * sd, -v[0] * sd + v[1] * cd, v[2])


def test_rows(m, centres, orders_per, r, ct, st):
    """(X, ∇×X/k) for every test field at a surface point (r, θ): type 1 (M, N) then type 2 (N, M)
    of each centre's orders."""
    rho, z = r * st, r * ct
    theta = mp.atan2(st, ct)
    t1, t2 = [], []
    n0 = max(m, 1)
    for zc in centres:
        rr = mp.sqrt(rho * rho + (z - zc) ** 2)
        c2, s2 = (z - zc) / rr, rho / rr
        delta = theta - mp.atan2(s2, c2)
        lmax = n0 + orders_per - 1
        d, pi_, tau = angular(m, lmax, c2, s2)
        hm, hn = vswf(m, lmax, rr, "h", d, pi_, tau, flip=True)
        for order in range(n0, lmax + 1):
            mm, nn = rotate(hm[order], delta), rotate(hn[order], delta)
            t1.append((mm, nn))
            t2.append((nn, mm))
    return t1 + t2


def integrate(x, eps, s, m, nmax, nth, rows_at, cols_kind):
    """Σ −i [n̂·(A1×B1) + n̂·(A2×B2)] over the particle, rows from rows_at(r, ct, st); columns
    internal (k₁ = s) or incident regular (k), arranged (∇×Y/k, Y). Also Σ|terms|."""
    shape = spheroid(x, eps)
    nodes, weights = gauss_legendre(nth)
    n0 = max(m, 1)
    kk = nmax - n0 + 1
    n = 2 * kk
    val, absv = mp.matrix(n, n), mp.matrix(n, n)
    for ct, w in zip(nodes, weights, strict=False):
        st = mp.sqrt(1 - ct * ct)
        r, dr = shape(ct, st)
        d, pi_, tau = angular(m, nmax, ct, st)
        nr, nt = mp.mpf(1), -dr / r
        fac = w * r * r * 2 * mp.pi
        if cols_kind == "internal":
            ym, yn = vswf(m, nmax, s * r, "j", d, pi_, tau, flip=False)
            sc = s
        else:
            ym, yn = vswf(m, nmax, r, "j", d, pi_, tau, flip=False)
            sc = 1
        cols = [(tuple(sc * c for c in yn[k]), ym[k]) for k in range(n0, nmax + 1)]
        cols += [(tuple(sc * c for c in ym[k]), yn[k]) for k in range(n0, nmax + 1)]
        rows = rows_at(r, ct, st)
        for i in range(n):
            a1, a2 = rows[i]
            for j in range(n):
                b1, b2 = cols[j]
                t1, t2 = triple(nr, nt, a1, b1), triple(nr, nt, a2, b2)
                val[i, j] += -1j * fac * (t1 + t2)
                absv[i, j] += abs(fac) * (abs(t1) + abs(t2))
    return val, absv


def worst_cancel(val, absv):
    w = mp.mpf(0)
    for i in range(val.rows):
        for j in range(val.cols):
            if abs(val[i, j]) > mp.mpf(10) ** -40 * absv[i, j]:
                w = max(w, absv[i, j] / abs(val[i, j]))
    return w


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth, nsrc = map(int, sys.argv[5:9])
    span = float(sys.argv[9]) if len(sys.argv) > 9 else 0.8
    t0 = time.time()
    mp.mp.dps = 50
    s = mp.mpc(sr, si)
    n0 = max(m, 1)
    kk = nmax - n0 + 1
    assert kk % nsrc == 0, "orders must split evenly over the sources"
    a = mp.mpf(x) * mp.mpf(eps) ** (mp.mpf(1) / 3)
    c = a / mp.mpf(eps)
    f = mp.sqrt(abs(c * c - a * a))
    centres = (
        [mp.mpf(0)]
        if nsrc == 1
        else [span * f * (2 * mp.mpf(i) / (nsrc - 1) - 1) for i in range(nsrc)]
    )
    per = kk // nsrc

    def rows_ds(r, ct, st):
        return test_rows(m, centres, per, r, ct, st)

    # The EBCM reference and P.
    q, p = qmatrices(x, eps, s, m, nmax, nth)
    t_ebcm = -(p * mp.inverse(q))
    qd, qd_abs = integrate(x, eps, s, m, nmax, nth, rows_ds, "internal")
    # S_D is surface-independent: on a sphere of radius 2x (enclosing every source), where its
    # integrands have no cancellation. With origin tests S_D = −I, so T = P Q_D⁻¹ S_D.
    sd, sd_abs = integrate(2 * x, 1.0, s, m, nmax, 3 * nth, rows_ds, "incident")
    t_ds = p * mp.inverse(qd) * sd
    with mp.workprec(53):
        t53 = rounded(p) * mp.inverse(rounded(qd)) * rounded(sd)
        t53_q = rounded(p) * mp.inverse(rounded(qd)) * sd  # S_D exact: Q_D's share of the floor
        t53_ebcm = -(rounded(p) * mp.inverse(rounded(q)))
    from columns import skeel

    sk_d, sk_e = skeel(qd), skeel(q)
    print(
        f"x={x:g} eps={eps:g} m={m} N={nmax} sources={nsrc}×{per} span={span}: "
        f"|T_DS − T_EBCM| {mp.nstr(err(t_ds, t_ebcm), 2)}; "
        f"cancellation Q_D {mp.nstr(worst_cancel(qd, qd_abs), 2)}, "
        f"S_D {mp.nstr(worst_cancel(sd, sd_abs), 2)}; "
        f"f64 floor: EBCM {mp.nstr(err(t53_ebcm, t_ebcm), 2)}, "
        f"DS {mp.nstr(err(t53, t_ds), 2)} (S_D exact {mp.nstr(err(t53_q, t_ds), 2)}); Skeel EBCM "
        f"{mp.nstr(sk_e, 2)}, Q_D {mp.nstr(sk_d, 2)} ({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
