"""Cancellation-free Q integrals at 53 bits: drop the non-positive powers of the y_n products.

The terms of power ≤ 0 of the y_n-part products' Laurent series integrate to exactly zero in every
element with n ≠ k (laurent_q.py with PMAX = 0), in every block. They are dropped there; the
diagonal keeps the full integrand. The integrand then uses R⁺ = R − R⁻, evaluated at each node
either from its own Laurent series (positive powers, while its terms decrease) or as R − R⁻
with R from the Bessel functions (when the singular part does not dominate).

Compares the 53-bit elements and T with the exact ones and with the plain 53-bit EBCM.

Usage: python regular.py x eps s_re s_im m nmax ntheta
"""

import sys
import time

import mpmath as mp
from columns import err
from ebcm import angular, gauss_legendre, qmatrices, spheroid
from laurent import product, radial_j, radial_y
from terms import element_terms

EXACT_R = len(sys.argv) > 8 and sys.argv[8] == "exact"
YKIND = {"h": "y", "h/x": "y/x", "Dh": "Dy"}


def sph(n, z, kind):
    f = mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + 0.5, z)
    if kind == "y":
        f = mp.sqrt(mp.pi / (2 * z)) * mp.bessely(n + 0.5, z)
    return f


def radial(kind, n, z):
    base = {"h": "y", "h/x": "y", "Dh": "y", "y": "y", "y/x": "y", "Dy": "y"}.get(kind, "j")
    if kind in ("Dh", "Dy", "Dj"):
        return sph(n - 1, z, base) - n * sph(n, z, base) / z
    f = sph(n, z, base)
    return f / z if "/" in kind else f


def jn_radial(kind, n, x):
    """The j_n part of an outgoing radial factor."""
    if kind == "Dh":
        return sph(n - 1, x, "j") - n * sph(n, x, "j") / x
    f = sph(n, x, "j")
    return f / x if kind == "h/x" else f


def keeps_singular(n, k, rt, ct_):
    """Only the diagonal n = k keeps the non-positive powers: for n ≠ k they integrate to zero."""
    return n == k


def regular_part(series, x, full):
    """R⁺ from the Laurent series (positive powers) or as full − R⁻, whichever is safe."""
    neg = sum((c * x**p for p, c in series.items() if p <= 0), mp.mpf(0))
    pos_terms = [c * x**p for p, c in sorted(series.items()) if p > 0]
    pos = sum(pos_terms, mp.mpf(0))
    # The series is usable when its terms decrease to well below the sum.
    tail = abs(pos_terms[-1]) if pos_terms else 0
    biggest = max((abs(t) for t in pos_terms), default=0)
    series_ok = tail <= 1e-18 * abs(pos) and biggest <= 4 * abs(pos)
    if series_ok:
        return pos
    return full - neg


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    t0 = time.time()
    mp.mp.dps = 50
    s = mp.mpc(sr, si)
    q, p = qmatrices(x, eps, s, m, nmax, nth)
    t = -(p * mp.inverse(q))
    n0 = max(m, 1)
    kk = nmax - n0 + 1
    nterms = nmax + 40
    with mp.workprec(53):
        s53 = +s
        a = mp.mpf(x) * mp.mpf(eps) ** (mp.mpf(1) / 3)
        c = a / mp.mpf(eps)
        beta = 1 / (c * c) - 1 / (a * a)
        shape = spheroid(x, eps)
        nodes, weights = gauss_legendre(nth)
        q53 = mp.matrix(2 * kk, 2 * kk)
        series_cache = {}
        for ct, w in zip(nodes, weights, strict=True):
            st = mp.sqrt(1 - ct * ct)
            r, _ = shape(ct, st)
            d, pi_, tau = angular(m, nmax, ct, st)
            cache = {}

            def cached(fn, *args, cache=cache):
                key = (fn.__name__, *args)
                if key not in cache:
                    cache[key] = fn(*args)
                return cache[key]

            for rt in (0, 1):
                for ctp in (0, 1):
                    for n in range(n0, nmax + 1):
                        for k in range(n0, nmax + 1):
                            keep = keeps_singular(n, k, rt, ctp)
                            tot = mp.mpc(0)
                            for ka, kb, power, coeff in element_terms(
                                n, k, rt, ctp, s53, d, pi_, tau, beta, st, ct
                            ):
                                rj = cached(radial, kb, k, s53 * r)
                                jpart = cached(jn_radial, ka, n, r) * rj * r**power
                                yfull = cached(radial, YKIND[ka], n, r) * rj * r**power
                                if keep:
                                    ypart = yfull
                                else:
                                    key = (YKIND[ka], kb, n, k, power)
                                    if key not in series_cache:
                                        series_cache[key] = product(
                                            radial_y(n, YKIND[ka], nterms),
                                            radial_j(k, kb, s53, nterms),
                                            power,
                                        )
                                    if EXACT_R:
                                        # R⁺ computed exactly, then rounded: isolates the
                                        # evaluation of R⁺ from the integration.
                                        with mp.workprec(200):
                                            ser = product(
                                                radial_y(n, YKIND[ka], nterms),
                                                radial_j(k, kb, s, nterms),
                                                power,
                                            )
                                            rx = mp.mpf(r)
                                            full = (
                                                radial(YKIND[ka], n, rx)
                                                * radial(kb, k, s * rx)
                                                * rx**power
                                            )
                                            neg = sum(
                                                (cf * rx**pp for pp, cf in ser.items() if pp <= 0),
                                                mp.mpf(0),
                                            )
                                            exact = full - neg
                                        ypart = +exact
                                    else:
                                        ypart = regular_part(series_cache[key], r, yfull)
                                tot += coeff * (jpart + 1j * ypart)
                            q53[rt * kk + n - n0, ctp * kk + k - n0] += w * tot
        p53 = mp.matrix([[+p[i, j] for j in range(2 * kk)] for i in range(2 * kk)])
        t53 = -(p53 * mp.inverse(q53))
        qplain, pplain = qmatrices(x, eps, s53, m, nmax, nth)
        tplain = -(pplain * mp.inverse(qplain))
        qr = mp.matrix([[+q[i, j] for j in range(2 * kk)] for i in range(2 * kk)])
        tfloor = -(p53 * mp.inverse(qr))
    worst_el = max(
        abs(q53[i, j] - q[i, j]) / abs(q[i, j])
        for i in range(2 * kk)
        for j in range(2 * kk)
        if abs(q[i, j]) > mp.mpf(10) ** -25 * max(abs(q[i, col]) for col in range(2 * kk))
    )
    ranked = []
    for i in range(2 * kk):
        rowmax = max(abs(q[i, col]) for col in range(2 * kk))
        for j in range(2 * kk):
            if abs(q[i, j]) > mp.mpf(10) ** -25 * rowmax:
                e = abs(q53[i, j] - q[i, j]) / abs(q[i, j])
                rt, n = divmod(i, kk)
                ctp, k = divmod(j, kk)
                ranked.append(
                    (e, (rt, ctp), n + n0, k + n0, keeps_singular(n + n0, k + n0, rt, ctp))
                )
    ranked.sort(key=lambda v: -v[0])
    for e, blk, n, k, keep in ranked[:6]:
        print(f"   element {blk} n={n} k={k} keep-singular={keep}: rel. error {mp.nstr(e, 2)}")
    print(
        f"x={x:g} eps={eps:g} m={m} N={nmax}: worst element rel. error {mp.nstr(worst_el, 2)}; "
        f"T error: plain {mp.nstr(err(tplain, t), 2)}, regularized {mp.nstr(err(t53, t), 2)}, "
        f"floor {mp.nstr(err(tfloor, t), 2)} ({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
