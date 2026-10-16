"""Regularized test functions: the outgoing test wave minus the outgoing k₁-wave of the same order.

Every element of Q is Q_nk = −i B(X_n, Y_k), with B(X, Y) = ∫_S n̂·(X × ∇×Y − Y × ∇×X), X the
outgoing wave at k (rows: (M_n, N_n) for the first K rows, (N_n, M_n) for the next, as (X, ∇×X/k))
and Y the regular internal wave at k₁ = s k. By the divergence theorem B(G, Y) = 0 for any k₁-field
G regular inside the particle, and B is surface-independent for the outgoing k₁-waves, which are
singular only at the origin. So with X̃_n = X_n(kr) − α_n X_n(k₁r):

    Q = Q̃ + diag(α) Q⁽ᵏ¹⁾,

where Q̃ integrates X̃ over the particle and Q⁽ᵏ¹⁾ = −i B(X_n(k₁r), Y_k) is evaluated on any sphere
(diagonal in the order). α_n = s^{n+1} (M rows) and s^{n+2} (N rows) cancel the leading singular
term of h_n(kr).

Checks Q̃ + diag(α) Q⁽ᵏ¹⁾ = Q at high precision, then measures the cancellation in Q̃ and the
53-bit error of T computed from 53-bit integrals of Q̃ (Q⁽ᵏ¹⁾ exact, rounded).

Usage: python subtract.py x eps s_re s_im m nmax ntheta
"""

import sys
import time

import mpmath as mp
from columns import err, rounded
from ebcm import angular, gauss_legendre, qmatrices, spheroid, vswf


def triple(nr, nt, a, b):
    return nr * (a[1] * b[2] - a[2] * b[1]) + nt * (a[2] * b[0] - a[0] * b[2])


def integrate(x, eps, s, m, nmax, nth, rows_of, sphere=False):
    """Σ_nodes of −i [n̂·(A1×B1) + n̂·(A2×B2)] with rows (A1, A2) = rows_of(...) per node and the
    internal columns (∇×Y/k, Y)... in the (curl Y, Y) arrangement; also Σ|terms|."""
    shape = spheroid(x, 1.0 if sphere else eps)
    nodes, weights = gauss_legendre(nth)
    n0 = max(m, 1)
    kk = nmax - n0 + 1
    n = 2 * kk
    val = mp.matrix(n, n)
    absv = mp.matrix(n, n)
    for ct, w in zip(nodes, weights, strict=False):
        st = mp.sqrt(1 - ct * ct)
        r, dr = shape(ct, st)
        d, pi_, tau = angular(m, nmax, ct, st)
        nr, nt = mp.mpf(1), -dr / r
        fac = w * r * r * 2 * mp.pi
        ym, yn = vswf(m, nmax, s * r, "j", d, pi_, tau, flip=False)
        cols = [(tuple(s * c for c in yn[k]), ym[k]) for k in range(n0, nmax + 1)]
        cols += [(tuple(s * c for c in ym[k]), yn[k]) for k in range(n0, nmax + 1)]
        rows = rows_of(r, d, pi_, tau)
        for i in range(n):
            a1, a2 = rows[i]
            for j in range(n):
                b1, b2 = cols[j]
                t1, t2 = triple(nr, nt, a1, b1), triple(nr, nt, a2, b2)
                val[i, j] += -1j * fac * (t1 + t2)
                absv[i, j] += abs(fac) * (abs(t1) + abs(t2))
    return val, absv


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    t0 = time.time()
    mp.mp.dps = 60
    s = mp.mpc(sr, si)
    n0 = max(m, 1)
    orders = list(range(n0, nmax + 1))
    alpha = [s ** (n + 1) for n in orders] + [s ** (n + 2) for n in orders]
    fit = len(sys.argv) > 8 and sys.argv[8] == "fit"

    def rows_k1(r, d, pi_, tau):
        hm, hn = vswf(m, nmax, s * r, "h", d, pi_, tau, flip=True)
        # (X, ∇×X/k) for X = M(k₁r): ∇×M(k₁r) = k₁ N(k₁r) = s N(k₁r) (k = 1).
        rows = [(hm[n], tuple(s * c for c in hn[n])) for n in orders]
        rows += [(hn[n], tuple(s * c for c in hm[n])) for n in orders]
        return rows

    def rows_tilde(r, d, pi_, tau):
        om, on = vswf(m, nmax, r, "h", d, pi_, tau, flip=True)
        plain = [(om[n], on[n]) for n in orders] + [(on[n], om[n]) for n in orders]
        k1 = rows_k1(r, d, pi_, tau)
        out = []
        for (a1, a2), (b1, b2), al in zip(plain, k1, alpha, strict=False):
            out.append(
                (
                    tuple(u - al * v for u, v in zip(a1, b1, strict=False)),
                    tuple(u - al * v for u, v in zip(a2, b2, strict=False)),
                )
            )
        return out

    def rows_plain(r, d, pi_, tau):
        om, on = vswf(m, nmax, r, "h", d, pi_, tau, flip=True)
        return [(om[n], on[n]) for n in orders] + [(on[n], om[n]) for n in orders]

    if fit:
        # α_n by least squares of α X_n(k₁r) to X_n(kr) over the surface nodes (both field and
        # curl components), weighted by the quadrature weight times r²: still exact for any α.
        shape = spheroid(x, eps)
        nodes, weights = gauss_legendre(nth)
        numer = [mp.mpc(0)] * len(alpha)
        denom = [mp.mpf(0)] * len(alpha)
        for ct, w in zip(nodes, weights, strict=False):
            st = mp.sqrt(1 - ct * ct)
            r, _ = shape(ct, st)
            d, pi_, tau = angular(m, nmax, ct, st)
            plain = rows_plain(r, d, pi_, tau)
            k1 = rows_k1(r, d, pi_, tau)
            for i, ((a1, a2), (b1, b2)) in enumerate(zip(plain, k1, strict=False)):
                for u, v in list(zip(a1, b1, strict=False)) + list(zip(a2, b2, strict=False)):
                    numer[i] += w * r * r * mp.conj(v) * u
                    denom[i] += w * r * r * abs(v) ** 2
        alpha[:] = [nu / de for nu, de in zip(numer, denom, strict=False)]
    q, p = qmatrices(x, eps, s, m, nmax, nth)
    t = -(p * mp.inverse(q))
    q_plain, abs_plain = integrate(x, eps, s, m, nmax, nth, rows_plain)
    q_tilde, abs_tilde = integrate(x, eps, s, m, nmax, nth, rows_tilde)
    qk1_sphere, _ = integrate(1.0, 1.0, s, m, nmax, nmax + 10, rows_k1, sphere=True)
    n = q.rows
    recon = q_tilde + mp.diag(alpha) * qk1_sphere
    big = max(abs(q[i, j]) for i in range(n) for j in range(n))
    ident = max(abs(recon[i, j] - q[i, j]) for i in range(n) for j in range(n)) / big
    plain_ok = max(abs(q_plain[i, j] - q[i, j]) for i in range(n) for j in range(n)) / big
    offdiag = max(abs(qk1_sphere[i, j]) for i in range(n) for j in range(n) if i != j)

    def worst_cancel(val, absv):
        w = mp.mpf(0)
        for i in range(n):
            for j in range(n):
                if abs(val[i, j]) > mp.mpf(10) ** -40 * absv[i, j]:
                    w = max(w, absv[i, j] / abs(val[i, j]))
        return w

    # 53 bits: Q̃ integrated at 53 bits, plus the exact correction rounded.
    with mp.workprec(53):
        qt53, _ = integrate(x, eps, s, m, nmax, nth, rows_tilde)
        qp53, _ = integrate(x, eps, s, m, nmax, nth, rows_plain)
        corr = rounded(mp.diag(alpha) * qk1_sphere)
        q53 = qt53 + corr
        p53 = rounded(p)
        t_tilde = -(p53 * mp.inverse(q53))
        t_plain = -(p53 * mp.inverse(qp53))
        t_exact_q = -(p53 * mp.inverse(rounded(q)))
    print(
        f"x={x:g} eps={eps:g} m={m} N={nmax}: identity {mp.nstr(ident, 2)} (plain check "
        f"{mp.nstr(plain_ok, 2)}, Q(k1) off-diagonal {mp.nstr(offdiag, 2)}); worst cancellation "
        f"{mp.nstr(worst_cancel(q_plain, abs_plain), 2)} -> "
        f"{mp.nstr(worst_cancel(q_tilde, abs_tilde), 2)}; "
        f"53-bit T error: plain {mp.nstr(err(t_plain, t), 2)}, "
        f"regularized {mp.nstr(err(t_tilde, t), 2)}, "
        f"floor {mp.nstr(err(t_exact_q, t), 2)} ({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
