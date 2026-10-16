"""A reference EBCM for axisymmetric particles in mpmath, at any working precision.

The Q and RgQ matrices of one azimuthal mode m come from their definition as surface integrals of
cross products of vector spherical wave functions (Mishchenko, Travis & Lacis 2002, section 5.8),
evaluated component by component at Gauss–Legendre nodes in cos θ, with no simplification of the
integrands. With `mp.prec = 53` every operation rounds as in f64 while the special functions stay
correctly rounded, so the losses measured there are the algorithm's own.

Conventions: e^{-iωt}; relative index s (Im s ≥ 0); k = 1, so lengths are size parameters and
x = k r_v is the equal-volume sphere's; spheroid aspect ratio eps = a/c (equatorial over polar
semi-axis), eps > 1 oblate.
"""

from __future__ import annotations

import functools

import mpmath as mp


@functools.cache
def _gauss_legendre(n, prec):
    with mp.workprec(prec + 20):
        xs, ws = [], []
        for i in range(1, n + 1):
            x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
            for _ in range(200):
                p0, p1 = mp.mpf(1), x
                for k in range(2, n + 1):
                    p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                dp = n * (x * p1 - p0) / (x * x - 1)
                dx = p1 / dp
                x -= dx
                if abs(dx) < mp.mpf(2) ** (-(prec + 15)):
                    break
            p0, p1 = mp.mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            xs.append(x)
            ws.append(2 / ((1 - x * x) * dp * dp))
        return tuple(xs), tuple(ws)


def gauss_legendre(n):
    """Nodes and weights on [-1, 1] at the current precision."""
    xs, ws = _gauss_legendre(n, mp.mp.prec)
    return [+v for v in xs], [+v for v in ws]


def spheroid(x, eps):
    """r(θ) and dr/dθ of the spheroid of equal-volume size parameter x and aspect ratio eps."""
    a = mp.mpf(x) * mp.mpf(eps) ** (mp.mpf(1) / 3)
    c = a / mp.mpf(eps)

    def shape(ct, st):
        den = c * c * st * st + a * a * ct * ct
        r = a * c / mp.sqrt(den)
        dr = -a * c * (c * c - a * a) * st * ct / (den * mp.sqrt(den))
        return r, dr

    return shape


def angular(m, nmax, ct, st):
    """Normalized d_{mn}, π_{mn} = m d/sinθ and τ_{mn} = dd/dθ for n = 0..nmax; for m < 0 from
    d_{−m,n} = (−1)^m d_{mn} (and τ likewise), π from its definition."""
    if m < 0:
        d, _, tau = angular(-m, nmax, ct, st)
        sign = (-1) ** (-m)
        d = [sign * v for v in d]
        tau = [sign * v for v in tau]
        pi_ = [m * v / st for v in d]
        return d, pi_, tau
    d = [mp.mpf(0)] * (nmax + 1)
    if m <= nmax:
        am = mp.mpf(1)
        for k in range(1, m + 1):
            am *= mp.sqrt(mp.mpf(2 * k - 1) / (2 * k))
        d[m] = am * st**m
        if m + 1 <= nmax:
            d[m + 1] = mp.sqrt(2 * m + 1) * ct * d[m]
        for n in range(m + 1, nmax):
            d[n + 1] = ((2 * n + 1) * ct * d[n] - mp.sqrt(n * n - m * m) * d[n - 1]) / mp.sqrt(
                (n + 1) ** 2 - m * m
            )
    tau = [mp.mpf(0)] * (nmax + 1)
    pi_ = [mp.mpf(0)] * (nmax + 1)
    for n in range(max(m, 1), nmax + 1):
        prev = d[n - 1] if n - 1 >= m else mp.mpf(0)
        tau[n] = (n * ct * d[n] - mp.sqrt(n * n - m * m) * prev) / st
        pi_[n] = m * d[n] / st
    return d, pi_, tau


def bessels(z, nmax, kind):
    """z_n(z) and [z z_n(z)]'/z for n = 0..nmax: j_n (kind 'j'), y_n ('y') or h_n^(1) ('h')."""
    j = [mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + mp.mpf(1) / 2, z) for n in range(nmax + 1)]
    if kind == "y":
        zn = [mp.sqrt(mp.pi / (2 * z)) * mp.bessely(n + mp.mpf(1) / 2, z) for n in range(nmax + 1)]
    elif kind == "h":
        y = [mp.sqrt(mp.pi / (2 * z)) * mp.bessely(n + mp.mpf(1) / 2, z) for n in range(nmax + 1)]
        zn = [jj + 1j * yy for jj, yy in zip(j, y, strict=False)]
    else:
        zn = j
    # [z z_n]' / z = z_{n-1} − n z_n / z, with z_{-1} from z_0: j_{-1} = cos z/z, y_{-1} = sin z/z.
    if kind == "h":
        zm1 = (mp.cos(z) + 1j * mp.sin(z)) / z
    elif kind == "y":
        zm1 = mp.sin(z) / z
    else:
        zm1 = mp.cos(z) / z
    der = [None] * (nmax + 1)
    for n in range(nmax + 1):
        prev = zm1 if n == 0 else zn[n - 1]
        der[n] = prev - n * zn[n] / z
    return zn, der


def vswf(m, nmax, z, kind, d, pi_, tau, flip):
    """(r, θ, φ) components of M_mn and N_mn at argument z for n = 0..nmax (φ factor dropped).

    With `flip`, π → −π: the (−1)^m X_{−m n} of the EBCM integrals.
    """
    zn, der = bessels(z, nmax, kind)
    out_m, out_n = [None] * (nmax + 1), [None] * (nmax + 1)
    sgn = -1 if flip else 1
    for n in range(max(m, 1), nmax + 1):
        g = mp.sqrt(mp.mpf(2 * n + 1) / (4 * mp.pi * n * (n + 1)))
        p = sgn * pi_[n]
        out_m[n] = (0, g * zn[n] * 1j * p, -g * zn[n] * tau[n])
        out_n[n] = (
            g * n * (n + 1) * zn[n] / z * d[n],
            g * der[n] * tau[n],
            g * der[n] * 1j * p,
        )
    return out_m, out_n


def qmatrices(x, eps, s, m, nmax, ntheta, perturb=None):
    """Q and RgQ of mode m (2K × 2K, K = nmax − max(m,1) + 1, blocks [11 12; 21 22])."""
    s = mp.mpc(s)
    shape = spheroid(x, eps)
    nodes, weights = gauss_legendre(ntheta)
    n0 = max(abs(m), 1)
    kk = nmax - n0 + 1
    # J[(a, b, outgoing)] accumulates K×K.
    acc = {}
    for key in [(a, b, h) for a in (1, 2) for b in (1, 2) for h in (True, False)]:
        acc[key] = mp.matrix(kk, kk)
    for ct, w in zip(nodes, weights, strict=False):
        st = mp.sqrt(1 - ct * ct)
        r, dr = shape(ct, st)
        d, pi_, tau = angular(m, nmax, ct, st)
        nvec = (mp.mpf(1), -dr / r, mp.mpf(0))
        fac = w * r * r * 2 * mp.pi
        inner_m, inner_n = vswf(m, nmax, s * r, "j", d, pi_, tau, flip=False)
        if perturb is not None:
            inner_m, inner_n = perturb("inner", inner_m), perturb("inner", inner_n)
        for out in (True, False):
            om, on = vswf(m, nmax, r, "h" if out else "j", d, pi_, tau, flip=True)
            if perturb is not None:
                kind = "out" if out else "reg"
                om, on = perturb(kind, om), perturb(kind, on)
            for a, xs in ((1, om), (2, on)):
                for b, ys in ((1, inner_m), (2, inner_n)):
                    mat = acc[(a, b, out)]
                    for i in range(n0, nmax + 1):
                        A = xs[i]
                        for jj in range(n0, nmax + 1):
                            B = ys[jj]
                            trip = nvec[0] * (A[1] * B[2] - A[2] * B[1]) + nvec[1] * (
                                A[2] * B[0] - A[0] * B[2]
                            )
                            mat[i - n0, jj - n0] += fac * trip

    def assemble(out):
        J = {(a, b): acc[(a, b, out)] for a in (1, 2) for b in (1, 2)}
        # Q¹¹ = −i k k₁ J¹² − i k² J²¹ (and likewise), with k = 1, k₁ = s: checked by the sphere
        # identity T¹¹ = −bₙ, T²² = −aₙ (check_sphere.py).
        q11 = -1j * s * J[1, 2] - 1j * J[2, 1]
        q12 = -1j * s * J[1, 1] - 1j * J[2, 2]
        q21 = -1j * s * J[2, 2] - 1j * J[1, 1]
        q22 = -1j * s * J[2, 1] - 1j * J[1, 2]
        q = mp.matrix(2 * kk, 2 * kk)
        for i in range(kk):
            for jj in range(kk):
                q[i, jj] = q11[i, jj]
                q[i, kk + jj] = q12[i, jj]
                q[kk + i, jj] = q21[i, jj]
                q[kk + i, kk + jj] = q22[i, jj]
        return q

    return assemble(True), assemble(False)


def tmatrix(x, eps, s, m, nmax, ntheta):
    q, rgq = qmatrices(x, eps, s, m, nmax, ntheta)
    return -(rgq * mp.inverse(q)), q, rgq


def mie_ab(x, s, nmax):
    """Bohren & Huffman's a_n, b_n (n = 1..nmax) at high precision, for the sphere check."""
    s = mp.mpc(s)
    a, b = [], []
    for n in range(1, nmax + 1):

        def psi(z, k=n):
            return z * mp.sqrt(mp.pi / (2 * z)) * mp.besselj(k + mp.mpf(1) / 2, z)

        def xi(z, k=n):
            return (
                z
                * mp.sqrt(mp.pi / (2 * z))
                * (mp.besselj(k + mp.mpf(1) / 2, z) + 1j * mp.bessely(k + mp.mpf(1) / 2, z))
            )

        dpsi = lambda z: mp.diff(psi, z)  # noqa: E731
        dxi = lambda z: mp.diff(xi, z)  # noqa: E731
        mx = s * x
        a.append(
            (s * psi(mx) * dpsi(x) - psi(x) * dpsi(mx)) / (s * psi(mx) * dxi(x) - xi(x) * dpsi(mx))
        )
        b.append(
            (psi(mx) * dpsi(x) - s * psi(x) * dpsi(mx)) / (psi(mx) * dxi(x) - s * xi(x) * dpsi(mx))
        )
    return a, b
