"""The cancellation in the plain Q integrals for other axisymmetric shapes: per row order n, the
worst over the columns of Σ|elementary products| / |Q_nk| (as cancel.py), for a spheroid, a
finite cylinder (the quadrature split at the edges) and a Chebyshev particle, at the same
equal-volume size. Somerville et al. (2012) found no cancellation of the electrostatic kind for
cylinders and Chebyshev particles in the small-size limit; this measures it at larger sizes,
where the f64 floor of the spheroid lies.

With a last argument `h` the internal functions are outgoing, h_k(s r) for j_k(s r): the
integrals a coated particle's shell adds (outgoing in the shell against outgoing outside).

Usage: python cancel_shapes.py shape param x s_re s_im m nmax ntheta "n1,n2,..." [h]
  shape: spheroid (param = aspect ratio a/c), cylinder (param = half-length / radius),
         chebyshev (param = "order:deformation", r = r₀(1 + δ cos(order θ)))
"""

import sys

import mpmath as mp
from ebcm import angular, gauss_legendre, spheroid, vswf


def nodes_on(pieces, n_total):
    """Gauss–Legendre nodes and weights in cos θ on each piece [lo, hi], split ∝ length."""
    nodes, weights = [], []
    for lo, hi in pieces:
        n = max(8, int(mp.nint(n_total * (hi - lo) / 2)))
        xs, ws = gauss_legendre(n)
        half, mid = (hi - lo) / 2, (hi + lo) / 2
        nodes += [mid + half * v for v in xs]
        weights += [half * w for w in ws]
    return nodes, weights


def cylinder(x, h):
    """r(θ), dr/dθ and the pieces of a cylinder of half-length h·a, radius a, equal volume x."""
    a = mp.mpf(x) * (mp.mpf(2) / (3 * mp.mpf(h))) ** (mp.mpf(1) / 3)
    length = mp.mpf(h) * a
    ce = length / mp.sqrt(length * length + a * a)  # cos θ at the top edge

    def shape(ct, st):
        if abs(ct) >= ce:
            r = length / abs(ct)
            return r, length * st / (ct * ct) * mp.sign(ct)
        return a / st, -a * ct / (st * st)

    return shape, [(-1, -ce), (-ce, ce), (ce, 1)]


def chebyshev(x, order, delta):
    order, delta = int(order), mp.mpf(delta)
    vol = mp.quad(lambda t: (1 + delta * mp.cos(order * t)) ** 3 * mp.sin(t), [0, mp.pi])
    r0 = mp.mpf(x) * (2 / vol) ** (mp.mpf(1) / 3)

    def shape(ct, st):
        t = mp.atan2(st, ct)
        return r0 * (1 + delta * mp.cos(order * t)), -r0 * delta * order * mp.sin(order * t)

    return shape


def main():
    name, param = sys.argv[1], sys.argv[2]
    x, sr, si = map(float, sys.argv[3:6])
    m, nmax, nth = map(int, sys.argv[6:9])
    rows = [int(v) for v in sys.argv[9].split(",")]
    inner = sys.argv[10] if len(sys.argv) > 10 else "j"
    mp.mp.dps = 40
    s = mp.mpc(sr, si)
    if name == "spheroid":
        shape, pieces = spheroid(x, float(param)), [(-1, 1)]
    elif name == "cylinder":
        shape, pieces = cylinder(x, float(param))
    else:
        order, delta = param.split(":")
        shape, pieces = chebyshev(x, order, delta), [(-1, 1)]
    nodes, weights = nodes_on(pieces, nth)
    n0 = max(m, 1)
    blocks = {
        (1, 1): ((1, 2, s), (2, 1, 1)),
        (1, 2): ((1, 1, s), (2, 2, 1)),
        (2, 1): ((2, 2, s), (1, 1, 1)),
        (2, 2): ((2, 1, s), (1, 2, 1)),
    }
    val, absv = {}, {}
    rmin, rmax = mp.inf, 0
    for ct, w in zip(nodes, weights, strict=True):
        st = mp.sqrt(1 - ct * ct)
        r, dr = shape(ct, st)
        rmin, rmax = min(rmin, r), max(rmax, r)
        d, pi_, tau = angular(m, nmax, ct, st)
        nth_ = -dr / r
        fac = w * r * r * 2 * mp.pi
        ym, yn = vswf(m, nmax, s * r, inner, d, pi_, tau, flip=False)
        xm, xn = vswf(m, nmax, r, "h", d, pi_, tau, flip=True)
        xx = {1: xm, 2: xn}
        yy = {1: ym, 2: yn}
        for b, terms in blocks.items():
            for a1, b1, wgt in terms:
                for i in rows:
                    aa = xx[a1][i]
                    for j in range(n0, nmax + 1):
                        bb = yy[b1][j]
                        p1, p2 = aa[1] * bb[2], aa[2] * bb[1]
                        p3, p4 = aa[2] * bb[0], aa[0] * bb[2]
                        key = (b, i, j)
                        val[key] = val.get(key, 0) - 1j * wgt * fac * ((p1 - p2) + nth_ * (p3 - p4))
                        absv[key] = absv.get(key, 0) + abs(wgt * fac) * (
                            abs(p1) + abs(p2) + abs(nth_) * (abs(p3) + abs(p4))
                        )
    print(
        f"{name} {param} inner={inner} x={x} s={sr}+{si}i m={m} N={nmax} nodes={len(nodes)}"
        f" x_min={mp.nstr(rmin, 3)} x_max={mp.nstr(rmax, 3)}"
    )
    for i in rows:
        worst = (mp.mpf(0), None)
        for (b, ii, j), v in val.items():
            if ii != i or abs(v) < mp.mpf(10) ** -30 * absv[(b, ii, j)]:
                continue
            ratio = absv[(b, ii, j)] / abs(v)
            if ratio > worst[0]:
                worst = (ratio, (b, j))
        (b, k) = worst[1]
        print(f"  n={i:3d}: worst Σ|terms|/|Q| {mp.nstr(worst[0], 3):>9} at block {b} k={k}")


if __name__ == "__main__":
    main()
