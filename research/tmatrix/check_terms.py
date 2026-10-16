"""terms.element_terms with the full Bessel functions must reproduce ebcm's Q."""

import mpmath as mp
from ebcm import angular, gauss_legendre, qmatrices, spheroid
from terms import element_terms

mp.mp.dps = 30
x, eps, m, nmax, nth = 4.0, 2.0, 1, 6, 60
s = mp.mpc(1.5, 0.01)
q, _ = qmatrices(x, eps, s, m, nmax, nth)
a = mp.mpf(x) * mp.mpf(eps) ** (mp.mpf(1) / 3)
c = a / mp.mpf(eps)
beta = 1 / (c * c) - 1 / (a * a)
shape = spheroid(x, eps)
nodes, weights = gauss_legendre(nth)
n0 = max(m, 1)
kk = nmax - n0 + 1


def sph(n, z, kind):
    f = mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + 0.5, z)
    if kind == "h":
        f += 1j * mp.sqrt(mp.pi / (2 * z)) * mp.bessely(n + 0.5, z)
    return f


def radial(kind, n, z):
    which = "h" if kind[0] in "hD" and kind != "Dj" else "j"
    if kind in ("Dh", "Dj"):
        return sph(n - 1, z, which) - n * sph(n, z, which) / z
    f = sph(n, z, which)
    return f / z if "/" in kind else f


worst = 0
for rt in (0, 1):
    for ctp in (0, 1):
        for n in range(n0, nmax + 1):
            for k in range(n0, nmax + 1):
                tot = 0
                for ct, w in zip(nodes, weights, strict=True):
                    st = mp.sqrt(1 - ct * ct)
                    r, _ = shape(ct, st)
                    d, pi_, tau = angular(m, nmax, ct, st)
                    for ka, kb, power, coeff in element_terms(
                        n, k, rt, ctp, s, d, pi_, tau, beta, st, ct
                    ):
                        tot += w * coeff * radial(ka, n, r) * radial(kb, k, s * r) * r**power
                ref = q[rt * kk + n - n0, ctp * kk + k - n0]
                worst = max(worst, abs(tot - ref) / max(abs(ref), mp.mpf(10) ** -12))
print("worst mismatch", mp.nstr(worst, 3))
