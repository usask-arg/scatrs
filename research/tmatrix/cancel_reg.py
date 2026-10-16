"""The cancellation left in the Q integrals after rule v2 (powers ≤ 0 dropped for n ≠ k): per
element Σ|terms| / |element| at high precision, worst per row order n.

Usage: python cancel_reg.py x eps s_re s_im m nmax ntheta
"""

import sys

import mpmath as mp
from ebcm import angular, gauss_legendre, spheroid
from laurent import product, radial_j, radial_y
from regular import YKIND, jn_radial, radial
from terms import element_terms

x, eps, sr, si = map(float, sys.argv[1:5])
m, nmax, nth = map(int, sys.argv[5:8])
mp.mp.dps = 40
s = mp.mpc(sr, si)
a = mp.mpf(x) * mp.mpf(eps) ** (mp.mpf(1) / 3)
c = a / mp.mpf(eps)
beta = 1 / (c * c) - 1 / (a * a)
shape = spheroid(x, eps)
nodes, weights = gauss_legendre(nth)
n0 = max(m, 1)
kk = nmax - n0 + 1
val = {}
absv = {}
plain_abs = {}
series = {}
for ct, w in zip(nodes, weights, strict=True):
    st = mp.sqrt(1 - ct * ct)
    r, _ = shape(ct, st)
    d, pi_, tau = angular(m, nmax, ct, st)
    rad = {}
    for rt in (0, 1):
        for ctp in (0, 1):
            for n in range(n0, nmax + 1):
                for k in range(n0, nmax + 1):
                    key = (rt, ctp, n, k)
                    for ka, kb, power, coeff in element_terms(
                        n, k, rt, ctp, s, d, pi_, tau, beta, st, ct
                    ):
                        for kind, order, z in ((ka, n, r), (kb, k, s * r)):
                            if (kind, order) not in rad:
                                rad[(kind, order)] = radial(
                                    kind if kind in ("j", "j/z", "Dj") else YKIND[kind], order, z
                                )
                        if (ka, n, "jn") not in rad:
                            rad[(ka, n, "jn")] = jn_radial(ka, n, r)
                        rj = rad[(kb, k)]
                        jpart = rad[(ka, n, "jn")] * rj * r**power
                        yfull = rad[(ka, n)] * rj * r**power
                        d_ = abs(n - k)
                        forms = {"a": yfull}
                        if n != k:
                            skey = (YKIND[ka], kb, n, k, power)
                            if skey not in series:
                                series[skey] = product(
                                    radial_y(n, YKIND[ka], nmax + 30),
                                    radial_j(k, kb, s, nmax + 30),
                                    power,
                                )
                            neg0 = sum(
                                (cf * r**p for p, cf in series[skey].items() if p <= 0), mp.mpf(0)
                            )
                            forms["c"] = yfull - neg0
                            if d_ >= 3:
                                neg = sum(
                                    (cf * r**p for p, cf in series[skey].items() if p < 0),
                                    mp.mpf(0),
                                )
                                forms["b"] = yfull - neg
                        for f, ypart in forms.items():
                            val[(key, f)] = val.get((key, f), 0) + w * coeff * (jpart + 1j * ypart)
                            absv[(key, f)] = absv.get((key, f), 0) + abs(w * coeff) * (
                                abs(jpart) + abs(ypart)
                            )
worst_by_n = {}
for (key, f), v in val.items():
    if f != "a":
        continue
    rt, ctp, n, k = key
    if abs(v) < mp.mpf(10) ** -30 * absv[(key, f)]:
        continue
    ratios = {g: absv[(key, g)] / abs(v) for g in "abc" if (key, g) in absv}
    best = min(ratios.values())
    cur = worst_by_n.get(n)
    if cur is None or best > cur[0]:
        worst_by_n[n] = (best, ratios, (rt, ctp, k))
for n in sorted(worst_by_n):
    best, ratios, where = worst_by_n[n]
    detail = ", ".join(f"{g} {mp.nstr(v, 2)}" for g, v in sorted(ratios.items()))
    print(f"n={n:3d}: worst best-of cancellation {mp.nstr(best, 2):>8} ({detail}) at {where}")
