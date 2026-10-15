"""The singular part of the Q integrands (the negative powers of x in the Laurent series of the
y_n-part products) and its integral, per element, with the terms of terms.py.

Usage: python laurent_q.py eps m nmax [per_power]
"""

import sys

import mpmath as mp
from ebcm import angular, gauss_legendre, spheroid
from laurent import product, radial_j, radial_y
from terms import element_terms

eps, m, nmax = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
per_power = len(sys.argv) > 5
PMAX = int(sys.argv[4]) if len(sys.argv) > 4 else -1
x = 5.0
mp.mp.dps = 50
s = mp.mpc(1.5, 0.01)
a = mp.mpf(x) * mp.mpf(eps) ** (mp.mpf(1) / 3)
c = a / mp.mpf(eps)
beta = 1 / (c * c) - 1 / (a * a)
shape = spheroid(x, eps)
nodes, weights = gauss_legendre(2 * nmax + 40)
pre = []
for ct, w in zip(nodes, weights, strict=True):
    st = mp.sqrt(1 - ct * ct)
    r, _ = shape(ct, st)
    pre.append((ct, w, st, r, angular(m, nmax, ct, st)))
ykind = {"h": "y", "h/x": "y/x", "Dh": "Dy"}
n0 = max(m, 1)
for rt in (0, 1):
    for ctp in (0, 1):
        worst, nonzero = (0, None), []
        for n in range(n0, nmax + 1):
            for k in range(n0, nmax + 1):
                moments, scale = {}, {}
                for ct, w, st, r, (d, pi_, tau) in pre:
                    for ka, kb, power, coeff in element_terms(
                        n, k, rt, ctp, s, d, pi_, tau, beta, st, ct
                    ):
                        # The y-part: h = j + i y.
                        ser = product(
                            radial_y(n, ykind[ka], n + 3), radial_j(k, kb, s, n + 3), power
                        )
                        for p, cf in ser.items():
                            if p > PMAX:
                                continue
                            v = w * 1j * coeff * cf * r**p
                            key = p if per_power else 0
                            moments[key] = moments.get(key, 0) + v
                            scale[key] = scale.get(key, 0) + abs(v)
                for key, mom in moments.items():
                    if scale[key] == 0:
                        continue
                    rel = abs(mom) / scale[key]
                    if rel > 1e-30:
                        nonzero.append((n, k, key))
                    if rel > worst[0]:
                        worst = (rel, (n, k, key))
        print(
            f"Q block ({rt},{ctp}): worst |∫ singular part| / ∫|…| "
            f"{mp.nstr(worst[0], 3)} at {worst[1]};"
            f" non-zero: {nonzero[:10]}{' …' if len(nonzero) > 10 else ''}"
        )
