"""Check: for every element with n ≠ k (any block), the terms of the y_n products of power p ≤ 0
integrate to zero as a group (rule v2); the group p ≤ d does not (so positive powers stay).

Usage: python laurent_rule2.py eps m nmax
"""

import sys

import mpmath as mp
from ebcm import angular, gauss_legendre, spheroid
from laurent import product, radial_j, radial_y
from terms import element_terms

eps, m, nmax = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
x = 5.0
mp.mp.dps = 50
s = mp.mpc(1.5, 0.01)
a = mp.mpf(x) * mp.mpf(eps) ** (mp.mpf(1) / 3)
c = a / mp.mpf(eps)
beta = 1 / (c * c) - 1 / (a * a)
shape = spheroid(x, eps)
nodes, weights = gauss_legendre(2 * nmax + 60)
pre = []
for ct, w in zip(nodes, weights, strict=True):
    st = mp.sqrt(1 - ct * ct)
    r, _ = shape(ct, st)
    pre.append((w, st, ct, r, angular(m, nmax, ct, st)))
ykind = {"h": "y", "h/x": "y/x", "Dh": "Dy"}
n0 = max(m, 1)
fail_zero, sharp = [], 0
total = 0
for rt in (0, 1):
    for ctp in (0, 1):
        for n in range(n0, nmax + 1):
            for k in range(n0, nmax + 1):
                d = abs(n - k)
                if d == 0:
                    continue
                g_lo, s_lo, g_hi, s_hi = 0, 0, 0, 0
                for w, st, ct, r, (dd, pi_, tau) in pre:
                    for ka, kb, power, coeff in element_terms(
                        n, k, rt, ctp, s, dd, pi_, tau, beta, st, ct
                    ):
                        ser = product(
                            radial_y(n, ykind[ka], n + 4), radial_j(k, kb, s, n + 4), power
                        )
                        for p, cf in ser.items():
                            v = w * coeff * cf * r**p
                            if p <= 0:
                                g_lo += v
                                s_lo += abs(v)
                            if p <= d:
                                g_hi += v
                                s_hi += abs(v)
                if s_hi == 0:
                    continue  # a structural zero
                total += 1
                if s_lo > 0 and abs(g_lo) > mp.mpf(10) ** -35 * s_lo:
                    fail_zero.append((rt, ctp, n, k, float(abs(g_lo) / s_lo)))
                if abs(g_hi) > mp.mpf(10) ** -20 * s_hi:
                    sharp += 1
print(
    f"eps={eps} m={m}: {total} elements with n != k; group p <= 0 non-zero in "
    f"{len(fail_zero)} {fail_zero[:5]}; group p <= d non-zero in {sharp}"
)
