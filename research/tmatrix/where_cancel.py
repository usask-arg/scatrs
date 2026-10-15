"""Where the cancellation left in a Q element happens: among the terms at a node, or across the
quadrature nodes; and, after integration, between the term types (where an identity between
integrals, such as an integration by parts, would act) or within each term's own integral.

For an element Q = Σ_nodes Σ_i t_i (t_i the i-th term of terms.element_terms, weight included),
at high precision, in the plain form and in the form with the powers ≤ 0 removed (n ≠ k):
  A = Σ_nodes Σ_i (|j part| + |y part|)   (what f64 rounding scales with)
  B = Σ_nodes |Σ_i t_i|                  (the integrand summed pointwise first)
  T = Σ_i |Σ_nodes t_i|                  (each term type integrated first)
  V = |∫ ν terms| + |∫ the others|       (the split an integration by parts of r'(θ) acts on)
  C = |Q|
A/B is the loss among the terms at a node; B/C the loss across the nodes. T/C is the part of the
latter between the term types after integration, which an exact identity between the integrals
could remove; what is left (A/T) is inside the single terms' integrals.

Usage: python where_cancel.py x eps s_re s_im m nmax ntheta_full "n1,n2" "k1,k2,..."
"""

import sys

import mpmath as mp
from ebcm import angular, gauss_legendre, spheroid
from laurent import product, radial_j, radial_y
from regular import YKIND, jn_radial, radial
from terms import element_terms


def main():
    x, eps, sr, si = map(float, sys.argv[1:5])
    m, nmax, nth = map(int, sys.argv[5:8])
    ns = [int(v) for v in sys.argv[8].split(",")]
    ks = [int(v) for v in sys.argv[9].split(",")]
    mp.mp.dps = 40
    s = mp.mpc(sr, si)
    a = mp.mpf(x) * mp.mpf(eps) ** (mp.mpf(1) / 3)
    c = a / mp.mpf(eps)
    beta = 1 / (c * c) - 1 / (a * a)
    shape = spheroid(x, eps)
    nodes, weights = gauss_legendre(nth)
    keys = [(rt, ctp, n, k) for n in ns for k in ks for rt in (0, 1) for ctp in (0, 1)]
    forms = ("plain", "reg")
    integ = {(key, f): {} for key in keys for f in forms}
    absum = {(key, f): mp.mpf(0) for key in keys for f in forms}
    ptsum = {(key, f): mp.mpf(0) for key in keys for f in forms}
    nu = {}
    series = {}
    for ct, w in zip(nodes, weights, strict=True):
        st = mp.sqrt(1 - ct * ct)
        r, _ = shape(ct, st)
        d, pi_, tau = angular(m, nmax, ct, st)
        rad = {}

        def get(kind, order, z, tag, rad=rad):
            if (kind, order, tag) not in rad:
                rad[(kind, order, tag)] = (
                    jn_radial(kind, order, z) if tag == "jn" else radial(kind, order, z)
                )
            return rad[(kind, order, tag)]

        for key in keys:
            rt, ctp, n, k = key
            node = {f: mp.mpc(0) for f in forms}
            for i, (ka, kb, power, coeff) in enumerate(
                element_terms(n, k, rt, ctp, s, d, pi_, tau, beta, st, ct)
            ):
                nu[(key, i)] = power == 4
                rj = get(kb, k, s * r, "")
                jpart = get(ka, n, r, "jn") * rj * r**power
                yfull = get(YKIND[ka], n, r, "") * rj * r**power
                skey = (YKIND[ka], kb, n, k, power)
                if skey not in series:
                    series[skey] = product(
                        radial_y(n, YKIND[ka], nmax + 30), radial_j(k, kb, s, nmax + 30), power
                    )
                neg0 = sum((cf * r**p for p, cf in series[skey].items() if p <= 0), mp.mpf(0))
                for f, ypart in (("plain", yfull), ("reg", yfull - neg0)):
                    t = w * coeff * (jpart + 1j * ypart)
                    integ[(key, f)][i] = integ[(key, f)].get(i, 0) + t
                    absum[(key, f)] += abs(w * coeff) * (abs(jpart) + abs(ypart))
                    node[f] += t
            for f in forms:
                ptsum[(key, f)] += abs(node[f])
    print(f"x={x} eps={eps} s={sr}+{si}i m={m} N={nmax} nodes={nth}")
    print(
        f"{'row,col':>10} {'form':>5} {'|Q|':>8} {'A/C':>8} {'A/B node':>9}"
        f" {'B/C across':>10} {'T/C types':>10} {'V/C nu':>8} {'A/T within':>10}"
    )
    for key in keys:
        rt, ctp, n, k = key
        for f in forms:
            parts = integ[(key, f)]
            q = sum(parts.values(), mp.mpc(0))
            aa = absum[(key, f)]
            if abs(q) < mp.mpf(10) ** -25 * aa:
                continue
            tt = sum((abs(v) for v in parts.values()), mp.mpf(0))
            vv = abs(sum((v for i, v in parts.items() if nu[(key, i)]), mp.mpc(0))) + abs(
                sum((v for i, v in parts.items() if not nu[(key, i)]), mp.mpc(0))
            )
            bb = ptsum[(key, f)]
            cq = abs(q)
            print(
                f"{f'{rt}{n},{ctp}{k}':>10} {f:>5} {mp.nstr(cq, 2):>8} {mp.nstr(aa / cq, 2):>8}"
                f" {mp.nstr(aa / bb, 3):>9} {mp.nstr(bb / cq, 3):>10} {mp.nstr(tt / cq, 3):>10}"
                f" {mp.nstr(vv / cq, 3):>8} {mp.nstr(aa / tt, 3):>10}",
                flush=True,
            )


if __name__ == "__main__":
    main()
