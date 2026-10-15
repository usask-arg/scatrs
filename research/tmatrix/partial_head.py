"""Partial heads: subtract only the Laurent powers p ≤ p_cut of the y_n products. If every
power's group (over the term types) integrates to zero, each cutoff is an exact form; this
measures whether it is (|Q_cut − Q| / |Q|) and the cancellation Σ|terms| / |Q| per cutoff,
against the forms the kernel has (plain; p < 0; p ≤ 0).

Usage: python partial_head.py x eps s_re s_im m nmax ntheta_full "n1,n2" "k1,k2,..."
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
    cuts = {key: list(range(key[3] - key[2] - 4, 1)) for key in keys}
    val = {}
    absv = {}
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
            for ka, kb, power, coeff in element_terms(n, k, rt, ctp, s, d, pi_, tau, beta, st, ct):
                rj = get(kb, k, s * r, "")
                jpart = get(ka, n, r, "jn") * rj * r**power
                yfull = get(YKIND[ka], n, r, "") * rj * r**power
                skey = (YKIND[ka], kb, n, k, power)
                if skey not in series:
                    series[skey] = sorted(
                        product(
                            radial_y(n, YKIND[ka], nmax + 30), radial_j(k, kb, s, nmax + 30), power
                        ).items()
                    )
                head = mp.mpc(0)
                heads = {}
                it = iter(series[skey])
                pending = next(it)
                for cut in cuts[key]:
                    while pending is not None and pending[0] <= cut:
                        head += pending[1] * r ** pending[0]
                        pending = next(it, None)
                    heads[cut] = head
                for cut in [None, *cuts[key]]:
                    ypart = yfull if cut is None else yfull - heads[cut]
                    kk = (key, cut)
                    val[kk] = val.get(kk, 0) + w * coeff * (jpart + 1j * ypart)
                    absv[kk] = absv.get(kk, 0) + abs(w * coeff) * (abs(jpart) + abs(ypart))
    print(f"x={x} eps={eps} s={sr}+{si}i m={m} N={nmax} nodes={nth}")
    for key in keys:
        rt, ctp, n, k = key
        q = val[(key, None)]
        if abs(q) < mp.mpf(10) ** -25 * absv[(key, None)]:
            continue
        row = []
        best = (absv[(key, None)] / abs(q), None)
        for cut in cuts[key]:
            exact = abs(val[(key, cut)] - q) / abs(q)
            loss = absv[(key, cut)] / abs(q)
            if exact < mp.mpf(10) ** -20 and loss < best[0]:
                best = (loss, cut)
            row.append(f"{cut}:{mp.nstr(loss, 2)}{'' if exact < mp.mpf(10) ** -20 else '(x)'}")
        cur = min(
            absv[(key, None)] / abs(q),
            *(
                absv[(key, cc)] / abs(q)
                for cc in (-1, 0)
                if (key, cc) in absv and abs(val[(key, cc)] - q) < mp.mpf(10) ** -20 * abs(q)
            ),
        )
        print(
            f"{rt}{n},{ctp}{k}: plain {mp.nstr(absv[(key, None)] / abs(q), 2)}"
            f"  kernel best {mp.nstr(cur, 2)}  partial best {mp.nstr(best[0], 2)} at p<={best[1]}"
        )
        print("    " + " ".join(row))


if __name__ == "__main__":
    main()
