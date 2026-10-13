"""Wigner d^l_{mn}(θ) in Arb at exact x = cos θ: the explicit finite sum, and Legendre functions.

Convention: d^l_{mn} equals the common d^j_{m'm}(β) with m' = m (Mishchenko, Travis & Lacis
2002, Appendix B). The sum is finite, so every value is certified.

The terms of the sum cancel to about 2^-l of their size, so each value is evaluated at
``PRECISION_BITS`` and the precision is doubled until the ball radius is at most
2^-``REL_BITS`` relative, or 2^-``ABS_BITS`` absolute (for values that are exactly zero).

Records ``m=..,n=..,x=..,l_max=..`` hold d^l for every l = 0..l_max. Records ending in
``,partial`` or ``,underflow`` hold only the ascending degrees ``ls`` of a recurrence up to
``l_max``. The partial ones check large degrees: ``LARGE_CASES``, ``POLE_CASES`` next to the
poles, and ``HUGE_CASES`` up to l = 2.5·10⁵ from Arb's Legendre functions. The underflow ones
check values that are normal doubles although the start value of the recurrence, at
l = max(|m|, |n|), lies below the double range (``UNDERFLOW_CASES``).
"""

from __future__ import annotations

import math

from flint import arb, ctx, fmpz

from scatrs_reference.records import record

PRECISION_BITS = 800
REL_BITS = 200
ABS_BITS = 1300
MAX_PRECISION_BITS = 1 << 16
L_MAX = 100
PAIRS = (
    (0, 0),
    (0, 2),
    (2, 2),
    (2, -2),
    (1, 1),
    (1, -1),
    (0, 1),
    (1, 0),
    (3, -7),
    (-4, 2),
    (5, 5),
    (0, -3),
    (10, 4),
    (-3, -5),
    (-6, -6),
    (-3, 3),
    (50, -37),
)
XS = (
    -1.0,
    -0.9999999999999999,  # -1 + 2^-53
    -0.9999999,
    -0.5,
    0.0,
    0.3,
    0.8660254037844386,
    0.9999999,
    0.9999999999999997,  # 1 - 3·2^-53
    1.0,
)

#: Degrees up to 2000 for the Greek-coefficient pairs and some general (m, n).
LARGE_L_MAX = 2000
LARGE_LS = (0, 1, 2, 3, 10, 37, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 1998, 1999, 2000)
LARGE_CASES = tuple(
    [(m, n, x) for (m, n) in ((0, 0), (2, 2), (2, -2), (0, 2)) for x in (-0.7, 0.3, 0.99, 0.9999999)]
    + [(m, n, x) for (m, n) in ((50, -37), (-37, -50), (7, 150)) for x in (-0.7, 0.3, 0.99)]
)

#: Next to a pole the plain three-term recurrence loses accuracy like l², which these degrees
#: up to 4000 expose: its f64 errors were 18(l+1)·2^-52 for (35, 34) and 266(l+1)·2^-52 at
#: l = 2000 for (0, 0) here.
POLE_L_MAX = 4000
POLE_LS = (0, 1, 2, 35, 36, 1000, 2000, 3000, 3998, 3999, 4000)
POLE_CASES = ((35, 34, 0.9999999108496082), (0, 0, -0.9999999999999999))

#: Very large degrees, from Arb's Legendre functions instead of the explicit sum (see
#: ``d_legendre``): (0, 0) is P_l, and d^l_{0,2} = d^l_{2,0} = √((l−2)!/(l+2)!) P_l^2. x = ±0.3
#: and 0.5 take the three-term form or the m = ±n difference form, |x| ≥ 0.9 the general one.
HUGE_L_MAX = 250_000
HUGE_LS = (100_000, 200_000, 249_998, 249_999, 250_000)
HUGE_CASES = tuple(
    [(0, 0, x) for x in (-0.95, 0.3, 0.9, 0.99999)] + [(0, 2, x) for x in (-0.9999, 0.5, 0.95)]
)

#: Start values of the recurrence, at l = max(|m|, |n|), below the normal double range, while
#: d^l is a normal double from some l on (all cases are (m, n, x, l_max, ls)):
#:
#: * x = cos θ next to a pole, sin θ ≈ 4.8e-3 (the Gauss–Legendre node nearest ±1 for about 500
#:   points): start values about 2^-1157 for (0, 150) and (150, 0), 2^-1333 for (-20, 150) and
#:   the subnormal 2^-1026 for (0, 133), normal from l = 227, 227, 419 and 135 on;
#: * large orders inside the interval: 2^-1113 for (0, 1200) at x = 0.85, 2^-1100 for
#:   (1100, 1100) at x = 0 and 2^-1105 for (1500, -1500) at x = -0.2, normal from l = 1225,
#:   1110 and 1511 on.
UNDERFLOW_X = 0.99998843
UNDERFLOW_LS = (133, 134, 135, 150, 151, 200, 226, 227, 228, 300, 418, 419, 420, 600, 999, 1000)
UNDERFLOW_CASES = (
    (0, 150, UNDERFLOW_X, 1000, UNDERFLOW_LS),
    (150, 0, -UNDERFLOW_X, 1000, UNDERFLOW_LS),
    (-20, 150, UNDERFLOW_X, 1000, UNDERFLOW_LS),
    (0, 133, UNDERFLOW_X, 1000, UNDERFLOW_LS),
    (0, 1200, 0.85, 3000, (1200, 1201, 1210, 1224, 1225, 1226, 1300, 1500, 2000, 2999, 3000)),
    (1100, 1100, 0.0, 2500, (1100, 1101, 1105, 1109, 1110, 1111, 1200, 1500, 2000, 2499, 2500)),
    (1500, -1500, -0.2, 3000, (1500, 1501, 1505, 1510, 1511, 1512, 1600, 2000, 2999, 3000)),
)


def _fact(k: int) -> arb:
    return arb(fmpz(math.factorial(k)))


def _pow(base: arb, e: int) -> arb:
    return arb(1) if e == 0 else base**e


def d_ref(l: int, m: int, n: int, x: float) -> arb:
    """The explicit sum at the current working precision."""
    if l < max(abs(m), abs(n)):
        return arb(0)
    big_x = arb(x)
    c = ((1 + big_x) / 2).sqrt()
    s = ((1 - big_x) / 2).sqrt()
    pref = (_fact(l + m) * _fact(l - m) * _fact(l + n) * _fact(l - n)).sqrt()
    total = arb(0)
    for k in range(0, 2 * l + 1):
        a, b, cc, d = l + n - k, k, m - n + k, l - m - k
        if min(a, b, cc, d) < 0:
            continue
        term = pref / (_fact(a) * _fact(b) * _fact(cc) * _fact(d))
        term *= _pow(c, 2 * l + n - m - 2 * k) * _pow(s, m - n + 2 * k)
        total += -term if (m - n + k) % 2 else term
    return total


def d_certified(l: int, m: int, n: int, x: float) -> arb:
    """``d_ref`` at a precision high enough that the radius meets the REL_BITS/ABS_BITS bound."""
    prec = PRECISION_BITS
    while True:
        with ctx.workprec(prec):
            v = d_ref(l, m, n, x)
        rad = v.rad()
        if rad <= abs(v.mid()) * arb(2) ** -REL_BITS or rad <= arb(2) ** -ABS_BITS:
            return v
        prec *= 2
        if prec > MAX_PRECISION_BITS:
            raise ArithmeticError(f"d^{l}_{m},{n}({x!r}) needs more than {MAX_PRECISION_BITS} bits")


def d_legendre(l: int, m: int, n: int, x: float) -> arb:
    """d^l_{mn} for (m, n) = (0, 0) or (0, 2) from Arb's P_l, at the current working precision.

    d^l_{0,2} = d^l_{2,0} = √((l−2)!/(l+2)!)·P_l^2(x), and P_l^2 = (1 − x²)P_l″ =
    2x·l(P_{l−1} − x·P_l)/(1 − x²) − l(l+1)P_l (Arb's ``legendre_p`` with m = 2 returns NaN at
    l = 2.5·10⁵). ``tests/test_generators.py`` checks both against ``d_ref``.
    """
    big_x = arb(x)
    p = big_x.legendre_p(l)
    if (m, n) == (0, 0):
        return p
    if (m, n) != (0, 2) or l < 2:
        raise ValueError(f"d_legendre covers (0, 0) and (0, 2) with l >= 2, not ({m}, {n}), l={l}")
    p_prev = big_x.legendre_p(l - 1)
    p2 = 2 * big_x * l * (p_prev - big_x * p) / (1 - big_x * big_x) - l * (l + 1) * p
    norm = 1 / (arb(fmpz(l - 1)) * l * (l + 1) * (l + 2)).sqrt()
    return norm * p2


def d_legendre_certified(l: int, m: int, n: int, x: float) -> arb:
    """``d_legendre`` with the radius bound of ``d_certified``."""
    prec = PRECISION_BITS
    while True:
        with ctx.workprec(prec):
            v = d_legendre(l, m, n, x)
        rad = v.rad()
        if v.is_finite() and (
            rad <= abs(v.mid()) * arb(2) ** -REL_BITS or rad <= arb(2) ** -ABS_BITS
        ):
            return v
        prec *= 2
        if prec > MAX_PRECISION_BITS:
            raise ArithmeticError(f"d^{l}_{m},{n}({x!r}) needs more than {MAX_PRECISION_BITS} bits")


def generate() -> tuple[int, list[dict]]:
    with ctx.workprec(PRECISION_BITS):
        recs = []
        cases = [(m, n, x, L_MAX) for (m, n) in PAIRS for x in XS] + [(0, 2, 0.3, 300)]
        for m, n, x, l_max in cases:
            recs.append(
                record(
                    "wigner_d",
                    f"m={m},n={n},x={x!r},l_max={l_max}",
                    {"m": m, "n": n, "x": x, "l_max": l_max},
                    {"d": [d_certified(l, m, n, x) for l in range(l_max + 1)]},
                )
            )
        for m, n, x in HUGE_CASES:
            recs.append(
                record(
                    "wigner_d",
                    f"m={m},n={n},x={x!r},l_max={HUGE_L_MAX},partial",
                    {"m": m, "n": n, "x": x, "l_max": HUGE_L_MAX, "ls": list(HUGE_LS)},
                    {"d": [d_legendre_certified(l, m, n, x) for l in HUGE_LS]},
                )
            )
        partial = (
            [(m, n, x, LARGE_L_MAX, LARGE_LS, "partial") for (m, n, x) in LARGE_CASES]
            + [(m, n, x, POLE_L_MAX, POLE_LS, "partial") for (m, n, x) in POLE_CASES]
            + [(*case, "underflow") for case in UNDERFLOW_CASES]
        )
        for m, n, x, l_max, ls, suffix in partial:
            recs.append(
                record(
                    "wigner_d",
                    f"m={m},n={n},x={x!r},l_max={l_max},{suffix}",
                    {"m": m, "n": n, "x": x, "l_max": l_max, "ls": list(ls)},
                    {"d": [d_certified(l, m, n, x) for l in ls]},
                )
            )
        return PRECISION_BITS, recs
