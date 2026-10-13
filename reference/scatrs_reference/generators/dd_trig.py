"""Arb values of double-double trigonometric functions at random double-double inputs.

Each function has a record of N random inputs and a record ``<name>_edges`` of hand-picked
inputs at the edges of its domain and of its algorithm's branches:

* ``sin``, ``cos``, ``tan``: the doubles closest to a multiple of π/2 in every binade below
  2^50 (the hardest cases of the argument reduction), the double-doubles and doubles nearest
  to k·π/2 for k of both signs up to the reduction limit, the switch points (k + 1/2)·π/2 of
  the reduction, π/4 (below which there is no reduction) and the limit 2^50 itself;
* ``atan``, ``asin``, ``acos``: tiny and huge arguments, ±1 and its neighbours, 1/√2
  (where asin and acos pass |y| = |x| to atan2), and the limits of atan2's fast path;
* ``atan2``: |y| = |x| in all quadrants, results next to 0, ±π/2 and ±π, extreme magnitudes
  and ratios, subnormal arguments, and the limits of the fast path and of the tiny-ratio branch.

Every edge input is an exact normalised double-double, and edge cases whose value is zero,
non-finite, at least ``MAX`` or below 2^-969 in magnitude (the full-precision domain) are
dropped.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Iterable

from flint import arb, ctx

from scatrs_reference.records import nearest_double, nearest_pow10, record, split

PRECISION_BITS = 600
N = 200
SEED = 5

MAX = sys.float_info.max
TINY = 2.0**-1074
#: sin and cos are NaN for |x| above 2^50.
MAX_REDUCIBLE = 2.0**50
FRAC_PI_4 = math.pi / 4
FRAC_1_SQRT_2 = math.sqrt(0.5)
#: atan2 takes its fast path when |y| and |x| both lie in [2^-450, 2^450].
ATAN2_FAST_MIN = 2.0**-450
ATAN2_FAST_MAX = 2.0**450
#: After scaling the larger argument into [1, 2), atan2 uses its Newton step when the smaller
#: is at least 2^-968, and the limits 0, ±π/2 or ±π of the result below that.
ATAN2_TINY_RATIO = 2.0**-968
#: Bits of the fixed-point π/2 used to search for the hardest reduction cases.
SEARCH_BITS = 400


def _dd(rng: random.Random, f: float) -> tuple[float, float]:
    if f == 0.0:
        return f, 0.0
    lo = rng.uniform(-0.5, 0.5) * math.ulp(f)
    return (f, lo) if f + lo == f else (f, 0.0)


def _logu(rng: random.Random, a: float, b: float) -> float:
    return nearest_pow10(rng.uniform(a, b))


def _signed(rng: random.Random, v: float) -> float:
    return v if rng.random() < 0.5 else -v


def _angle(rng: random.Random) -> float:
    u = rng.random()
    if u < 0.3:
        return _signed(rng, _logu(rng, -30, -0.11))  # |x| < π/4: relative accuracy
    if u < 0.6:
        return rng.uniform(-10.0, 10.0)
    if u < 0.85:
        return rng.uniform(-(2.0**20), 2.0**20)
    return rng.uniform(-(2.0**50), 2.0**50)


SAMPLERS = {
    "sin": _angle,
    "cos": _angle,
    "tan": lambda r: (
        _signed(r, _logu(r, -30, -0.11)) if r.random() < 0.4 else r.uniform(-10.0, 10.0)
    ),
    "atan": lambda r: _signed(r, _logu(r, -30, 30)),
    "asin": lambda r: _signed(r, _logu(r, -30, 0)),
    "acos": lambda r: _signed(r, _logu(r, -30, 0)),
}

FUNCTIONS = {
    "sin": lambda x: x.sin(),
    "cos": lambda x: x.cos(),
    "tan": lambda x: x.tan(),
    "atan": lambda x: x.atan(),
    "asin": lambda x: x.asin(),
    "acos": lambda x: x.acos(),
}


def _as_arb(p: tuple[float, float]) -> arb:
    """The exact value of a normalised double-double input."""
    assert p[0] + p[1] == p[0], f"non-normalised input {p}"
    s = arb(p[0]) + arb(p[1])
    assert s.rad() == 0, f"inexact input {p}"
    return s


def _in_domain(v: arb) -> bool:
    """Finite, nonzero, below MAX and at least 2^-969 in magnitude."""
    return bool(v.is_finite() and abs(v) >= arb(2) ** -969 and abs(v) < arb(MAX))


# --- hardest cases of the argument reduction ----------------------------------------------


def _half_pi_fixed() -> int:
    """floor(π/2 · 2^SEARCH_BITS)."""
    with ctx.workprec(SEARCH_BITS + 64):
        return int((arb.pi() * arb(2) ** (SEARCH_BITS - 1)).floor().unique_fmpz())


def _round_div(a: int, b: int) -> int:
    q, r = divmod(a, b)
    return q + 1 if 2 * r >= b else q


def closest_to_multiple_of_half_pi(e: int, half_pi: int | None = None) -> tuple[float, int]:
    """The double x in [2^e, 2^(e+1)), e >= 0, closest to an integer multiple k·π/2: (x, k).

    The doubles of the binade are m·2^(e−52) with 2^52 <= m < 2^53, and
    |m·2^(e−52) − k·π/2| = 2^(e−52)·|m − kβ| with β = (π/2)·2^(52−e). In fixed point, the
    pairs (k, k·H − m·2^S), with H = ⌊(π/2)·2^SEARCH_BITS⌋ and S = SEARCH_BITS − 52 + e, form
    a two-dimensional integer lattice. After Lagrange–Gauss reduction (with k weighted so
    that both coordinates are comparable), every lattice point with k < 2^(e+1) and a
    residual below a bound R is enumerated; R doubles until a point inside the binade is
    found, which is then the closest one (all points with a smaller residual were enumerated).
    """
    hp = _half_pi_fixed() if half_pi is None else half_pi
    s = SEARCH_BITS - 52 + e
    k_hi = ((2**53) << s) // hp + 2
    w = max(1, (1 << s) // (k_hi * k_hi))
    # Basis vectors: (weighted k, residual, k, p) for k = 1 and for p = 1.
    b1 = [w, hp, 1, 0]
    b2 = [0, -(1 << s), 0, 1]

    def norm2(v: list[int]) -> int:
        return v[0] * v[0] + v[1] * v[1]

    if norm2(b1) > norm2(b2):
        b1, b2 = b2, b1
    while True:
        q = _round_div(b1[0] * b2[0] + b1[1] * b2[1], norm2(b1))
        b2 = [b2[i] - q * b1[i] for i in range(4)]
        if norm2(b2) >= norm2(b1):
            break
        b1, b2 = b2, b1
    det = abs(b1[0] * b2[1] - b1[1] * b2[0])
    bound = w * k_hi
    radius = 4 * (1 << s) // max(1, (2**52 << s) // hp)
    while True:
        # Coefficients of the points with |weighted k| <= bound and |residual| <= radius.
        c1_max = (bound * abs(b2[1]) + radius * abs(b2[0])) // det + 1
        c2_max = (bound * abs(b1[1]) + radius * abs(b1[0])) // det + 1
        best: tuple[int, int, int] | None = None
        for c1 in range(-c1_max, c1_max + 1):
            for c2 in range(-c2_max, c2_max + 1):
                k = c1 * b1[2] + c2 * b2[2]
                if k <= 0:
                    continue
                m = (k * hp + (1 << (s - 1))) >> s
                d = abs(k * hp - (m << s))
                if 2**52 <= m < 2**53 and d <= radius and (best is None or d < best[0]):
                    best = (d, k, m)
        if best is not None:
            return math.ldexp(best[2], e - 52), best[1]
        radius *= 2


def _hardest_reduction_cases() -> list[float]:
    """The closest double to a multiple of π/2 in each binade [2^e, 2^(e+1)), 0 <= e < 50."""
    hp = _half_pi_fixed()
    return [closest_to_multiple_of_half_pi(e, hp)[0] for e in range(50)]


# --- edge cases ---------------------------------------------------------------------------


def _below(x: float) -> float:
    return math.nextafter(x, -math.inf)


def _above(x: float) -> float:
    return math.nextafter(x, math.inf)


def _around(*xs: float) -> list[float]:
    """Each x with its neighbouring doubles."""
    return [y for x in xs for y in (_below(x), x, _above(x))]


def _both_signs(xs: Iterable[float]) -> list[float]:
    return [y for x in xs for y in (x, -x)]


def _with_lo(hi: float, frac: float) -> tuple[float, float]:
    """(hi, frac·ulp(hi)), with the low word halved until the pair is normalised."""
    lo = frac * math.ulp(hi)
    while hi + lo != hi:
        lo /= 2.0
    return hi, lo


def _variants(xs: Iterable[float]) -> list[tuple[float, float]]:
    """Each x as a double-double with a zero low word and with low words near ±ulp/2."""
    out: list[tuple[float, float]] = []
    for x in xs:
        for p in ((x, 0.0), _with_lo(x, 0.49), _with_lo(x, -0.49)):
            if p not in out:
                out.append(p)
    return out


def _dedup(ps: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in ps:
        if p not in out:
            out.append(p)
    return out


def _nearest(v: arb) -> list[tuple[float, float]]:
    """The double-double and the double nearest to v."""
    return [split(v)[:2], (nearest_double(v), 0.0)]


TINY_RESULTS = [2.0**-960, 1e-290, 1e-100, 1e-30, 1e-17, 1e-10]


def _angle_edges(tan: bool) -> list[tuple[float, float]]:
    """Edge inputs of sin and cos (or, with ``tan``, of tan: fewer large multiples)."""
    half_pi = arb.pi() / 2
    k_max = int((arb(MAX_REDUCIBLE) / half_pi).floor().unique_fmpz())  # k·π/2 <= 2^50
    out = _variants(
        _both_signs([*TINY_RESULTS, 0.5, *_around(FRAC_PI_4), 1.0, 2.0, 3.0, 10.0, 100.0])
        + _both_signs([1e6, 1e10, 1e15, _below(MAX_REDUCIBLE), MAX_REDUCIBLE])
    )
    out += [_with_lo(s * MAX_REDUCIBLE, -s * 0.2) for s in (1.0, -1.0)]  # 2^50 - 0.05: inside
    switch_ks = [0, 1, 2, 3] if tan else [0, 1, 2, 3, 4, 7, 100, 12345, 2**20, 2**40, k_max - 1]
    for k in switch_ks:  # (k + 1/2)·π/2: the reduction rounds k either way
        for s in (1, -1):
            v = s * (k + arb(0.5)) * half_pi
            out += [split(v)[:2], *_variants([nearest_double(v)])]
    ks = [1, 2, 3, 4, 101]
    if not tan:
        ks += [5, 6, 7, 8, 100, 12345, 2**20 + 1, 2**30 - 1, 2**40 + 3, k_max]
    for k in ks:  # results next to 0 and ±1, for both signs of k
        for s in (1, -1):
            out += _nearest(s * k * half_pi)
    hard = _hardest_reduction_cases()
    for x in hard[:21] if tan else hard:  # tan: |x| < 2^21
        out += [(x, 0.0), (-x, 0.0)]
    return _dedup(p for p in out if abs(_as_arb(p)) <= MAX_REDUCIBLE)


def _unit_edges() -> list[tuple[float, float]]:
    """Edge inputs of asin and acos."""
    xs = [*TINY_RESULTS, 2.0**-969, *_around(ATAN2_FAST_MIN), 0.1, 0.5, *_around(FRAC_1_SQRT_2)]
    xs += [0.9, 0.99, 1.0 - 1e-10, _below(1.0), 1.0]
    out = _variants(_both_signs(xs))
    out.append(split(1 / arb(2).sqrt())[:2])
    out += [(s, -s * f) for s in (1.0, -1.0) for f in (2.0**-60, 2.0**-100)]
    return _dedup(out)


ATAN_EDGES = [*TINY_RESULTS, 2.0**-969, *_around(ATAN2_TINY_RATIO), *_around(ATAN2_FAST_MIN)]
ATAN_EDGES += [0.25, 0.5, *_around(1.0), 2.0, 10.0, 1e10, 1e16, 1e20, 1e100]
ATAN_EDGES += [*_around(ATAN2_FAST_MAX), 1e300, _below(MAX), MAX]

ATAN2_EDGES = [
    # |y| = |x| in every quadrant, and neighbours
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
    (3.0, 3.0),
    (1.0, _below(1.0)),
    (1.0, _above(1.0)),
    (_below(1.0), -1.0),
    (-1.0, _above(1.0)),
    (0.75, 0.75),
    # results next to 0, ±π/2 and ±π
    (1e-40, 1.0),
    (-1e-40, 1.0),
    (1e-40, -1.0),
    (-1e-40, -1.0),
    (1e-20, -1.0),
    (2.0**-60, -1.0),
    (1e-16, -1.0),
    (1.0, 1e-40),
    (1.0, -1e-40),
    (-1.0, 1e-40),
    (-1.0, -1e-40),
    (1e20, 1.0),
    (-1e20, -1.0),
    (1.0, 1e-16),
    (1e-300, -1e300),
    (-1e-300, -1e300),
    (1e300, 1e-300),
    # extreme magnitudes and ratios
    (MAX, MAX),
    (MAX, -MAX),
    (-MAX, -MAX),
    (MAX, 1.0),
    (1.0, MAX),
    (MAX, 0.5 * MAX),
    (1e300, 1e300),
    (1e300, -3e299),
    (1e-300, 1e-300),
    (1e-300, 2e-300),
    (2e-300, -1e-300),
    (TINY, TINY),
    (TINY, 3 * TINY),
    (-3 * TINY, TINY),
    (1e-310, 3e-310),
    (-5e-324, -1e-320),
    (1e-300, 1e-10),
    (1e-300, -1e-10),
    (1e-140, 1e150),
    (1e-10, 1e300),
    (1e-300, 1e300),
    # limits of the fast path
    (ATAN2_FAST_MAX, 1.0),
    (_above(ATAN2_FAST_MAX), 1.0),
    (ATAN2_FAST_MIN, 1.0),
    (_below(ATAN2_FAST_MIN), 1.0),
    (ATAN2_FAST_MIN, ATAN2_FAST_MAX),
    (1.0, ATAN2_FAST_MIN),
    (ATAN2_FAST_MAX, -ATAN2_FAST_MAX),
    (_below(ATAN2_FAST_MIN), -_below(ATAN2_FAST_MIN)),
    (ATAN2_FAST_MAX, _above(ATAN2_FAST_MAX)),
    # limits of the tiny-ratio branch after scaling
    (ATAN2_TINY_RATIO, 1.0),
    (_below(ATAN2_TINY_RATIO), 1.0),
    (1.5 * ATAN2_TINY_RATIO, 1.0),
    (ATAN2_TINY_RATIO, -1.0),
    (1.0, ATAN2_TINY_RATIO),
    (1e300 * ATAN2_TINY_RATIO, 1.5e300),
    (2.0**-1060, 2.0**-92),
    (1.5 * 2.0**-1060, -(2.0**-92)),
    (2.0**-92, 2.0**-1060),
    # generic quadrants
    (1.0, 2.0),
    (2.0, 1.0),
    (-1.0, 2.0),
    (-2.0, -1.0),
    (0.5, -3.0),
    (1e10, 3.0),
    (3.0, 1e10),
    (-7.0, -1e-5),
]

Pair = tuple[tuple[float, float], tuple[float, float]]


def _pair_edges(pairs: list[tuple[float, float]]) -> list[Pair]:
    """Edge (y, x) pairs with low-word variants of both arguments."""
    out: list[Pair] = []
    for y, x in pairs:
        for yp in _variants([y]):
            for xp in _variants([x]):
                if (yp, xp) not in out:
                    out.append((yp, xp))
    return out


def _unary_edges_record(name: str, xs: list[tuple[float, float]]) -> dict:
    kept = [(p, v) for p in xs if _in_domain(v := FUNCTIONS[name](_as_arb(p)))]
    return record(
        "dd_trig",
        f"{name}_edges",
        {"x_hi": [p[0] for p, _ in kept], "x_lo": [p[1] for p, _ in kept]},
        {"value": [v for _, v in kept]},
    )


def _atan2_record(case: str, ys: list, xs: list, edges: bool = False) -> dict:
    kept = []
    for y, x in zip(ys, xs, strict=True):
        v = arb.atan2(_as_arb(y), _as_arb(x))
        if not edges or _in_domain(v):
            kept.append((y, x, v))
    return record(
        "dd_trig",
        case,
        {
            "y_hi": [y[0] for y, _, _ in kept],
            "y_lo": [y[1] for y, _, _ in kept],
            "x_hi": [x[0] for _, x, _ in kept],
            "x_lo": [x[1] for _, x, _ in kept],
        },
        {"value": [v for _, _, v in kept]},
    )


def generate() -> tuple[int, list[dict]]:
    with ctx.workprec(PRECISION_BITS):
        rng = random.Random(SEED)
        recs = []
        for name, sample in SAMPLERS.items():
            xs = [_dd(rng, sample(rng)) for _ in range(N)]
            recs.append(
                record(
                    "dd_trig",
                    name,
                    {"x_hi": [h for h, _ in xs], "x_lo": [lo for _, lo in xs]},
                    {"value": [FUNCTIONS[name](_as_arb(p)) for p in xs]},
                )
            )
        ys = [_dd(rng, _signed(rng, _logu(rng, -30, 30))) for _ in range(N)]
        xs = [_dd(rng, _signed(rng, _logu(rng, -30, 30))) for _ in range(N)]
        recs.append(_atan2_record("atan2", ys, xs))

        angle_edges = _angle_edges(tan=False)
        edges = {
            "sin": angle_edges,
            "cos": angle_edges,
            "tan": _angle_edges(tan=True),
            "atan": _variants(_both_signs(ATAN_EDGES)),
            "asin": _unit_edges(),
            "acos": _unit_edges(),
        }
        for name, inputs in edges.items():
            recs.append(_unary_edges_record(name, inputs))
        pairs = _pair_edges(ATAN2_EDGES)
        ys, xs = [y for y, _ in pairs], [x for _, x in pairs]
        recs.append(_atan2_record("atan2_edges", ys, xs, edges=True))
        return PRECISION_BITS, recs
