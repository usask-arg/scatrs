"""Arb values of double-double elementary functions at random double-double inputs.

Each function has a record of N random inputs (``SAMPLERS``) and a record ``<name>_edges`` of
hand-picked inputs at the edges of its domain and of its algorithm's branches (``EDGES``):
overflow and underflow thresholds, subnormal inputs, values next to 1, and the switch
points between series and reduction formulas. Every edge input is an exact normalised
double-double, and edge cases whose value is zero, non-finite, at least ``MAX`` or below
2^-969 in magnitude (the full-precision domain) are dropped.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable, Iterable

from flint import arb, ctx

from scatrs_reference.records import record, split

PRECISION_BITS = 600
N = 200
SEED = 4

MAX = sys.float_info.max
TINY = 2.0**-1074
#: 1024·fl(ln 2): the largest double whose exponential is finite (just below ln MAX).
LN_MAX = 709.782712893384
#: fl(ln(2·MAX)): sinh and cosh overflow just above it.
LN_2MAX = 710.4758600739439
HALF_LN_2 = 0.34657359027997264


def _dd(rng: random.Random, f: float) -> tuple[float, float]:
    """A normalised double-double with leading word f and a random low word."""
    if f == 0.0:
        return f, 0.0
    lo = rng.uniform(-0.5, 0.5) * math.ulp(f)
    return (f, lo) if f + lo == f else (f, 0.0)


def _logu(rng: random.Random, lo_exp: float, hi_exp: float) -> float:
    return 10.0 ** rng.uniform(lo_exp, hi_exp)


def _signed(rng: random.Random, v: float) -> float:
    return v if rng.random() < 0.5 else -v


def _ln1p_sample(r: random.Random) -> float:
    u = r.random()
    if u < 0.5:
        return _signed(r, _logu(r, -30, -0.7))
    if u < 0.75:
        return _logu(r, -0.6, 6)
    return -_logu(r, -0.6, -0.0005)


SAMPLERS: dict[str, Callable[[random.Random], float]] = {
    "sqrt": lambda r: _logu(r, -300, 300),
    "exp": lambda r: r.uniform(-670.0, 709.0) if r.random() < 0.5 else _signed(r, _logu(r, -30, 0)),
    "exp_m1": lambda r: _signed(r, _logu(r, -30, 1.5)),
    "ln": lambda r: _logu(r, -300, 300) if r.random() < 0.5 else 1.0 + _signed(r, _logu(r, -15, -0.7)),
    "ln_1p": _ln1p_sample,
    "log2": lambda r: _logu(r, -300, 300),
    "log10": lambda r: _logu(r, -300, 300),
    "cbrt": lambda r: _signed(r, _logu(r, -300, 300)),
    "sinh": lambda r: _signed(r, _logu(r, -30, 2.8)),
    "cosh": lambda r: _signed(r, _logu(r, -30, 2.8)),
    "tanh": lambda r: _signed(r, _logu(r, -30, 1.7)),
    "asinh": lambda r: _signed(r, _logu(r, -30, 30)),
    "acosh": lambda r: 1.0 + _logu(r, -15, 30),
    "atanh": lambda r: _signed(r, _logu(r, -30, -0.3))
    if r.random() < 0.5
    else _signed(r, 1.0 - _logu(r, -15, -0.0005)),
    "exp2": lambda r: r.uniform(-960.0, 1020.0),
}

FUNCTIONS: dict[str, Callable[[arb], arb]] = {
    "sqrt": lambda x: x.sqrt(),
    "exp": lambda x: x.exp(),
    "exp_m1": lambda x: x.expm1(),
    "ln": lambda x: x.log(),
    "ln_1p": lambda x: x.log1p(),
    "log2": lambda x: x.log() / arb(2).log(),
    "log10": lambda x: x.log() / arb(10).log(),
    "cbrt": lambda x: x.root(3) if x >= 0 else -((-x).root(3)),
    "sinh": lambda x: x.sinh(),
    "cosh": lambda x: x.cosh(),
    "tanh": lambda x: x.tanh(),
    "asinh": lambda x: x.asinh(),
    "acosh": lambda x: x.acosh(),
    "atanh": lambda x: x.atanh(),
    "exp2": lambda x: arb(2) ** x,
}


def _as_arb(p: tuple[float, float]) -> arb:
    s = arb(p[0]) + arb(p[1])
    assert s.rad() == 0, f"inexact input {p}"
    return s


# --- edge cases -------------------------------------------------------------------------


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


def _in_domain(v: arb) -> bool:
    """Finite, nonzero, below MAX and at least 2^-969 in magnitude."""
    return bool(v.is_finite() and abs(v) >= arb(2) ** -969 and abs(v) < arb(MAX))


SQRT_CBRT_EDGES = [TINY, 3 * TINY, 1e-310, 2.0**-1022, 2.0**-969, 1e-300, 1e-30, 1e-3]
SQRT_CBRT_EDGES += [_below(1.0), 1.0, _above(1.0), 2.0, 3.0, 4.0, 8.0, 27.0, 0.5]
SQRT_CBRT_EDGES += [1e300, 2.0**1023, _below(MAX), MAX]
LOG_EDGES = [TINY, 3 * TINY, 1e-310, 2.0**-1022, 2.0**-969, 1e-300, *_around(1e-270)]
LOG_EDGES += [*_around(1e270), 1e300, _below(MAX), MAX]
LOG_EDGES += [*_around(1.0), 1.0 + 1e-10, 1.0 - 1e-10, *_around(0.75), *_around(1.25)]
LOG_EDGES += [0.7, 0.74, 0.76, 1.24, 1.26, 1.3, 0.5, 2.0, math.e, 10.0, 1024.0]
TINY_RESULTS = [2.0**-960, 1e-290, 1e-100, 1e-30, 1e-17, 1e-10]
HYPERBOLIC = [*TINY_RESULTS, *_around(2.0**-11), *_around(HALF_LN_2), 0.45, *_around(0.5)]
HYPERBOLIC += [1.0, 10.0, *_around(40.0), 100.0, 700.0, 709.0, *_around(LN_MAX)]
HYPERBOLIC += [710.0, 710.4, 710.475, *_around(LN_2MAX)]


def _near_multiples_of_ln_2() -> list[tuple[float, float]]:
    """Double-doubles nearest to k·ln 2 (tiny reduced arguments for exp), and the switch
    points (k + 1/2)·ln 2 of its reduction."""
    ln2 = arb(2).log()
    out = [split(arb(k) * ln2)[:2] for k in (1, 5, -7, 100, -900, 1023)]
    for k in (3, -12, 500):
        out += _variants([float((k + 0.5) * ln2)])
    return out


def _edges() -> dict[str, list[tuple[float, float]]]:
    """Edge inputs per function (needs the working precision for the ln 2 multiples)."""
    return {
        "sqrt": _variants(SQRT_CBRT_EDGES),
        "exp": _variants(
            [*_around(LN_MAX), 709.0, 708.5, -670.0, -669.5, -600.0, -100.0, 100.0]
            + [1e-300, 2.0**-960]
            + _both_signs([*_around(HALF_LN_2), 0.5, 1.0, 1e-20, *_around(2.0**-11)])
        )
        + _near_multiples_of_ln_2(),
        "exp_m1": _variants(
            _both_signs([*TINY_RESULTS, *_around(2.0**-11), *_around(HALF_LN_2), 0.5, 1.0, 40.0])
            + [-700.0, 700.0, *_around(LN_MAX)]
        ),
        "ln": _variants(LOG_EDGES),
        "ln_1p": _variants(
            _both_signs([*TINY_RESULTS, *_around(2.0**-11), *_around(0.25), 0.26, 0.5])
            + [-0.75, -0.9, -0.99, _above(-1.0), 1.0, 10.0, 1e300, MAX]
        )
        + [(-1.0, 2.0**-60), (-1.0, 2.0**-100)],
        "log2": _variants(LOG_EDGES),
        "log10": _variants(LOG_EDGES),
        "cbrt": _variants(_both_signs(SQRT_CBRT_EDGES)),
        "sinh": _variants(_both_signs(HYPERBOLIC)),
        "cosh": _variants([0.0, *_both_signs(HYPERBOLIC)]),
        "tanh": _variants(
            _both_signs([*TINY_RESULTS, *_around(2.0**-12), *_around(HALF_LN_2 / 2), 0.5, 1.0])
            + _both_signs([10.0, 19.0, 20.0, *_around(40.0), 100.0, 1e300, MAX])
        ),
        "asinh": _variants(
            _both_signs([*TINY_RESULTS, 0.1, 0.25, 0.5, 1.0, 2.0, 1e10, *_around(1e150)])
            + _both_signs([1e154, 1e300, _below(MAX), MAX])
        ),
        "acosh": _variants([_above(1.0), 1.0 + 1e-10, 1.1, 1.25, 2.0, 10.0, 1e10])
        + _variants([*_around(1e150), 1e300, MAX])
        + [(1.0, 2.0**-60), (1.0, 2.0**-100)],
        "atanh": _variants(
            _both_signs([*TINY_RESULTS, 0.1, *_around(1.0 / 9.0), 0.25, 0.5, 0.9, 0.99])
            + _both_signs([1.0 - 1e-10, _below(1.0)])
        )
        + [(s, s * f) for s in (1.0, -1.0) for f in (-(2.0**-60), -(2.0**-100))],
        "exp2": _variants(
            [-969.0, -968.5, -960.0, 1e-300, 1000.0, 1023.0, 1023.5, _below(1024.0)]
            + _both_signs([0.5, 1.0, 10.0])
        )
        + [(1024.0, -(2.0**-50)), (1024.0, -(2.0**-45))],
    }


HYPOT_EDGES = [
    (3.0, 4.0), (MAX, 1.0), (MAX, 1e300), (1e308, 1e308), (0.7 * MAX, 0.7 * MAX),
    (_below(MAX), TINY), (2.0**-960, 2.0**-960), (2.0**-969, 2.0**-1000), (1.0, 1e-200),
    (1e-200, 1.0), (1.0, 2.0**-27), (1.0, 1.0), (-1.0, 1.0), (1e150, 1e-150), (0.0, 5.0),
    (-0.0, -3.0),
]
POWF_EDGES = [
    (_above(1.0), 1e15), (_below(1.0), 1e15), (2.0, 1023.5), (2.0, -960.0), (10.0, 308.0),
    (10.0, -290.0), (1e300, 1.02), (1e-300, 0.9), (MAX, 1.0), (MAX, 0.5), (0.999999, 1e8),
    (4.0, 0.5), (3.0, 2.0), (2.0, 0.5), (0.5, -1000.0), (TINY, -0.9), (TINY, 0.9),
    (-2.0, 3.0), (-2.0, 4.0), (-1.5, 11.0), (-1.5, -11.0), (-1.000001, 1000001.0),
    (-1.000001, -1000001.0), (-0.999999, 1000000.0), (-3.0, 7.0), (-3.0, 8.0), (-3.0, -7.0),
    (-10.0, 300.0), (-1e-300, -1.0), (-2.0, 1023.0),
]
POWI_EDGES = [
    (1e-3, -100), (0.71, -2000), (0.5, -969), (2.0, 1023), (2.0, -969), (1.0001, 100000),
    (-1.5, 1001), (-1.5, -1000), (10.0, 308), (10.0, -290), (MAX, 1), (2.0**-1000, -1),
    (1e-300, -1), (1e-150, -2), (3.0, 600), (-3.0, 601), (1.5, 0), (_below(1.0), 2**31 - 1),
    (_above(1.0), -(2**31)),
]


def _binary_record(
    case: str,
    pairs: list[tuple[tuple[float, float], tuple[float, float] | int]],
    f: Callable[[arb, arb | int], arb],
    second: str,
    edges: bool = False,
) -> dict:
    """A record of f(x, y) with double-double x and y (or integer n when ``second == "n"``).

    Edge records drop the cases whose value is outside the full-precision domain.
    """
    x_hi, x_lo, y_hi, y_lo, ns, vs = [], [], [], [], [], []
    for x, y in pairs:
        v = f(_as_arb(x), y if isinstance(y, int) else _as_arb(y))
        if edges and not _in_domain(v):
            continue
        x_hi.append(x[0])
        x_lo.append(x[1])
        if isinstance(y, int):
            ns.append(y)
        else:
            y_hi.append(y[0])
            y_lo.append(y[1])
        vs.append(v)
    inputs: dict[str, list] = {"x_hi": x_hi, "x_lo": x_lo}
    if second == "n":
        inputs["n"] = ns
    else:
        inputs |= {"y_hi": y_hi, "y_lo": y_lo}
    return record("dd_elementary", case, inputs, {"value": vs})


def _hypot(a: arb, b: arb) -> arb:
    return (a**2 + b**2).sqrt()


def _pow(a: arb, b: arb | int) -> arb:
    return a**b


def _edge_pairs(
    pairs: list[tuple[float, float | int]],
) -> list[tuple[tuple[float, float], tuple[float, float] | int]]:
    """Edge (x, y) pairs with low-word variants of x (and of y when it is a non-integer)."""
    out: list[tuple[tuple[float, float], tuple[float, float] | int]] = []
    for x, y in pairs:
        for xp in _variants([x]):
            if isinstance(y, int) or y == math.trunc(y):
                out.append((xp, y if isinstance(y, int) else (y, 0.0)))
            else:
                out.extend((xp, yp) for yp in _variants([y]))
    return out


def generate() -> tuple[int, list[dict]]:
    with ctx.workprec(PRECISION_BITS):
        rng = random.Random(SEED)
        recs = []
        for name, sample in SAMPLERS.items():
            xs = [_dd(rng, sample(rng)) for _ in range(N)]
            recs.append(
                record(
                    "dd_elementary",
                    name,
                    {"x_hi": [h for h, _ in xs], "x_lo": [lo for _, lo in xs]},
                    {"value": [FUNCTIONS[name](_as_arb(p)) for p in xs]},
                )
            )
        xs = [_dd(rng, _signed(rng, _logu(rng, -150, 150))) for _ in range(N)]
        ys = [_dd(rng, _signed(rng, _logu(rng, -150, 150))) for _ in range(N)]
        recs.append(_binary_record("hypot", list(zip(xs, ys)), _hypot, "y"))
        xs = [_dd(rng, _logu(rng, -5, 5)) for _ in range(N)]
        ys = [_dd(rng, rng.uniform(-20.0, 20.0)) for _ in range(N)]
        recs.append(_binary_record("powf", list(zip(xs, ys)), _pow, "y"))
        xs = [_dd(rng, _signed(rng, _logu(rng, -3, 3))) for _ in range(N)]
        ns = [rng.randint(-100, 100) for _ in range(N)]
        recs.append(_binary_record("powi", list(zip(xs, ns)), _pow, "n"))

        for name, pairs in _edges().items():
            kept = [(p, v) for p in pairs if _in_domain(v := FUNCTIONS[name](_as_arb(p)))]
            recs.append(
                record(
                    "dd_elementary",
                    f"{name}_edges",
                    {"x_hi": [p[0] for p, _ in kept], "x_lo": [p[1] for p, _ in kept]},
                    {"value": [v for _, v in kept]},
                )
            )
        for case, pairs, f, second in (
            ("hypot_edges", HYPOT_EDGES, _hypot, "y"),
            ("powf_edges", POWF_EDGES, _pow, "y"),
            ("powi_edges", POWI_EDGES, _pow, "n"),
        ):
            recs.append(_binary_record(case, _edge_pairs(pairs), f, second, edges=True))
        return PRECISION_BITS, recs
