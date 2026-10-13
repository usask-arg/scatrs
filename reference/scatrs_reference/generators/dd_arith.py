"""Exact results of double-double basic operations on random and adversarial normalised inputs.

Each record holds N cases drawn from four families (by case index modulo 4):

0. uniformly random significands and low words;
1. structured words: leading words that are powers of two, have all-ones significands or
   lie a few ulps from a power of two, and low words near ±ulp/2, zero or tiny;
2. for additive operations, partial cancellation b.hi = ∓(a.hi ± k·ulp(a.hi)) with small k
   (other operations: structured words, with b's low word near ±ulp/2);
3. as 2, with a.hi near a power of two so that b.hi often lies in the adjacent binade, and
   sometimes b.hi near ∓2·a.hi or ∓a.hi/2 (other operations: random leading words with
   low words near ±ulp/2).

``div_big_y`` divides by |y| in [2^950, MAX], where 1/y.hi is subnormal. ``rem`` checks the
truncated remainder x - y·trunc(x/y), computed exactly with ``fractions.Fraction``.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from flint import arb, ctx, fmpq

from scatrs_reference.records import record

PRECISION_BITS = 600
N = 2000
SEED = 3
OPS = ("add", "sub", "mul", "div", "add_f64", "mul_f64", "div_f64")
ADDITIVE = ("add", "sub", "add_f64")


def _sign(rng: random.Random) -> float:
    return rng.choice((-1.0, 1.0))


def _structured_significand(rng: random.Random) -> float:
    """A significand in [1, 2) that stresses rounding: 1, all ones, or near a power of two."""
    kind = rng.randrange(4)
    if kind == 0:
        return 1.0
    if kind == 1:
        return 2.0 - 2.0**-52
    j = rng.randint(1, 8)
    return 1.0 + j * 2.0**-52 if kind == 2 else 2.0 - j * 2.0**-52


def _hi(rng: random.Random, exponent: int, structured: bool) -> float:
    m = _structured_significand(rng) if structured else rng.uniform(1.0, 2.0)
    return _sign(rng) * min(m, 2.0 - 2.0**-52) * 2.0**exponent


def _lo(rng: random.Random, hi: float, kind: str) -> float:
    """A low word for ``hi`` such that hi + lo == hi."""
    u = math.ulp(hi)
    if kind == "uniform":
        lo = rng.uniform(-0.5, 0.5) * u
    elif kind == "half":  # within a few units of ±ulp/2, including exactly ±ulp/2
        lo = _sign(rng) * (0.5 - rng.randint(0, 8) * 2.0**-53) * u
    elif kind == "tiny":
        lo = rng.uniform(-0.5, 0.5) * u * 2.0 ** -rng.randint(1, 60)
    else:
        lo = 0.0
    while hi + lo != hi:  # an odd tie, or a negative lo below a power of two
        lo /= 2.0
    return lo


def _structured_lo(rng: random.Random, hi: float) -> float:
    return _lo(rng, hi, rng.choice(("half", "half", "tiny", "zero")))


def _random_dd(rng: random.Random, exponent: int | None = None) -> tuple[float, float]:
    e = rng.randint(-60, 60) if exponent is None else exponent
    hi = _hi(rng, e, structured=False)
    return hi, _lo(rng, hi, "uniform")


def _structured_dd(rng: random.Random, exponent: int | None = None) -> tuple[float, float]:
    e = rng.randint(-60, 60) if exponent is None else exponent
    hi = _hi(rng, e, structured=True)
    return hi, _structured_lo(rng, hi)


def _cancelling(rng: random.Random, ah: float, op: str, scale: float) -> tuple[float, float]:
    """b with b.hi = ∓(scale·ah + k·ulp) so that a ± b cancels partially."""
    t = scale * ah
    bh = t + rng.randint(-16, 16) * math.ulp(t)
    if op != "sub":
        bh = -bh
    return bh, _lo(rng, bh, rng.choice(("uniform", "half", "zero")))


def _pair(rng: random.Random, op: str, i: int) -> tuple[float, float, float, float]:
    family = i % 4
    if family == 0:
        return (*_random_dd(rng), *_random_dd(rng))
    if family == 1:
        return (*_structured_dd(rng), *_structured_dd(rng))
    if op in ADDITIVE:
        ah, al = _structured_dd(rng) if family == 3 else _random_dd(rng)
        scale = 1.0 if family == 2 else rng.choice((1.0, 1.0, 0.5, 2.0))
        return (ah, al, *_cancelling(rng, ah, op, scale))
    a = _structured_dd(rng) if family == 2 else _random_dd(rng)
    bh = _hi(rng, rng.randint(-60, 60), structured=family == 2)
    return (*a, bh, _lo(rng, bh, "half"))


def _exact(ah: float, al: float, bh: float, bl: float, op: str) -> arb:
    a, b = arb(ah) + arb(al), arb(bh) + arb(bl)
    return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]


def _basic(op: str, rng: random.Random) -> dict:
    cols: dict[str, list[float]] = {"a_hi": [], "a_lo": [], "b_hi": [], "b_lo": []}
    z = []
    for i in range(N):
        ah, al, bh, bl = _pair(rng, op, i)
        if op.endswith("_f64"):
            bl = 0.0
        for key, v in zip(cols, (ah, al, bh, bl), strict=True):
            cols[key].append(v)
        z.append(_exact(ah, al, bh, bl, op.removesuffix("_f64")))
    return record("dd_arith", op, cols, {"z": z})


def _div_big_y(rng: random.Random) -> dict:
    cols: dict[str, list[float]] = {"a_hi": [], "a_lo": [], "b_hi": [], "b_lo": []}
    z = []
    for i in range(N):
        ey = rng.randint(950, 1023)
        bh, bl = _structured_dd(rng, ey) if i % 2 else _random_dd(rng, ey)
        ea = rng.randint(ey - 60, 1023)
        ah, al = _random_dd(rng, ea) if i % 4 < 2 else _structured_dd(rng, ea)
        for key, v in zip(cols, (ah, al, bh, bl), strict=True):
            cols[key].append(v)
        z.append(_exact(ah, al, bh, bl, "div"))
    return record("dd_arith", "div_big_y", cols, {"z": z})


def _split_exact(v: Fraction) -> tuple[float, float] | None:
    """(hi, lo) with hi + lo == v exactly and hi == fl(hi + lo), or None."""
    hi = float(v)
    lo = float(v - Fraction(hi))
    if Fraction(hi) + Fraction(lo) != v or hi + lo != hi:
        return None
    return hi, lo


def _rem_case(rng: random.Random, i: int) -> tuple[float, float, float, float]:
    family = i % 5
    ey = rng.randint(-60, 60)
    if family in (0, 4):  # x = k·y: exact multiples (0) or their double-double roundings (4)
        while True:
            bh, bl = _random_dd(rng, ey)
            if family == 0 and rng.random() < 0.5:
                bl = 0.0
            k = int(_sign(rng)) * rng.randint(1, 2 ** rng.randint(1, 40))
            prod = k * (Fraction(bh) + Fraction(bl))
            if family == 4:
                hi = float(prod)
                lo = float(prod - Fraction(hi))
                if hi + lo == hi:
                    return hi, lo, bh, bl
            elif (split := _split_exact(prod)) is not None:
                return (*split, bh, bl)
    qe = {1: (-3, 12), 2: (12, 60), 3: (60, 200)}[family]
    bh, bl = _random_dd(rng, ey) if i % 2 else _structured_dd(rng, ey)
    ah, al = (_random_dd if i % 3 else _structured_dd)(rng, ey + rng.randint(*qe))
    return ah, al, bh, bl


def _rem(rng: random.Random) -> dict:
    cols: dict[str, list[float]] = {"a_hi": [], "a_lo": [], "b_hi": [], "b_lo": []}
    z = []
    for i in range(N):
        ah, al, bh, bl = _rem_case(rng, i)
        x = Fraction(ah) + Fraction(al)
        y = Fraction(bh) + Fraction(bl)
        r = x - y * int(x / y)  # int() truncates towards zero, as f64 `%` does
        for key, v in zip(cols, (ah, al, bh, bl), strict=True):
            cols[key].append(v)
        z.append(arb(fmpq(r.numerator, r.denominator)))
    return record("dd_arith", "rem", cols, {"z": z})


def generate() -> tuple[int, list[dict]]:
    with ctx.workprec(PRECISION_BITS):
        rng = random.Random(SEED)
        recs = [_basic(op, rng) for op in OPS]
        recs.append(_div_big_y(rng))
        recs.append(_rem(rng))
        return PRECISION_BITS, recs
