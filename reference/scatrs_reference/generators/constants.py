"""Mathematical constants used by DoubleDouble (FloatConst, reduction and series tables)."""

from __future__ import annotations

import math

from flint import arb, ctx, fmpz

from scatrs_reference.records import record, split

PRECISION_BITS = 400


def _constants() -> dict[str, arb]:
    pi, two, ten = arb.pi(), arb(2), arb(10)
    c = {
        "PI": pi,
        "TAU": 2 * pi,
        "FRAC_PI_2": pi / 2,
        "FRAC_PI_3": pi / 3,
        "FRAC_PI_4": pi / 4,
        "FRAC_PI_6": pi / 6,
        "FRAC_PI_8": pi / 8,
        "FRAC_3PI_4": 3 * pi / 4,
        "FRAC_1_PI": 1 / pi,
        "FRAC_2_PI": 2 / pi,
        "FRAC_2_SQRT_PI": 2 / pi.sqrt(),
        "SQRT_2": two.sqrt(),
        "FRAC_1_SQRT_2": 1 / two.sqrt(),
        "E": arb(1).exp(),
        "LN_2": two.log(),
        "LN_10": ten.log(),
        "LOG2_E": 1 / two.log(),
        "LOG10_E": 1 / ten.log(),
        "LOG10_2": two.log() / ten.log(),
        "LOG2_10": ten.log() / two.log(),
    }
    for k in range(31):
        c[f"INV_FACT_{k}"] = 1 / arb(fmpz(math.factorial(k)))
    for i in range(9):
        c[f"POW10_2POW{i}"] = arb(fmpz(10)) ** (1 << i)
    for name, base in (("FRAC_PI_2_TAIL", pi / 2), ("LN_2_TAIL", two.log())):
        hi, lo, _, _ = split(base)
        c[name] = base - arb(hi) - arb(lo)
    return c


def generate() -> tuple[int, list[dict]]:
    with ctx.workprec(PRECISION_BITS):
        recs = [record("constants", name, {}, {"value": [v]}) for name, v in _constants().items()]
    return PRECISION_BITS, recs
