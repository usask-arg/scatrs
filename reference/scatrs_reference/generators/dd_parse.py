"""Exact values of decimal literals, for DoubleDouble::from_decimal_str.

The record ``literals`` holds N random literals with 1 to 40 significant digits, a decimal
point in any position (or none), a sign and an exponent, whose values lie in
(10^-289, 10^289). The record ``literals_edges`` holds hand-picked literals: the ends of
the full-precision range [2^-969, MAX] (including values next to the f64 overflow
threshold), exact powers of ten and integers next to 2^106, mantissas whose 39th
significant digit rounds up or down, long strings of digits or zeros (which must not
overflow or underflow early), and every accepted form of the decimal point and exponent.
"""

from __future__ import annotations

import random
from fractions import Fraction

from flint import arb, ctx, fmpq

from scatrs_reference.records import record

PRECISION_BITS = 600
N = 300
SEED = 6

#: 2^-969, below which a double-double loses precision.
MIN_FULL = Fraction(1, 2**969)
#: The largest finite double.
MAX = Fraction((2**53 - 1) * 2**971)

PI_120 = (
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
    "862803482534211706798214808651328230664709"
)

EDGE_LITERALS = [
    # The ends of the full-precision range.
    "1.7976931348623157e308",  # f64::MAX, shortest form
    str(MAX.numerator),  # f64::MAX, every digit
    "1.797693134862315807e308",  # just below the f64 overflow threshold MAX + ulp/2
    "-1.7976931348623158e308",  # rounds to -MAX as an f64
    "9.999999999999999999999999999999999999999e307",
    "1e308",
    "1e289",
    "1e-289",
    "1e-291",
    "2.0041683600089728e-292",  # just above 2^-969
    # Exact powers of ten and integers next to 2^106.
    "1e22",
    "1e23",
    "1e-22",
    "1e-23",
    "1e31",
    "1e32",
    "1e45",
    "1e46",
    "1e-45",
    "81129638414606681695789005144063",  # 2^106 - 1
    "81129638414606681695789005144065",  # 2^106 + 1
    "170141183460469231731687303715884105727",  # 2^127 - 1
    # Rounding of the 38-digit mantissa.
    "99999999999999999999999999999999999999",  # 38 nines: kept exactly
    "999999999999999999999999999999999999999",  # 39 nines: rounds up to 10^39
    "9.9999999999999999999999999999999999999999e10",
    "-1.23456789012345678901234567890123456785e-7",  # 39th digit 5: rounds up
    "1.234567890123456789012345678901234567849999e100",  # 39th digit 4: rounds down
    "1.000000000000000000000000000000000000000000000000000000000000000000000001",
    PI_120,
    "-" + PI_120 + "e-200",
    # Long strings: leading zeros and dropped digits must not underflow or overflow early.
    "0." + "0" * 399 + "1e410",  # 10^10
    "1" + "0" * 400 + "e-400",  # 1
    "0." + "0" * 300 + "123456789012345678901234567890123456789012345e310",
    "000000000000000000000000000000000000000000012.5",
    "1" * 300 + "e-280",
    # Every accepted form of the point and the exponent.
    "0.1",
    "-0.2",
    "+0.3",
    ".5",
    "5.",
    "1.e1",
    "1E+10",
    "1e-0",
    "1e+" + "0" * 40 + "100",
    "6.02214076e23",
    "1.602176634e-19",
    # Dyadic values, exact in double precision.
    "0.25",
    "2.5",
    "-0.125",
    "4.76837158203125e-7",  # 2^-21
]


def _literal(rng: random.Random) -> tuple[str, Fraction]:
    while True:
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 40))).lstrip("0") or "1"
        point = rng.randint(0, len(digits))
        mantissa = digits[:point] + "." + digits[point:] if rng.random() < 0.7 else digits
        frac_digits = len(digits) - point if "." in mantissa else 0
        exponent = rng.randint(-280, 280)
        value = Fraction(int(digits), 10**frac_digits) * Fraction(10) ** exponent
        if Fraction(1, 10**289) < value < Fraction(10**289):
            sign = rng.choice(("", "-", "+"))
            text = f"{sign}{mantissa}e{exponent}"
            return text, -value if sign == "-" else value


def _as_arb(v: Fraction) -> arb:
    return arb(fmpq(v.numerator, v.denominator))


def _edges() -> list[tuple[str, Fraction]]:
    out = [(text, Fraction(text)) for text in EDGE_LITERALS]
    for text, v in out:
        assert MIN_FULL <= abs(v) < 2 * MAX, f"{text} is outside the full-precision range"
    return out


def generate() -> tuple[int, list[dict]]:
    with ctx.workprec(PRECISION_BITS):
        rng = random.Random(SEED)
        recs = []
        for case, pairs in (("literals", [_literal(rng) for _ in range(N)]), ("literals_edges", _edges())):
            values = [_as_arb(v) for _, v in pairs]
            recs.append(record("dd_parse", case, {"text": [t for t, _ in pairs]}, {"value": values}))
        return PRECISION_BITS, recs
