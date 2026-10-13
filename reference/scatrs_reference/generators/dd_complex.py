"""Arb values of complex functions at random Complex<DoubleDouble> arguments.

num_complex computes these from the real DoubleDouble functions: ``exp`` as
e^re·(cos im, sin im), ``ln`` as (ln |z|, arg z), ``sqrt`` as √|z|·(cos, sin)(arg z / 2), and
division as ((ac + bd), (bc − ad))/(c² + d²). Each record holds N random arguments followed by
a few hand-picked ones on the branches of num_complex's ``sqrt`` (zero real or imaginary
part) and next to the branch cut. Arguments are exact normalised double-double pairs with no
negative-zero parts (Arb does not distinguish the sides of a branch cut by the sign of zero).
"""

from __future__ import annotations

import math
import random

from flint import acb, arb, ctx

from scatrs_reference.records import nearest_cos, nearest_pow10, nearest_sin, record

PRECISION_BITS = 600
N = 200
SEED = 62

Pair = tuple[float, float]
Z = tuple[Pair, Pair]


def _dd(rng: random.Random, f: float) -> Pair:
    if f == 0.0:
        return 0.0, 0.0
    lo = rng.uniform(-0.5, 0.5) * math.ulp(f)
    return (f, lo) if f + lo == f else (f, 0.0)


def _polar(rng: random.Random, log10_min: float, log10_max: float) -> Z:
    r = nearest_pow10(rng.uniform(log10_min, log10_max))
    t = rng.uniform(-math.pi, math.pi)
    return _dd(rng, r * nearest_cos(t)), _dd(rng, r * nearest_sin(t))


def _exp_arg(rng: random.Random) -> Z:
    im = rng.uniform(-10.0, 10.0) if rng.random() < 0.6 else rng.uniform(-1e6, 1e6)
    return _dd(rng, rng.uniform(-300.0, 300.0)), _dd(rng, im)


def _exact(p: Pair) -> arb:
    assert p[0] + p[1] == p[0], f"non-normalised input {p}"
    s = arb(p[0]) + arb(p[1])
    assert s.rad() == 0, f"inexact input {p}"
    return s


def _acb(z: Z) -> acb:
    return acb(_exact(z[0]), _exact(z[1]))


def _d(x: float) -> Pair:
    return x, 0.0


EDGES: dict[str, list[Z]] = {
    "exp": [
        (_d(0.0), (math.pi, 1.2246467991473532e-16)),  # e^(iπ) = -1 (to double-double π)
        (_d(1.0), _d(0.0)),
        (_d(-600.0), _d(3.0)),
        (_d(700.0), _d(-3.0)),
    ],
    "ln": [
        (_d(-1.0), _d(0.0)),  # iπ
        (_d(0.0), _d(2.0)),
        (_d(0.0), _d(-2.0)),
        (_d(-3.0), _d(1e-30)),  # next to the branch cut, above
        (_d(-3.0), _d(-1e-30)),  # and below
        (_d(1e-200), _d(1e200)),
    ],
    "sqrt": [
        (_d(-4.0), _d(0.0)),  # 2i: num_complex's zero-imaginary branch
        (_d(9.0), _d(0.0)),
        (_d(0.0), _d(2.0)),  # num_complex's zero-real branch
        (_d(0.0), _d(-8.0)),
        (_d(-1.0), _d(1e-20)),  # next to the branch cut
        (_d(-1.0), _d(-1e-20)),
        (_d(1e-30), _d(1e30)),
    ],
}


def generate() -> tuple[int, list[dict]]:
    with ctx.workprec(PRECISION_BITS):
        rng = random.Random(SEED)
        recs = []
        unary = {
            "exp": (_exp_arg, lambda z: z.exp()),
            "ln": (lambda r: _polar(r, -30.0, 30.0), lambda z: z.log()),
            "sqrt": (lambda r: _polar(r, -30.0, 30.0), lambda z: z.sqrt()),
        }
        for name, (sample, f) in unary.items():
            zs = [sample(rng) for _ in range(N)] + EDGES[name]
            values = [f(_acb(z)) for z in zs]
            recs.append(
                record(
                    "dd_complex",
                    name,
                    {
                        "re_hi": [z[0][0] for z in zs],
                        "re_lo": [z[0][1] for z in zs],
                        "im_hi": [z[1][0] for z in zs],
                        "im_lo": [z[1][1] for z in zs],
                    },
                    {"re": [v.real for v in values], "im": [v.imag for v in values]},
                )
            )
        xs = [_polar(rng, -30.0, 30.0) for _ in range(N)]
        ys = [_polar(rng, -30.0, 30.0) for _ in range(N)]
        quotients = [_acb(x) / _acb(y) for x, y in zip(xs, ys, strict=True)]
        recs.append(
            record(
                "dd_complex",
                "div",
                {
                    "re_hi": [z[0][0] for z in xs],
                    "re_lo": [z[0][1] for z in xs],
                    "im_hi": [z[1][0] for z in xs],
                    "im_lo": [z[1][1] for z in xs],
                    "y_re_hi": [z[0][0] for z in ys],
                    "y_re_lo": [z[0][1] for z in ys],
                    "y_im_hi": [z[1][0] for z in ys],
                    "y_im_lo": [z[1][1] for z in ys],
                },
                {"re": [v.real for v in quotients], "im": [v.imag for v in quotients]},
            )
        )
        return PRECISION_BITS, recs
