"""Exact results of double-double basic operations on random normalised inputs."""

from __future__ import annotations

import math
import random

from flint import arb, ctx

from scatrs_reference.records import record

PRECISION_BITS = 600
N = 1000
SEED = 3
OPS = ("add", "sub", "mul", "div", "add_f64", "mul_f64", "div_f64")


def _random_dd(rng: random.Random) -> tuple[float, float]:
    hi = rng.choice((-1.0, 1.0)) * rng.uniform(1.0, 2.0) * 2.0 ** rng.randint(-60, 60)
    lo = rng.uniform(-0.5, 0.5) * math.ulp(hi)
    return (hi, lo) if hi + lo == hi else (hi, 0.0)


def generate() -> tuple[int, list[dict]]:
    with ctx.workprec(PRECISION_BITS):
        rng = random.Random(SEED)
        recs = []
        for op in OPS:
            a_hi, a_lo, b_hi, b_lo, z = [], [], [], [], []
            for i in range(N):
                ah, al = _random_dd(rng)
                if op in ("add", "sub") and i % 10 == 0:
                    # Massive cancellation: b = ∓a with a different low word.
                    bh = -ah if op == "add" else ah
                    bl = rng.uniform(-0.5, 0.5) * math.ulp(bh)
                else:
                    bh, bl = _random_dd(rng)
                if op.endswith("_f64"):
                    bl = 0.0
                a, b = arb(ah) + arb(al), arb(bh) + arb(bl)
                exact = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[
                    op.removesuffix("_f64")
                ]
                a_hi.append(ah)
                a_lo.append(al)
                b_hi.append(bh)
                b_lo.append(bl)
                z.append(exact)
            recs.append(
                record(
                    "dd_arith",
                    op,
                    {"a_hi": a_hi, "a_lo": a_lo, "b_hi": b_hi, "b_lo": b_lo},
                    {"z": z},
                )
            )
        return PRECISION_BITS, recs
