"""Gauss–Legendre nodes and weights with certified Arb enclosures (arb.legendre_p_root).

Records ``n=<n>`` hold the whole rule, nodes in ascending order. Records ``n=<n>,partial``
check large rules cheaply: they hold only the nodes at the ascending indices ``indices``,
the ``EDGE`` nodes nearest to each of ±1 and the ``MIDDLE`` nodes nearest to 0 (symmetric
about the middle, so the stored nodes still ascend and are symmetric).

The record ``n=<CAP>,roots`` has the same layout for the largest rule scatrs accepts. That
whole rule takes minutes, so the Rust integration tests skip it; a unit test in
``scatrs/src/special/gauss_legendre.rs`` computes just these roots instead.
"""

from __future__ import annotations

from flint import arb, ctx

from scatrs_reference.records import record

PRECISION_BITS = 300
NS = (1, 2, 3, 4, 5, 8, 16, 33, 64, 100, 128, 257, 500, 1000)
PARTIAL_NS = (5000, 20000)
#: Must equal ``scatrs::special::MAX_GAUSS_LEGENDRE_POINTS`` (the unit test looks it up).
CAP = 2**17
EDGE = 8
MIDDLE = 4


def _partial_indices(n: int) -> list[int]:
    """Ascending node indices: EDGE at each end and MIDDLE (MIDDLE + 1 for odd n) in the middle."""
    lo = n // 2 - MIDDLE // 2
    middle = range(lo, n - lo)
    return sorted({*range(EDGE), *middle, *range(n - EDGE, n)})


def _rule(n: int, indices: list[int]) -> tuple[list[arb], list[arb]]:
    """Nodes and weights at the ascending indices (Arb indexes the roots in decreasing order)."""
    roots = [arb.legendre_p_root(n, n - 1 - i, weight=True) for i in indices]
    return [x for x, _ in roots], [w for _, w in roots]


def generate() -> tuple[int, list[dict]]:
    with ctx.workprec(PRECISION_BITS):
        recs = []
        for n in NS:
            nodes, weights = _rule(n, list(range(n)))
            recs.append(
                record("gauss_legendre", f"n={n}", {"n": n}, {"nodes": nodes, "weights": weights})
            )
        for n, suffix in [*((n, "partial") for n in PARTIAL_NS), (CAP, "roots")]:
            indices = _partial_indices(n)
            nodes, weights = _rule(n, indices)
            recs.append(
                record(
                    "gauss_legendre",
                    f"n={n},{suffix}",
                    {"n": n, "indices": indices},
                    {"nodes": nodes, "weights": weights},
                )
            )
        return PRECISION_BITS, recs
