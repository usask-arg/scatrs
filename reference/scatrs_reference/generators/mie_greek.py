"""Phase matrices and Greek coefficients of spheres, from the Arb Mie coefficients.

For each case (x, m, N, L) the generator takes a_n, b_n for n = 1..N from
`scatrs_reference.generators.mie` (finite computations in ball arithmetic), and computes:

* the phase-matrix elements F₁₁ = (|S₁|² + |S₂|²)/2, F₁₂ = (|S₂|² − |S₁|²)/2, F₃₃ = Re(S₂S₁*),
  F₃₄ = Im(S₂S₁*) of the N-term series at the cosines ``COSINES``, normalized by 4/(x²Q_sca)
  with Q_sca the N-term sum;
* the Greek coefficients α₁…α₄, β₁, β₂ for l = 0..L of the N-term series, by Gauss–Legendre
  quadrature on Arb's own Legendre roots (``arb.legendre_p_root``) with n_q = N + ⌈(L + 1)/2⌉
  nodes, which is exact for these polynomial integrands, normalized so that α₁ at l = 0 is 1.
  The Wigner d-functions d⁰⁰, d⁰², d²², d²⁻² come from the three-term recurrence in l in ball
  arithmetic, seeded with the closed forms at the lowest order (checked against the explicit
  sum `wigner_d.d_ref` by the tests).

Everything is a finite computation in ball arithmetic, so every record is certified. The
precision starts at ``PRECISION_BITS`` and doubles until every stored value's radius is below
2^-``REL_BITS`` of its scale (the largest |F₁₁| for the phase matrix, the largest |α₁| for the
Greek coefficients).
"""

from __future__ import annotations

import math

from flint import acb, arb, ctx

from scatrs_reference import disk
from scatrs_reference.generators import mie
from scatrs_reference.records import record

GENERATOR = "mie_greek"
PRECISION_BITS = 320
REL_BITS = 200
MAX_PRECISION_BITS = 2560

#: (x, Re m, Im m, L or None for all orders 2N).
CASES = (
    (0.1, 1.5, 0.0, None),
    (1.0, 1.33, 0.0, None),
    (1.0, 1.5, 0.1, None),
    (3.0, 0.75, 0.0, None),
    (5.0, 1.5, 0.01, None),
    (5.0, 2.5, 1.0, None),
    (10.0, 1.33, 1e-8, None),
    (10.0, 0.2, 3.0, None),
    (25.0, 1.5, 0.001, None),
    (25.0, 1.5, 0.001, 16),
    (50.0, 1.33, 0.0, 40),
    (50.0, 10.0, 10.0, 30),
    (0.5, 0.05, 4.0, None),
    (2.0, 1.0001, 0.0, None),
    (2.0, 1.75, 0.44, None),
    (20.0, 0.75, 0.0, None),
    (100.0, 1.5, 0.01, 50),
    (200.0, 1.33, 0.0, 20),
)

#: Cosines of the phase-matrix records.
COSINES = (1.0, 0.9, 0.5, 0.0, -0.5, -0.9, -1.0)


def default_n(x: float) -> int:
    """The f64 default truncation of scatrs, ⌈x + 6x^(1/3) + 3⌉ (exact for these x)."""
    return math.ceil(x + 6.0 * x ** (1.0 / 3.0) + 3.0)


def pi_tau(mu: arb, n: int) -> tuple[list[arb], list[arb]]:
    """π_k(μ), τ_k(μ) for k = 1..n by the upward recurrences (Bohren & Huffman Eq. 4.47)."""
    pis, taus = [], []
    pi_prev, pi = arb(0), arb(1)
    for k in range(1, n + 1):
        pis.append(pi)
        taus.append(k * mu * pi - (k + 1) * pi_prev)
        pi_prev, pi = pi, ((2 * k + 1) * mu * pi - (k + 1) * pi_prev) / k
    return pis, taus


def elements(s: mie.Solution, mu: arb) -> tuple[arb, arb, arb, arb]:
    """(|S₁|² + |S₂|²)/2, (|S₂|² − |S₁|²)/2, Re(S₂S₁*), Im(S₂S₁*) of the n_ref-term series."""
    pis, taus = pi_tau(mu, s.n_ref)
    s1, s2 = acb(0), acb(0)
    for k in range(1, s.n_ref + 1):
        c = arb(2 * k + 1) / (k * (k + 1))
        a, b = s.a[k - 1], s.b[k - 1]
        s1 += c * (a * pis[k - 1] + b * taus[k - 1])
        s2 += c * (a * taus[k - 1] + b * pis[k - 1])
    n1, n2 = abs(s1) ** 2, abs(s2) ** 2
    p = s2 * s1.conjugate()
    return (n1 + n2) / 2, (n2 - n1) / 2, p.real, p.imag


def wigner_column(m: int, n: int, x: arb, l_max: int) -> list[arb]:
    """d^l_{mn}(x) for l = 0..l_max: the explicit sum at l₀ = max(|m|, |n|), then the
    three-term recurrence (Mishchenko, Travis & Lacis 2002, Eq. B.22)."""
    l0 = max(abs(m), abs(n))
    out = [arb(0)] * (l_max + 1)
    if l_max < l0:
        return out
    prev, cur = arb(0), start_value(m, n, x)
    out[l0] = cur
    for l in range(l0, l_max):
        a = (2 * l + 1) * (l * (l + 1) * x - m * n)
        b = (l + 1) * (arb(l * l - m * m) * (l * l - n * n)).sqrt() if l > 0 else arb(0)
        c = l * (arb((l + 1) ** 2 - m * m) * ((l + 1) ** 2 - n * n)).sqrt()
        if l == 0:
            # d¹ from d⁰ when l₀ = 0 (only m = n = 0): d¹₀₀ = x d⁰₀₀.
            nxt = x * cur
        else:
            nxt = (a * cur - b * prev) / c
        prev, cur = cur, nxt
        out[l + 1] = cur
    return out


def start_value(m: int, n: int, x: arb) -> arb:
    """d^l_{mn}(x) at its lowest order l = max(|m|, |n|), for the four (m, n) used here."""
    if (m, n) == (0, 0):
        return arb(1)
    if (m, n) == (2, 2):
        return ((1 + x) / 2) ** 2  # cos⁴(θ/2)
    if (m, n) == (2, -2):
        return ((1 - x) / 2) ** 2  # sin⁴(θ/2)
    if (m, n) == (0, 2):
        return arb(6).sqrt() / 4 * (1 - x * x)
    raise ValueError((m, n))


def greek(s: mie.Solution, order: int) -> dict[str, list[arb]]:
    """α₁…α₄, β₁, β₂ for l = 0..order, normalized so that α₁ at l = 0 is 1."""
    n_q = s.n_ref + (order + 2) // 2
    sums = {k: [arb(0)] * (order + 1) for k in ("a1", "a4", "plus", "minus", "b1", "b2")}
    for k in range(n_q):
        mu, w = arb.legendre_p_root(n_q, k, weight=True)
        f11, f12, f33, f34 = (w * v for v in elements(s, mu))
        d00 = wigner_column(0, 0, mu, order)
        d22 = wigner_column(2, 2, mu, order)
        d2m2 = wigner_column(2, -2, mu, order)
        d02 = wigner_column(0, 2, mu, order)
        for l in range(order + 1):
            sums["a1"][l] += f11 * d00[l]
            sums["a4"][l] += f33 * d00[l]
            sums["plus"][l] += (f11 + f33) * d22[l]
            sums["minus"][l] += (f11 - f33) * d2m2[l]
            sums["b1"][l] += f12 * d02[l]
            sums["b2"][l] += f34 * d02[l]
    r0 = sums["a1"][0]
    out: dict[str, list[arb]] = {
        k: [] for k in ("alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2")
    }
    for l in range(order + 1):
        # The (2l + 1)/2 factors cancel in the normalization by α₁⁰ = ½·r₀, except for l.
        f = arb(2 * l + 1) / r0
        out["alpha1"].append(f * sums["a1"][l])
        out["alpha4"].append(f * sums["a4"][l])
        out["alpha2"].append(f * (sums["plus"][l] + sums["minus"][l]) / 2)
        out["alpha3"].append(f * (sums["plus"][l] - sums["minus"][l]) / 2)
        out["beta1"].append(f * sums["b1"][l])
        out["beta2"].append(f * sums["b2"][l])
    return out


def phase_matrix(s: mie.Solution) -> dict[str, list[arb]]:
    """F₁₁, F₁₂, F₃₃, F₃₄ at ``COSINES``, normalized by 4/(x²Q_sca)."""
    q_sca = arb(0)
    for k in range(1, s.n_ref + 1):
        q_sca += (2 * k + 1) * (abs(s.a[k - 1]) ** 2 + abs(s.b[k - 1]) ** 2)
    q_sca *= 2 / (s.x * s.x)
    c = 4 / (s.x * s.x * q_sca)
    out: dict[str, list[arb]] = {"f11": [], "f12": [], "f33": [], "f34": []}
    for mu in COSINES:
        for name, v in zip(out, elements(s, arb(mu)), strict=True):
            out[name].append(c * v)
    return out


def _tight(outputs: dict[str, list[arb]], scale: arb) -> bool:
    bound = scale * arb(2) ** -REL_BITS
    return all(v.is_finite() and v.rad() <= bound for vs in outputs.values() for v in vs)


def case_records(x: float, m_re: float, m_im: float, order: int | None) -> list[dict]:
    n = default_n(x)
    l_max = 2 * n if order is None else order
    case = f"x={x!r},m=({m_re!r},{m_im!r}),L={l_max}"
    inputs = {"x": x, "m_re": m_re, "m_im": m_im, "n_terms": n, "order": l_max}
    prec = PRECISION_BITS
    while prec <= MAX_PRECISION_BITS:
        with ctx.workprec(prec):
            try:
                s = mie.solve(x, m_re, m_im, n)
            except disk.DiskContainsZero:  # a disk met a pole of u_k: retry at higher precision
                prec *= 2
                continue
            f = phase_matrix(s)
            g = greek(s, l_max)
            f_scale = max(abs(v) for v in f["f11"])
            g_scale = max(abs(v) for v in g["alpha1"])
            if _tight(f, f_scale) and _tight(g, g_scale):
                # Built at the working precision: `record` splits each value into hi, lo and
                # lo2, which needs more than double-double precision.
                return [
                    record(GENERATOR, case + ",phase", {**inputs, "mu": list(COSINES)}, f),
                    record(GENERATOR, case + ",greek", inputs, g),
                ]
        prec *= 2
    raise ArithmeticError(f"x={x}, m=({m_re}, {m_im}) needs more than {MAX_PRECISION_BITS} bits")


def generate() -> tuple[int, list[dict]]:
    records: list[dict] = []
    for case in CASES:
        records.extend(case_records(*case))
    return PRECISION_BITS, records
