"""Arb reference values for Lorenz–Mie scattering by homogeneous spheres.

Conventions are those of Bohren & Huffman (1983), with time dependence e^(−iωt): the
refractive index m = n + ik has k ≥ 0 for absorbing spheres, and

    a_n = [(D_n(mx)/m + n/x) ψ_n(x) − ψ_{n−1}(x)] / [(D_n(mx)/m + n/x) ξ_n(x) − ξ_{n−1}(x)],

b_n likewise with m·D_n(mx) in place of D_n(mx)/m. Here ψ_n(ρ) = ρ j_n(ρ), ξ_n = ψ_n − iχ_n with
χ_n(ρ) = −ρ y_n(ρ), and D_n = ψ_n′/ψ_n. Each value is computed at the exact double inputs.

Method. Every step is rigorous; nothing calls Arb's Bessel functions, which lose most of their
accuracy at large orders (5 of 320 bits at order 2000):

* u_k(ρ) = ψ_k(ρ)/ψ_{k−1}(ρ) for k ≤ n + 1, by the downward recurrence
  u_k = 1/((2k + 1)/ρ − u_{k+1}), started at k₀ ≥ 2|ρ| from the enclosure |u_{k₀+1}| ≤ 1. The
  enclosure holds because u ↦ 1/(c − u) maps the unit disk into itself when |c| ≥ 2, and the
  continued fraction converges to u_k (Pincherle's theorem: ψ is the minimal solution of the
  recurrence). Each step from k ≥ 2|ρ| shrinks the uncertainty at least ninefold. Then
  D_n(ρ) = (n + 1)/ρ − u_{n+1}(ρ). Complex ρ uses disk arithmetic (`scatrs_reference.disk`).
* ψ_n(x) = ψ_{n−1}(x)·u_n(x) from ψ₀ = sin x; ξ_n = ξ_{n−1}·w_n, with
  w_n = ξ_n/ξ_{n−1} = (2n − 1)/x − 1/w_{n−1} from w₀ = ξ₀/ξ₋₁ = −i, in disk arithmetic.
* a_n and b_n from the formulas above, and their derivatives from the closed forms
  ∂a_n/∂m = i(D − zD′)/(m²Δ_a²), ∂b_n/∂m = −i(D + zD′)/Δ_b²,
  ∂a_n/∂x = i(m² − 1)(n(n + 1)/x² + D²)/(m²Δ_a²) and ∂b_n/∂x = i(m² − 1)/Δ_b², where z = mx,
  D = D_n(z), D′ = n(n + 1)/z² − 1 − D² and Δ_a, Δ_b are the denominators of a_n and b_n.
  ``tests/test_mie.py`` checks them against Arb central differences.
* The series Q_ext, Q_sca, Q_abs = Q_ext − Q_sca (exactly 0 for real m), Q_back, g, S₁ and S₂
  are summed to n_ref terms. π_n and τ_n come from Arb's Legendre polynomials:
  π_n = n(μP_n − P_{n−1})/(μ² − 1) and τ_n = n(n + 1)P_n − μπ_n, with the closed forms
  π_n(±1) = (±1)^(n+1) n(n + 1)/2 at μ = ±1.

Each case is computed at ``PRECISION_BITS``, and the precision is doubled until every stored
value has a radius of at most 2^-``REL_BITS`` of its scale (the largest value of its array; for
Q_abs, Q_ext; for Q_back, Q_sca; for S₁ and S₂, the RMS amplitude x√Q_sca/2).

Records. For each case ``x=…,m=(…, …)`` the file holds a record ``…,coefficients`` with a_n and
b_n at the orders ``orders`` (``mie.jsonl``; with the four derivatives in
``mie_derivatives.jsonl``), and a record ``…,series`` with the series at the cosines ``mu``
(and, in ``mie_derivatives.jsonl``, their derivatives with respect to x, Re m and Im m; for S₁
and S₂, with respect to x and m).

Labels. Coefficient records are finite computations, so certified. A series record of values
is certified when n_ref is large enough for the tail bound of `tail_bounds` (proved in the
theory chapter, "Series tail bound"): 2n_ref + 3 ≥ 2|m|x and n_ref ≥ 1.5x − ½. The bound is
then added to each value's error. Otherwise, when that would take more than
``CERTIFY_MAX_ORDER`` terms, the series is summed to ⌈x + 18x^(1/3) + 32⌉ terms and labelled
high-precision. The series of derivatives have no tail bound yet: they are high-precision.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from flint import acb, arb, ctx

from scatrs_reference import disk
from scatrs_reference.records import PORTABLE_BITS, nearest_double, nearest_pow10, record

PRECISION_BITS = 320
REL_BITS = 200
MAX_PRECISION_BITS = 2560
#: Series needing more terms than this to certify are summed to the high-precision length.
CERTIFY_MAX_ORDER = 60_000

XS = (
    1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
    2000.0, 5000.0, 10000.0, 20000.0,
)  # fmt: skip
#: (Re m, Im m): near-unity, water and aerosol, high index, strong absorption and metal-like.
MS = (
    (1.0001, 0.0), (1.01, 0.0), (1.33, 0.0), (1.33, 1e-8), (1.5, 1e-3), (1.5, 0.1),
    (1.75, 0.44), (2.5, 1.0), (4.0, 0.01), (10.0, 0.0), (10.0, 10.0), (1.5, 5.0), (0.2, 3.0),
    (0.05, 4.0),
)  # fmt: skip
RANDOM_CASES = 40
SEED = 20261013
#: Scattering angles of the amplitude records, in degrees.
ANGLES_DEG = (0.0, 0.5, 5.0, 30.0, 60.0, 90.0, 135.0, 170.0, 180.0)
DERIVATIVE_XS = (1e-6, 1e-2, 0.5, 2.0, 10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0)
DERIVATIVE_MS = (
    (1.0001, 0.0), (1.33, 0.0), (1.33, 1e-8), (1.5, 0.1), (2.5, 1.0), (10.0, 10.0), (0.2, 3.0),
    (1.5, 5.0),
)  # fmt: skip

Case = tuple[float, float, float]


def size_terms(x: float, c: int, d: int) -> int:
    """⌈x + c·x^(1/3) + d⌉, computed exactly (scatrs keeps (c, d) = (6, 3) terms in f64)."""
    with ctx.workprec(PORTABLE_BITS):
        v = arb(x) + c * arb(x).root(3) + d
        return int(v.ceil().unique_fmpz())


def orders(n: int) -> list[int]:
    """The orders whose coefficients a record stores: all of 1..n for n ≤ 12, else 11 of them."""
    if n <= 12:
        return list(range(1, n + 1))
    return sorted({1, 2, 3, n // 64, n // 16, n // 4, n // 2, 3 * n // 4, n - 2, n - 1, n} - {0})


def cosines() -> list[float]:
    """cos θ for θ in ``ANGLES_DEG``, rounded to the nearest double (exactly 0 at 90°)."""
    with ctx.workprec(PORTABLE_BITS):
        return [nearest_double((arb(a) / 180).cos_pi()) for a in ANGLES_DEG]


def _ceil_abs(v: arb | acb) -> int:
    """An integer ≥ |w| for every w in the (finite) ball ``v``."""
    return int(abs(v).abs_upper().ceil().unique_fmpz())


def ratios_real(x: arb, n_top: int) -> list[arb]:
    """u_k(x) = ψ_k(x)/ψ_{k−1}(x) for k = 1, …, n_top + 1, at index k (index 0 is unused).

    A ball that straddles a pole of u_k (a zero of ψ_{k−1}) makes the values NaN.
    """
    start = max(n_top + 1, 2 * _ceil_abs(x)) + ctx.prec // 2
    out: list[arb] = [arb(0)] * (n_top + 2)
    inv_x = 1 / x
    u = arb(0, 1)  # encloses u_{start+1}
    for k in range(start, 0, -1):
        u = 1 / ((2 * k + 1) * inv_x - u)
        if k <= n_top + 1:
            out[k] = u
    return out


def ratios_complex(z: acb, n_top: int) -> list[acb]:
    """u_k(z) = ψ_k(z)/ψ_{k−1}(z) for k = 1, …, n_top + 1, at index k, in disk arithmetic.

    A disk that may contain a pole of u_k raises `disk.DiskContainsZero`.
    """
    start = max(n_top + 1, 2 * _ceil_abs(z)) + ctx.prec // 2
    out: list[acb] = [acb(0)] * (n_top + 2)
    inv_z = 1 / z
    u = disk.Disk(acb(0), arb(1))  # encloses u_{start+1}
    for k in range(start, 0, -1):
        u = disk.inv(disk.sub_from((2 * k + 1) * inv_z, u))
        if k <= n_top + 1:
            out[k] = disk.ball(u)
    return out


@dataclass
class Solution:
    """Mie coefficients of one sphere for n = 1..n_ref (index n − 1), at the working precision."""

    x: arb
    m: acb
    n_ref: int
    a: list[acb]
    b: list[acb]
    da_dx: list[acb]
    da_dm: list[acb]
    db_dx: list[acb]
    db_dm: list[acb]
    #: D_n(mx).
    d: list[acb]
    #: The denominator Δ_a of a_n.
    delta_a: list[acb]
    #: The denominator Δ_b of b_n.
    delta_b: list[acb]
    #: T = ψ_N/ξ_N at N = n_ref, for the tail bound.
    t_last: acb
    #: R = ξ_{N−1}/ξ_N at N = n_ref, for the tail bound.
    r_last: acb


def solve(x: float, m_re: float, m_im: float, n_ref: int) -> Solution:
    """a_n, b_n and their derivatives for n = 1..n_ref at the current working precision."""
    return solve_exact(arb(x), acb(m_re, m_im), n_ref)


def solve_exact(xa: arb, m: acb, n_ref: int) -> Solution:
    """`solve` at the exact (ball) inputs ``xa`` and ``m``; real m (an exactly zero imaginary
    part) takes the real recurrence for D_n(mx)."""
    z = m * xa
    ux = ratios_real(xa, n_ref)
    if m.imag == 0:
        uz = [acb(u) for u in ratios_real(m.real * xa, n_ref)]
    else:
        uz = ratios_complex(z, n_ref)
    sin_x, cos_x = xa.sin(), xa.cos()
    psi_prev, psi = cos_x, sin_x  # ψ₋₁, ψ₀
    xi = disk.disk(acb(sin_x, -cos_x))  # ξ₀
    w = disk.Disk(acb(0, -1), arb(0))  # w₀ = ξ₀/ξ₋₁
    i = acb(0, 1)
    m2 = m * m
    s = Solution(
        x=xa,
        m=m,
        n_ref=n_ref,
        a=[],
        b=[],
        da_dx=[],
        da_dm=[],
        db_dx=[],
        db_dm=[],
        d=[],
        delta_a=[],
        delta_b=[],
        t_last=acb(0),
        r_last=acb(0),
    )
    xi_prev = disk.ball(xi)
    xin = xi_prev
    for n in range(1, n_ref + 1):
        psi_prev, psi = psi, psi * ux[n]
        w = disk.sub_from(acb((2 * n - 1) / xa), disk.inv(w))
        xi_prev = disk.ball(xi)
        xi = disk.mul(xi, w)
        xin = disk.ball(xi)
        d = (n + 1) / z - uz[n + 1]
        big_a = d / m + n / xa
        big_b = m * d + n / xa
        delta_a = big_a * xin - xi_prev
        delta_b = big_b * xin - xi_prev
        s.a.append((big_a * psi - psi_prev) / delta_a)
        s.b.append((big_b * psi - psi_prev) / delta_b)
        dp = n * (n + 1) / (z * z) - 1 - d * d
        s.da_dm.append(i * (d - z * dp) / (m2 * delta_a * delta_a))
        s.db_dm.append(-i * (d + z * dp) / (delta_b * delta_b))
        s.da_dx.append(i * (m2 - 1) * (n * (n + 1) / (xa * xa) + d * d) / (m2 * delta_a * delta_a))
        s.db_dx.append(i * (m2 - 1) / (delta_b * delta_b))
        s.d.append(d)
        s.delta_a.append(delta_a)
        s.delta_b.append(delta_b)
    s.t_last = psi / xin
    s.r_last = xi_prev / xin
    return s


def _pi_tau(mu: float, n_ref: int) -> tuple[list[arb], list[arb]]:
    """π_n(μ) and τ_n(μ) for n = 1..n_ref (index n − 1)."""
    mu_a = arb(mu)
    pis, taus = [], []
    if mu in (1.0, -1.0):
        for n in range(1, n_ref + 1):
            sign = 1 if mu == 1.0 or n % 2 == 1 else -1
            pi_n = arb(sign * n * (n + 1)) / 2
            pis.append(pi_n)
            taus.append(n * (n + 1) * mu_a.legendre_p(n) - mu_a * pi_n)
        return pis, taus
    p_prev = arb(1)
    for n in range(1, n_ref + 1):
        p = mu_a.legendre_p(n)
        pi_n = n * (mu_a * p - p_prev) / (mu_a * mu_a - 1)
        pis.append(pi_n)
        taus.append(n * (n + 1) * p - mu_a * pi_n)
        p_prev = p
    return pis, taus


@dataclass
class Sums:
    """Partial sums of the series up to n_ref."""

    q_ext: arb
    q_sca: arb
    s_back: acb  # Σ (2n + 1)(−1)^n (a_n − b_n)
    g_sum: arb  # G, with g = 4G/(x² Q_sca)
    s1: list[acb]
    s2: list[acb]
    pi_tau: list[tuple[list[arb], list[arb]]]


def sum_series(s: Solution, mus: Sequence[float]) -> Sums:
    """The partial sums of every series of `series_outputs`, to n_ref terms."""
    two_x2 = 2 / (s.x * s.x)
    q_ext, q_sca, s_back, g_sum = arb(0), arb(0), acb(0), arb(0)
    n_ref = s.n_ref
    for n in range(1, n_ref + 1):
        a, b = s.a[n - 1], s.b[n - 1]
        q_ext += (2 * n + 1) * (a + b).real
        q_sca += (2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)
        sign = 1 if n % 2 == 0 else -1
        s_back += (2 * n + 1) * sign * (a - b)
        if n < n_ref:
            a1, b1 = s.a[n], s.b[n]
            g_sum += (
                arb(n * (n + 2)) / (n + 1) * ((a * a1.conjugate()).real + (b * b1.conjugate()).real)
            )
        g_sum += arb(2 * n + 1) / (n * (n + 1)) * (a * b.conjugate()).real
    s1, s2, pts = [], [], []
    for mu in mus:
        pis, taus = _pi_tau(mu, n_ref)
        t1, t2 = acb(0), acb(0)
        for n in range(1, n_ref + 1):
            c = arb(2 * n + 1) / (n * (n + 1))
            a, b = s.a[n - 1], s.b[n - 1]
            t1 += c * (a * pis[n - 1] + b * taus[n - 1])
            t2 += c * (a * taus[n - 1] + b * pis[n - 1])
        s1.append(t1)
        s2.append(t2)
        pts.append((pis, taus))
    return Sums(two_x2 * q_ext, two_x2 * q_sca, s_back, g_sum, s1, s2, pts)


@dataclass
class Tail:
    """|a_n| + |b_n| ≤ amplitude·ratio^(n − N) for every n > N = n_ref."""

    amplitude: arb
    ratio: arb


def tail_bounds(s: Solution) -> Tail | None:
    """The geometric bound on the coefficients beyond n_ref, or None where it does not apply.

    See the theory chapter, "Series tail bound". With N = n_ref, U_z bounds |u_k(mx)|, U_x
    bounds u_k(x) and R bounds |ξ_{k−1}/ξ_k| for all k > N; then |T_{n+1}| ≤ U_x·R·|T_n| for
    T_n = ψ_n/ξ_n, and the factors that turn T_n into a_n and b_n are bounded by F_a and F_b.
    """
    x, m, big_n = s.x, s.m, s.n_ref
    z_abs = abs(m * x)
    m_abs = abs(m)
    if not (2 * big_n + 3 >= 2 * z_abs and (2 * big_n + 1) / x >= 3 and abs(s.r_last) <= 1):
        return None
    u_z = 1 / ((2 * big_n + 3) / z_abs - 1)
    u_x = 1 / ((2 * big_n + 3) / x - 1)
    r_b = 1 / ((2 * big_n + 1) / x - 1)
    inv_m2 = 1 / (m * m)
    c1 = abs(inv_m2 - 1) / x
    e1 = u_z / m_abs + u_x
    c2 = abs(1 + inv_m2) / x
    e2 = 1 / (m_abs * m_abs * x) + u_z / m_abs + r_b
    den_a = (big_n + 1) * c2 - e2
    d_b = (2 * big_n + 3) / x - m_abs * u_z - r_b
    if not (den_a > 0 and d_b > 0):
        return None
    # f(n) = ((n + 1)c1 + e1)/(n c2 − e2) is monotone for n ≥ N + 1, so its supremum is at most
    # the sum of its value at N + 1 and its limit c1/c2.
    f_a = ((big_n + 2) * c1 + e1) / den_a + c1 / c2
    f_b = (u_x + m_abs * u_z) / d_b
    amplitude = abs(s.t_last) * (f_a + f_b)
    if not amplitude.is_finite():
        return None
    return Tail(amplitude, u_x * r_b)


def geometric_tail(weight: Callable[[int], arb], n0: int, amplitude: arb, ratio: arb) -> arb | None:
    """A bound on Σ_{n ≥ n0} weight(n)·amplitude·ratio^(n − n0 + 1).

    Valid for positive weights whose ratio weight(n + 1)/weight(n) does not increase with n
    (products of factors a·n + b with a, b ≥ 0). None if the terms do not decrease geometrically.
    """
    w0 = weight(n0)
    r = ratio * (weight(n0 + 1) / w0)
    if not r < 1:
        return None
    return w0 * amplitude * ratio / (1 - r)


def _grow(v: arb, t: arb) -> arb:
    return v + arb(0, t)


def _grow_c(v: acb, t: arb) -> acb:
    return v + acb(arb(0, t), arb(0, t))


def series_outputs(s: Solution, sums: Sums, tail: Tail | None) -> dict[str, list] | None:
    """Q_ext, Q_sca, Q_abs, Q_back, g and S₁, S₂, each grown by its tail bound if ``tail``.

    None if a tail sum does not converge geometrically.
    """
    x2 = s.x * s.x
    q_ext, q_sca, s_back, g_sum = sums.q_ext, sums.q_sca, sums.s_back, sums.g_sum
    s1, s2 = list(sums.s1), list(sums.s2)
    if tail is not None:
        n0 = s.n_ref + 1
        c, q = tail.amplitude, tail.ratio
        t_lin = geometric_tail(lambda n: arb(2 * n + 1), n0, c, q)
        t_sq = geometric_tail(lambda n: arb(2 * n + 1), n0, c * c, q * q)
        t_g1 = geometric_tail(lambda n: arb(n + 1), n0, c * c, q * q)
        t_g2 = geometric_tail(lambda n: arb(1), n0, c * c, q * q)
        t_s = geometric_tail(lambda n: arb((2 * n + 1) * n * (n + 1)) / 4, n0, c, q)
        if None in (t_lin, t_sq, t_g1, t_g2, t_s):
            return None
        last = abs(s.a[-1]) + abs(s.b[-1])
        q_ext = _grow(q_ext, 2 * t_lin / x2)
        q_sca = _grow(q_sca, 2 * t_sq / x2)
        s_back = _grow_c(s_back, t_lin)
        g_sum = _grow(g_sum, (s.n_ref + 1) * last * c * q + q * t_g1 + t_g2)
        s1 = [_grow_c(v, t_s) for v in s1]
        s2 = [_grow_c(v, t_s) for v in s2]
    q_abs = arb(0) if s.m.imag == 0 else q_ext - q_sca
    return {
        "q_ext": [q_ext],
        "q_sca": [q_sca],
        "q_abs": [q_abs],
        "q_back": [abs(s_back) ** 2 / x2],
        "g": [4 * g_sum / (x2 * q_sca)],
        "s1_re": [v.real for v in s1],
        "s1_im": [v.imag for v in s1],
        "s2_re": [v.real for v in s2],
        "s2_im": [v.imag for v in s2],
    }


def series_derivatives(s: Solution, sums: Sums) -> dict[str, list]:
    """∂/∂x, ∂/∂Re m and ∂/∂Im m of the five efficiencies, and ∂/∂x and d/dm of S₁ and S₂.

    Partial sums to n_ref terms, at the cosines of ``sums``.
    """
    x = s.x
    x2 = x * x
    i = acb(0, 1)
    q_sca = sums.q_sca
    q_back = abs(sums.s_back) ** 2 / x2
    g = 4 * sums.g_sum / (x2 * q_sca)
    q_ext = sums.q_ext
    out: dict[str, list] = {}
    # ∂a/∂p for p = x, Re m, Im m.
    da = {"dx": s.da_dx, "dm_re": s.da_dm, "dm_im": [i * v for v in s.da_dm]}
    db = {"dx": s.db_dx, "dm_re": s.db_dm, "dm_im": [i * v for v in s.db_dm]}
    for p in ("dx", "dm_re", "dm_im"):
        d_ext, d_sca, d_back, d_g = arb(0), arb(0), acb(0), arb(0)
        n_ref = s.n_ref
        for n in range(1, n_ref + 1):
            a, b, pa, pb = s.a[n - 1], s.b[n - 1], da[p][n - 1], db[p][n - 1]
            d_ext += (2 * n + 1) * (pa + pb).real
            d_sca += (2 * n + 1) * 2 * ((a.conjugate() * pa).real + (b.conjugate() * pb).real)
            sign = 1 if n % 2 == 0 else -1
            d_back += (2 * n + 1) * sign * (pa - pb)
            if n < n_ref:
                a1, b1, pa1, pb1 = s.a[n], s.b[n], da[p][n], db[p][n]
                d_g += (
                    arb(n * (n + 2))
                    / (n + 1)
                    * (
                        (pa * a1.conjugate() + a * pa1.conjugate()).real
                        + (pb * b1.conjugate() + b * pb1.conjugate()).real
                    )
                )
            d_g += arb(2 * n + 1) / (n * (n + 1)) * (pa * b.conjugate() + a * pb.conjugate()).real
        d_ext *= 2 / x2
        d_sca *= 2 / x2
        d_back_q = 2 * (sums.s_back.conjugate() * d_back).real / x2
        if p == "dx":
            d_ext -= 2 * q_ext / x
            d_sca -= 2 * q_sca / x
            d_back_q -= 2 * q_back / x
        d_g_out = 4 * d_g / (x2 * q_sca) - g * d_sca / q_sca
        if p == "dx":
            d_g_out -= 2 * g / x
        out[f"q_ext_{p}"] = [d_ext]
        out[f"q_sca_{p}"] = [d_sca]
        # Q_abs is identically 0 for real m, so its derivatives along x and Re m are too.
        real_m = s.m.imag == 0 and p != "dm_im"
        out[f"q_abs_{p}"] = [arb(0) if real_m else d_ext - d_sca]
        out[f"q_back_{p}"] = [d_back_q]
        out[f"g_{p}"] = [d_g_out]
    for key, coeff_a, coeff_b in (("dx", s.da_dx, s.db_dx), ("dm", s.da_dm, s.db_dm)):
        d1, d2 = [], []
        for pis, taus in sums.pi_tau:
            t1, t2 = acb(0), acb(0)
            for n in range(1, s.n_ref + 1):
                c = arb(2 * n + 1) / (n * (n + 1))
                pa, pb = coeff_a[n - 1], coeff_b[n - 1]
                t1 += c * (pa * pis[n - 1] + pb * taus[n - 1])
                t2 += c * (pa * taus[n - 1] + pb * pis[n - 1])
            d1.append(t1)
            d2.append(t2)
        out[f"s1_{key}_re"] = [v.real for v in d1]
        out[f"s1_{key}_im"] = [v.imag for v in d1]
        out[f"s2_{key}_re"] = [v.real for v in d2]
        out[f"s2_{key}_im"] = [v.imag for v in d2]
    return out


def _tight(outputs: dict[str, list], scales: dict[str, arb]) -> bool:
    """Whether every value's radius is at most 2^-REL_BITS of its scale (default: its array's
    largest midpoint)."""
    eps = arb(2) ** -REL_BITS
    for name, vs in outputs.items():
        scale = scales.get(name)
        if scale is None:
            scale = max((abs(v.mid()) for v in vs), default=arb(0))
        for v in vs:
            if not (v.is_finite() and v.rad() <= eps * max(abs(v.mid()), scale)):
                return False
    return True


def _coefficient_outputs(s: Solution, ns: Sequence[int], derivatives: bool) -> dict[str, list]:
    if derivatives:
        arrays = {"da_dx": s.da_dx, "da_dm": s.da_dm, "db_dx": s.db_dx, "db_dm": s.db_dm}
    else:
        arrays = {"a": s.a, "b": s.b}
    out: dict[str, list] = {}
    for name, vs in arrays.items():
        out[f"{name}_re"] = [vs[n - 1].real for n in ns]
        out[f"{name}_im"] = [vs[n - 1].imag for n in ns]
    return out


def _array_scales(out: dict[str, list]) -> dict[str, arb]:
    """One scale per complex array, shared by its real and imaginary parts: the largest
    |re| + |im| over the array (an upper bound on its largest modulus)."""
    scales: dict[str, arb] = {}
    for name in out:
        if name.endswith("_re"):
            stem = name[:-3]
            re, im = out[f"{stem}_re"], out[f"{stem}_im"]
            scale = max((abs(r.mid()) + abs(i.mid()) for r, i in zip(re, im, strict=True)))
            scales[f"{stem}_re"] = scales[f"{stem}_im"] = scale
    return scales


def n_ref(x: float, m_re: float, m_im: float) -> tuple[int, bool]:
    """(terms to sum, whether the tail bound can certify the series at that length)."""
    n_hp = size_terms(x, 18, 32)
    with ctx.workprec(PORTABLE_BITS):
        n_z = _ceil_abs(acb(m_re, m_im) * arb(x)) + 1
        n_x = int((3 * arb(x) / 2).ceil().unique_fmpz()) + 1
    n_cert = max(n_hp, n_z, n_x)
    return (n_cert, True) if n_cert <= CERTIFY_MAX_ORDER else (n_hp, False)


def case_name(x: float, m_re: float, m_im: float) -> str:
    return f"x={x!r},m=({m_re!r},{m_im!r})"


def _case_records(case: Case, derivatives: bool) -> list[dict]:
    x, m_re, m_im = case
    generator = "mie_derivatives" if derivatives else "mie"
    name = case_name(x, m_re, m_im)
    ns = orders(size_terms(x, 6, 3))
    mus = cosines()
    terms, certify = n_ref(x, m_re, m_im)
    prec = PRECISION_BITS
    while prec <= MAX_PRECISION_BITS:
        with ctx.workprec(prec):
            try:
                s = solve(x, m_re, m_im, terms)
            except disk.DiskContainsZero:  # a disk met a pole of u_k: retry at higher precision
                prec *= 2
                continue
            coefficients = _coefficient_outputs(s, ns, derivatives)
            sums = sum_series(s, mus)
            if derivatives:
                series, label = series_derivatives(s, sums), "high-precision"
                # A derivative of Q_abs or Q_back may be tiny or zero; measure it against the
                # derivatives of Q_ext and Q_sca.
                series_scales: dict[str, arb] = {}
                for p in ("dx", "dm_re", "dm_im"):
                    ext, sca = series[f"q_ext_{p}"][0].mid(), series[f"q_sca_{p}"][0].mid()
                    scale = max(abs(ext), abs(sca))
                    series_scales[f"q_abs_{p}"] = series_scales[f"q_back_{p}"] = scale
            else:
                tail = tail_bounds(s) if certify else None
                series = series_outputs(s, sums, tail)
                if series is None:
                    series = series_outputs(s, sums, None)
                    tail = None
                assert series is not None
                label = "certified" if tail is not None else "high-precision"
                q_ext, q_sca = series["q_ext"][0], series["q_sca"][0]
                s_rms = s.x * q_sca.sqrt() / 2
                series_scales = {"q_abs": abs(q_ext.mid()), "q_back": abs(q_sca.mid())}
                series_scales |= {k: abs(s_rms.mid()) for k in series if k.startswith("s")}
            if _tight(coefficients, _array_scales(coefficients)) and _tight(series, series_scales):
                inputs = {"x": x, "m_re": m_re, "m_im": m_im}
                return [
                    record(
                        generator, f"{name},coefficients", inputs | {"orders": ns}, coefficients
                    ),
                    record(
                        generator,
                        f"{name},series",
                        inputs | {"mu": mus, "n_ref": terms},
                        series,
                        label,
                    ),
                ]
        prec *= 2
    raise ArithmeticError(f"{name} needs more than {MAX_PRECISION_BITS} bits")


def value_records(case: Case) -> list[dict]:
    return _case_records(case, derivatives=False)


def derivative_records(case: Case) -> list[dict]:
    return _case_records(case, derivatives=True)


def cases() -> list[Case]:
    """The grid ``XS`` × ``MS`` and ``RANDOM_CASES`` random cases inside the target domain."""
    out = [(x, m_re, m_im) for x in XS for (m_re, m_im) in MS]
    rng = random.Random(SEED)
    for _ in range(RANDOM_CASES):
        x = nearest_pow10(rng.uniform(-6.0, 4.3))
        if rng.random() < 0.15:
            m_re, m_im = rng.uniform(0.05, 1.0), rng.uniform(1.0, 8.0)
        else:
            m_re = rng.uniform(1.0, 3.0)
            m_im = 0.0 if rng.random() < 0.2 else nearest_pow10(rng.uniform(-8.0, 0.5))
        out.append((x, m_re, m_im))
    return out


def derivative_cases() -> list[Case]:
    return [(x, m_re, m_im) for x in DERIVATIVE_XS for (m_re, m_im) in DERIVATIVE_MS]


def _map(fn: Callable[[Case], list[dict]], items: list[Case]) -> list[dict]:
    """``fn`` over ``items`` in worker processes, results in order.

    ``SCATRS_REFERENCE_WORKERS`` sets the number of processes (default: one per CPU; 1 runs
    in this process). The records do not depend on it.
    """
    workers = int(os.environ.get("SCATRS_REFERENCE_WORKERS", "0")) or os.cpu_count() or 1
    if workers == 1:
        results = [fn(c) for c in items]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(fn, items, chunksize=1))
    return [r for rs in results for r in rs]


def generate() -> tuple[int, list[dict]]:
    return PRECISION_BITS, _map(value_records, cases())


def generate_derivatives() -> tuple[int, list[dict]]:
    return PRECISION_BITS, _map(derivative_records, derivative_cases())
