"""Arb reference values for Lorenz–Mie scattering by homogeneous spheres: the coefficients.

Conventions are those of Bohren & Huffman (1983), with time dependence e^(−iωt): the
refractive index m = n + ik has k ≥ 0 for absorbing spheres, and

    a_n = [(D_n(mx)/m + n/x) ψ_n(x) − ψ_{n−1}(x)] / [(D_n(mx)/m + n/x) ξ_n(x) − ξ_{n−1}(x)],

b_n likewise with m·D_n(mx) in place of D_n(mx)/m. Here ψ_n(ρ) = ρ j_n(ρ), ξ_n = ψ_n − iχ_n with
χ_n(ρ) = −ρ y_n(ρ), and D_n = ψ_n′/ψ_n. Each value is computed at the exact inputs.

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
"""

from __future__ import annotations

from dataclasses import dataclass

from flint import acb, arb, ctx

from scatrs_reference import disk


def _ceil_abs(v: arb | acb) -> int:
    return int(abs(v).abs_upper().ceil().unique_fmpz())


def ratios_real(x: arb, n_top: int) -> list[arb]:
    """u_k(x) = ψ_k(x)/ψ_{k−1}(x) for k = 1, …, n_top + 1, at index k (index 0 is unused)."""
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
    """u_k(z) = ψ_k(z)/ψ_{k−1}(z) for k = 1, …, n_top + 1, at index k, in disk arithmetic."""
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
    #: D_n(mx) and the denominators Δ_a, Δ_b of a_n and b_n.
    d: list[acb]
    delta_a: list[acb]
    delta_b: list[acb]
    #: T = ψ_N/ξ_N and R = ξ_{N−1}/ξ_N at N = n_ref, for the tail bound.
    t_last: acb
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
    s = Solution(xa, m, n_ref, [], [], [], [], [], [], [], [], [], acb(0), acb(0))
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
