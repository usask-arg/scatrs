"""Checks of the Mie generator against independent computations in Arb."""

import pytest
from flint import acb, arb, ctx

from scatrs_reference import disk
from scatrs_reference.generators import mie


@pytest.mark.parametrize(
    ("x", "m_re", "m_im"), [(0.7, 1.5, 0.01), (5.0, 2.5, 1.2), (4.0, 1.1, 0.3)]
)
def test_closed_form_derivatives_match_central_differences(x, m_re, m_im):
    # Central differences with step 2^-120 at 600 bits: truncation ~2^-240, rounding ~2^-480.
    with ctx.workprec(600):
        n = 12
        s = mie.solve(x, m_re, m_im, n)
        h = arb(2) ** -120

        def at(dx, dre):
            p = mie.solve_exact(arb(x) + dx, acb(arb(m_re) + dre, arb(m_im)), n)
            return p.a, p.b

        (ax_p, bx_p), (ax_n, bx_n) = at(h, 0), at(-h, 0)
        (am_p, bm_p), (am_n, bm_n) = at(0, h), at(0, -h)
        for k in range(n):
            pairs = [
                (s.da_dx[k], (ax_p[k] - ax_n[k]) / (2 * h)),
                (s.db_dx[k], (bx_p[k] - bx_n[k]) / (2 * h)),
                (s.da_dm[k], (am_p[k] - am_n[k]) / (2 * h)),
                (s.db_dm[k], (bm_p[k] - bm_n[k]) / (2 * h)),
            ]
            for got, fd in pairs:
                assert abs(got - fd) <= arb(2) ** -200 * abs(fd) + arb(2) ** -400


@pytest.mark.parametrize(
    ("x", "m_re", "m_im"),
    [(0.7, 1.5, 0.01), (30.0, 1.33, 1e-8), (8.0, 0.2, 3.0), (12.0, 1.5, 0.0)],
)
def test_absorption_identity(x, m_re, m_im):
    # For real m (the real recurrence for D_n) both sides vanish.
    # Re a_n − |a_n|² = −Im(D_n/m)/|Δ_a|², and likewise for b_n with m D_n.
    with ctx.workprec(400):
        s = mie.solve(x, m_re, m_im, 40)
        for k in range(40):
            a, b, d = s.a[k], s.b[k], s.d[k]
            lhs_a = a.real - abs(a) ** 2
            rhs_a = -(d / s.m).imag / abs(s.delta_a[k]) ** 2
            lhs_b = b.real - abs(b) ** 2
            rhs_b = -(s.m * d).imag / abs(s.delta_b[k]) ** 2
            assert abs(lhs_a - rhs_a) <= arb(2) ** -300 * (abs(a) + arb(2) ** -1000)
            assert abs(lhs_b - rhs_b) <= arb(2) ** -300 * (abs(b) + arb(2) ** -1000)


def test_disk_arithmetic_encloses_the_exact_results():
    # Exact centres and radii, and the boundary points where each bound is attained: a radius
    # below the true bound (for example r/|c|² for the inverse) misses them.
    with ctx.workprec(200):
        r, t = arb(2) ** -10, arb(2) ** -12
        c, e = acb(2), acb(3)
        d, f = disk.Disk(c, r), disk.Disk(e, t)
        w = acb("0.75", "-0.25")
        # 1/w is farthest from 1/c at the point of the disk nearest the origin.
        assert disk.ball(disk.inv(d)).contains(1 / (c - r))
        # |v w − c e| reaches |c| t + |e| r + r t at the far ends of both disks.
        assert disk.ball(disk.mul(d, f)).contains((c + r) * (e + t))
        for p in (c + r, c - r, c + acb(0, r), c - acb(0, r)):
            assert disk.ball(disk.inv(d)).contains(1 / p)
            assert disk.ball(disk.sub_from(w, d)).contains(w - p)
        with pytest.raises(ArithmeticError):
            disk.inv(disk.Disk(acb(0), arb(1)))
