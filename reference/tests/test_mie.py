"""Checks of the Mie generator against independent computations in Arb."""

import json
from fractions import Fraction
from pathlib import Path

import pytest
from flint import acb, arb, ctx, fmpq

from scatrs_reference import disk
from scatrs_reference.generators import mie

DATA = Path(__file__).resolve().parent.parent / "data"


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
    # Re a_n − |a_n|² = −Im(D_n/m)/|Δ_a|², and likewise for b_n with m D_n.
    # For real m (the real recurrence for D_n) both sides vanish.
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


def test_wiscombe_published_values():
    # Wiscombe (1979): x = 1000, m = 0.75, Q_sca = 1.997908, S(0) = 4.994770e5 + 1.336502e4 i
    # (conjugated: MIEV0 uses e^(+iwt)).
    with ctx.workprec(320):
        s = mie.solve(1000.0, 0.75, 0.0, 1100)
        sums = mie.sum_series(s, [1.0])
        assert abs(sums.q_sca - arb("1.997908")) < arb("1e-6")
        assert abs(sums.s1[0] - acb("4.994770e5", "1.336502e4")) < arb("0.1")


def test_tail_bound_dominates_the_next_terms():
    # The first four take the high-precision length; the last three sit at the thresholds of
    # the bound (N = ⌈|m|x⌉ + 1 twice, N = ⌈1.5x⌉ + 1 once), where U_z or q is largest.
    cases = [
        (1.0, 1.5, 0.01),
        (50.0, 1.33, 0.0),
        (20.0, 4.0, 0.01),
        (5.0, 0.2, 3.0),
        (50.0, 10.0, 0.0),
        (20.0, 10.0, 10.0),
        (500.0, 1.33, 0.0),
    ]
    for x, m_re, m_im in cases:
        terms, certify = mie.n_ref(x, m_re, m_im)
        assert certify
        with ctx.workprec(320):
            s = mie.solve(x, m_re, m_im, terms)
            tail = mie.tail_bounds(s)
            assert tail is not None and tail.ratio < 1
            longer = mie.solve(x, m_re, m_im, terms + 30)
            for n in range(terms + 1, terms + 31):
                size = abs(longer.a[n - 1]) + abs(longer.b[n - 1])
                assert size <= tail.amplitude * tail.ratio ** (n - terms)


def test_tail_bound_refuses_outside_its_hypotheses():
    with ctx.workprec(320):
        # N = 400 < |m|x = 500: the enclosure of u_k(mx) does not hold yet.
        assert mie.tail_bounds(mie.solve(50.0, 10.0, 0.0, 400)) is None


def test_certified_series_contain_a_longer_sum():
    # At N = 20 the tail is visible (about 1e-11 in Q_ext), so a missing or wrong tail fails.
    x, m_re, m_im, terms = 10.0, 1.5, 0.1, 20
    mus = [1.0, 0.0, -1.0]
    with ctx.workprec(320):
        s = mie.solve(x, m_re, m_im, terms)
        tail = mie.tail_bounds(s)
        assert tail is not None
        out = mie.series_outputs(s, mie.sum_series(s, mus), tail)
        longer = mie.solve(x, m_re, m_im, terms + 100)
        ref = mie.series_outputs(longer, mie.sum_series(longer, mus), None)
        bare = mie.series_outputs(s, mie.sum_series(s, mus), None)
        assert out is not None and ref is not None and bare is not None
        for name in out:
            for v, w in zip(out[name], ref[name], strict=True):
                assert v.contains(w), name
        # Without its tail the short sum misses the long one.
        assert not bare["q_ext"][0].overlaps(ref["q_ext"][0])


def test_amplitude_symmetries_and_backscatter():
    # S₁(0) = S₂(0), S₁(π) = −S₂(π) and Q_back = 4|S₁(π)|²/x², which ties the sign convention
    # of the backscattering sum to S₁.
    with ctx.workprec(320):
        for x, m_re, m_im in [(0.3, 1.33, 0.0), (7.0, 1.5, 0.1), (30.0, 0.2, 3.0)]:
            s = mie.solve(x, m_re, m_im, mie.n_ref(x, m_re, m_im)[0])
            sums = mie.sum_series(s, [1.0, -1.0])
            eps = arb(2) ** -250
            assert (sums.s1[0] - sums.s2[0]).abs_upper() <= eps * abs(sums.s1[0])
            assert (sums.s1[1] + sums.s2[1]).abs_upper() <= eps * abs(sums.s1[1])
            q_back = abs(sums.s_back) ** 2 / (s.x * s.x)
            from_s1 = 4 * abs(sums.s1[1]) ** 2 / (s.x * s.x)
            assert (q_back - from_s1).abs_upper() <= eps * q_back


def test_pi_tau_match_exact_rational_recurrences():
    # Bohren & Huffman (1983), Eq. 4.47, in exact rationals at the exact double cosines.
    with ctx.workprec(320):
        for mu in (0.375, -0.8125, 0.0, 1.0, -1.0, 0.1):
            q = fmpq(*Fraction(mu).as_integer_ratio())
            pis, taus = mie._pi_tau(mu, 60)
            pi_prev, pi = fmpq(0), fmpq(1)
            for n in range(1, 61):
                tau = n * q * pi - (n + 1) * pi_prev
                for got, exact in ((pis[n - 1], pi), (taus[n - 1], tau)):
                    want = arb(exact)
                    assert got.overlaps(want), (mu, n)
                    assert got.rad() <= arb(2) ** -250 * (abs(want) + 1), (mu, n)
                pi_prev, pi = pi, ((2 * n + 1) * q * pi - (n + 1) * pi_prev) / n


def test_optical_theorem():
    with ctx.workprec(320):
        for x, m_re, m_im in [(0.1, 1.33, 0.0), (7.0, 1.5, 0.1), (300.0, 0.2, 3.0)]:
            s = mie.solve(x, m_re, m_im, mie.n_ref(x, m_re, m_im)[0])
            sums = mie.sum_series(s, [1.0])
            q = 4 * sums.s1[0].real / (s.x * s.x)
            assert abs(q - sums.q_ext) <= arb(2) ** -250 * sums.q_ext


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


def test_records_follow_the_label_rules():
    for line in (DATA / "mie.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["case"].endswith(",coefficients"):
            assert rec["label"] == "certified"
        else:
            x, m_re, m_im = (rec["inputs"][k] for k in ("x", "m_re", "m_im"))
            terms, certify = mie.n_ref(x, m_re, m_im)
            assert rec["inputs"]["n_ref"] == terms
            assert (rec["label"] == "certified") == certify, rec["case"]
    for line in (DATA / "mie_derivatives.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        expected = "certified" if rec["case"].endswith(",coefficients") else "high-precision"
        assert rec["label"] == expected
